from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import List, Sequence

from langchain_core.embeddings import Embeddings


class SimpleOpenAIEmbeddings(Embeddings):
    """Lightweight OpenAI embedding wrapper that avoids tiktoken downloads."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        batch_size: int = 16,
    ) -> None:
        try:
            from openai import OpenAI  # local import to avoid mandatory dependency
        except ImportError as exc:  # pragma: no cover - informative error path
            raise RuntimeError(
                "The 'openai' package is required for OpenAI embeddings. Install it via"
                " `pip install openai` or include it in requirements."
            ) from exc

        self._client = OpenAI()
        self._model = model
        self._batch_size = max(1, batch_size)

    @staticmethod
    def _clean_texts(texts: Sequence[str]) -> List[str]:
        return [t.replace("\n", " ") for t in texts]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        cleaned = self._clean_texts(texts)
        results: List[List[float]] = []

        for start in range(0, len(cleaned), self._batch_size):
            batch = cleaned[start : start + self._batch_size]
            response = self._client.embeddings.create(
                model=self._model,
                input=list(batch),
            )
            results.extend([item.embedding for item in response.data])
        return results

    def embed_query(self, text: str) -> List[float]:
        response = self._client.embeddings.create(
            model=self._model,
            input=[text.replace("\n", " ")],
        )
        return response.data[0].embedding


def build_openai_embeddings(*, model: str) -> Embeddings:
    return SimpleOpenAIEmbeddings(model=model)


class LocalHashingEmbeddings(Embeddings):
    """Kleiner, lokaler Bag-of-Words-Hashing-Embedder ohne externe Downloads."""

    def __init__(self, *, dim: int = 768) -> None:
        self._dim = max(128, int(dim))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def _embed(self, text: str) -> List[float]:
        tokens = self._tokenize(text)
        if not tokens:
            return [0.0] * self._dim

        counts = Counter(tokens)
        vector = [0.0] * self._dim

        for token, count in counts.items():
            digest = hashlib.sha1(token.encode("utf-8")).hexdigest()
            index = int(digest, 16) % self._dim
            vector[index] += float(count)

        norm = math.sqrt(sum(v * v for v in vector))
        if norm:
            vector = [v / norm for v in vector]
        return vector

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def _parse_hash_dim(model: str) -> int:
    try:
        return int(model.rsplit("-", 1)[-1])
    except (ValueError, IndexError):
        return 768


def build_hf_embeddings(*, model: str) -> Embeddings:
    dim = _parse_hash_dim(model)
    return LocalHashingEmbeddings(dim=dim)
