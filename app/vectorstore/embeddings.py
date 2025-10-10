from __future__ import annotations

from typing import List, Sequence

from langchain_core.embeddings import Embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings


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


def build_hf_embeddings(*, model: str) -> Embeddings:
    return HuggingFaceEmbeddings(model_name=model)
