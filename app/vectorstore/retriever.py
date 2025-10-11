from __future__ import annotations

import os

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from app.paths import get_index_dir


from app.vectorstore.embeddings import (
    build_hf_embeddings,
    build_openai_embeddings,
)

def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


def _embedding() -> Embeddings:
    provider = _env("EMBEDDINGS_PROVIDER", "huggingface").lower()
    if provider == "openai":
        model = _env("EMBEDDING_MODEL", "text-embedding-3-small")
        return build_openai_embeddings(model=model)
    else:
        model = _env("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return build_hf_embeddings(model=model)





def load_vectorstore() -> VectorStore:
    emb = _embedding()
    backend = _env("VECTORSTORE_BACKEND", "faiss").lower()

    if backend == "faiss":
        index_dir = str(get_index_dir())
        return FAISS.load_local(index_dir, emb, allow_dangerous_deserialization=True)
    

def get_retriever(k: int = 4):
    try:
        vs = load_vectorstore()
    except FileNotFoundError:
        class _EmptyRetriever:
            def invoke(self, _query: str):
                return []

        return _EmptyRetriever()

    return vs.as_retriever(search_kwargs={"k": k})
