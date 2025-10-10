from __future__ import annotations

import os

from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


from app.vectorstore.embeddings import (
    build_hf_embeddings,
    build_openai_embeddings,
)

INDEX_DIR = os.getenv("INDEX_DIR", "data/index/faiss")
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "huggingface")
OPENAI_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
HF_EMBED_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTORSTORE_BACKEND = os.getenv("VECTORSTORE_BACKEND", "faiss").lower()


def _embedding() -> Embeddings:
    if EMBEDDINGS_PROVIDER.lower() == "openai":
        return build_openai_embeddings(model=OPENAI_EMBED_MODEL)
    else:
        return build_hf_embeddings(model=HF_EMBED_MODEL)





def load_vectorstore() -> VectorStore:
    emb = _embedding()
    backend = VECTORSTORE_BACKEND

    if backend == "faiss":
        return FAISS.load_local(
            INDEX_DIR, emb, allow_dangerous_deserialization=True
        )
    

def get_retriever(k: int = 4):
    try:
        vs = load_vectorstore()
    except FileNotFoundError:
        class _EmptyRetriever:
            def invoke(self, _query: str):
                return []

        return _EmptyRetriever()

    return vs.as_retriever(search_kwargs={"k": k})
