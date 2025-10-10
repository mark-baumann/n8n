from __future__ import annotations

import os

from langchain_community.vectorstores import FAISS, Qdrant
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from qdrant_client import QdrantClient

from app.vectorstore.embeddings import (
    build_hf_embeddings,
    build_openai_embeddings,
)

INDEX_DIR = os.getenv("INDEX_DIR", "data/index/faiss")
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai")
OPENAI_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
HF_EMBED_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
VECTORSTORE_BACKEND = os.getenv("VECTORSTORE_BACKEND", "faiss").lower()


def _embedding() -> Embeddings:
    if EMBEDDINGS_PROVIDER.lower() == "openai":
        return build_openai_embeddings(model=OPENAI_EMBED_MODEL)
    else:
        return build_hf_embeddings(model=HF_EMBED_MODEL)


def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL:
        raise RuntimeError(
            "QDRANT_URL ist nicht gesetzt. Hinterlege die Qdrant-Cloud-URL in der .env Datei."
        )
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


def load_vectorstore() -> VectorStore:
    emb = _embedding()
    backend = VECTORSTORE_BACKEND

    if backend == "faiss":
        return FAISS.load_local(
            INDEX_DIR, emb, allow_dangerous_deserialization=True
        )
    

def get_retriever(k: int = 4):
    vs = load_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})
