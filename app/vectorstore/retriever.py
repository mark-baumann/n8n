from __future__ import annotations
import os
from typing import List, Optional, Tuple
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document

INDEX_DIR = os.getenv("INDEX_DIR", "data/index/faiss")
EMBEDDINGS_PROVIDER = os.getenv("EMBEDDINGS_PROVIDER", "openai")
OPENAI_EMBED_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
HF_EMBED_MODEL = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

def _embedding() -> Embeddings:
    if EMBEDDINGS_PROVIDER.lower() == "openai":
        return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    else:
        return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)

def load_vectorstore() -> FAISS:
    emb = _embedding()
    vs = FAISS.load_local(
        INDEX_DIR, emb, allow_dangerous_deserialization=True
    )
    return vs

def get_retriever(k: int = 4):
    vs = load_vectorstore()
    return vs.as_retriever(search_kwargs={"k": k})
