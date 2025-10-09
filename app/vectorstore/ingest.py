from __future__ import annotations
import os, glob
from pathlib import Path
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from app.vectorstore.retriever import INDEX_DIR, EMBEDDINGS_PROVIDER, OPENAI_EMBED_MODEL, HF_EMBED_MODEL

DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

def _embedding():
    if EMBEDDINGS_PROVIDER.lower() == "openai":
        return OpenAIEmbeddings(model=OPENAI_EMBED_MODEL)
    else:
        return HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)

def _load_documents() -> List[Document]:
    docs: List[Document] = []
    base = Path(DOCS_DIR)
    base.mkdir(parents=True, exist_ok=True)

    for path in base.rglob("*"):
        if path.is_dir():
            continue
        p = str(path)
        if p.lower().endswith((".md", ".txt")):
            docs.extend(TextLoader(p, autodetect_encoding=True).load())
        elif p.lower().endswith(".pdf"):
            docs.extend(PyPDFLoader(p).load())

    if not docs:
        # Fallback: Beispielcontent
        default_md = base / "example.md"
        if not default_md.exists():
            default_md.write_text(
                "# Beispiel

Dies ist ein Beispiel-Dokument für das RAG-System.
"
                "Es beschreibt, wie das Projekt aufgebaut ist und dient als Test.
",
                encoding="utf-8"
            )
        docs.extend(TextLoader(str(default_md), autodetect_encoding=True).load())

    return docs

def build_index():
    docs = _load_documents()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    emb = _embedding()
    vs = FAISS.from_documents(chunks, emb)
    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"[OK] FAISS-Index gespeichert unter: {INDEX_DIR}  (Chunks: {len(chunks)})")

if __name__ == "__main__":
    build_index()
