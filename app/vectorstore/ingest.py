from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vectorstore.embeddings import build_hf_embeddings, build_openai_embeddings
from app.vectorstore.retriever import (
    EMBEDDINGS_PROVIDER,
    HF_EMBED_MODEL,
    INDEX_DIR,
    OPENAI_EMBED_MODEL,
    VECTORSTORE_BACKEND,
)
import shutil

DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


def _embedding():
    if EMBEDDINGS_PROVIDER.lower() == "openai":
        return build_openai_embeddings(model=OPENAI_EMBED_MODEL)
    else:
        return build_hf_embeddings(model=HF_EMBED_MODEL)


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
                "# Beispiel\n\n"
                "Dies ist ein Beispiel-Dokument für das RAG-System.\n"
                "Es beschreibt, wie das Projekt aufgebaut ist und dient als Test.\n",
                encoding="utf-8",
            )
        docs.extend(TextLoader(str(default_md), autodetect_encoding=True).load())

    return docs


def build_index():
    docs = _load_documents()
    if not docs:
        print(
            "[INFO] Keine Dokumente zum Indizieren gefunden – vorhandener Index wird entfernt.",
            flush=True,
        )
        index_path = Path(INDEX_DIR)
        if index_path.exists():
            shutil.rmtree(index_path)
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    emb = _embedding()
    backend = VECTORSTORE_BACKEND

    if backend == "faiss":
        try:
            vs = FAISS.from_documents(chunks, emb)
        except Exception as exc:
            raise RuntimeError(
                "Konnte den FAISS-Index nicht aufbauen. Prüfe bitte, ob die Embedding-API "
                "erreichbar ist (z. B. Proxy-Konfiguration) oder wechsle per "
                "EMBEDDINGS_PROVIDER=huggingface auf lokale Modelle."
            ) from exc
        os.makedirs(INDEX_DIR, exist_ok=True)
        vs.save_local(INDEX_DIR)
        print(
            f"[OK] FAISS-Index gespeichert unter: {INDEX_DIR}  (Chunks: {len(chunks)})"
        )
        return

    

    raise ValueError(
        f"Unbekannter VECTORSTORE_BACKEND '{VECTORSTORE_BACKEND}'. Erlaubt: faiss"
    )


if __name__ == "__main__":
    build_index()
