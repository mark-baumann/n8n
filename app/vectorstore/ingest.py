from __future__ import annotations

import os
import shutil
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
)

DOCS_DIR = os.getenv("DOCS_DIR", "data/docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))


def _embedding():
    provider = EMBEDDINGS_PROVIDER.lower()
    if provider == "openai":
        try:
            return build_openai_embeddings(model=OPENAI_EMBED_MODEL)
        except Exception as exc:
            print(
                "[WARN] Konnte OpenAI-Embeddings nicht initialisieren. Fallback auf "
                "Hashing-Embeddings.",
                flush=True,
            )
            print(f"        Grund: {exc}", flush=True)
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
    provider_used = EMBEDDINGS_PROVIDER.lower()

    try:
        vs = FAISS.from_documents(chunks, emb)
    except Exception as exc:
        if EMBEDDINGS_PROVIDER.lower() == "openai":
            print(
                "[WARN] Fehler beim Erzeugen des Index mit OpenAI-Embeddings. "
                "Versuche Hashing-Embeddings...",
                flush=True,
            )
            try:
                emb = build_hf_embeddings(model=HF_EMBED_MODEL)
                vs = FAISS.from_documents(chunks, emb)
                provider_used = "huggingface"
            except Exception as hf_exc:  # pragma: no cover - defensive path
                raise RuntimeError(
                    "Konnte den FAISS-Index nicht aufbauen. Prüfe bitte, ob die "
                    "Embedding-Backends verfügbar sind."
                ) from hf_exc
        else:
            raise RuntimeError(
                "Konnte den FAISS-Index nicht aufbauen. Prüfe bitte, ob die "
                "Embedding-API erreichbar ist (z. B. Proxy-Konfiguration)."
            ) from exc
    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(
        f"[OK] FAISS-Index gespeichert unter: {INDEX_DIR}  (Chunks: {len(chunks)})\n"
        f"     Verwendeter Embedding-Provider: {provider_used}"
    )
    return


if __name__ == "__main__":
    build_index()
