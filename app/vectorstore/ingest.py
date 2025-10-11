from __future__ import annotations

import os
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.vectorstore.embeddings import build_hf_embeddings, build_openai_embeddings
from app.paths import get_docs_dir, get_index_dir
try:
    from app.api.docs_registry import add_document as _add_doc
except Exception:
    _add_doc = None  # type: ignore
import shutil


def _env(key: str, default: str) -> str:
    return os.getenv(key, default)


DOCS_DIR = str(get_docs_dir())
CHUNK_SIZE = int(_env("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(_env("CHUNK_OVERLAP", "150"))


def _embedding():
    provider = _env("EMBEDDINGS_PROVIDER", "huggingface").lower()
    if provider == "openai":
        model = _env("EMBEDDING_MODEL", "text-embedding-3-small")
        return build_openai_embeddings(model=model)
    else:
        model = _env("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        return build_hf_embeddings(model=model)


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

    # Annotate metadata with file_name and doc_id (stable)
    for d in docs:
        try:
            src = (d.metadata or {}).get("source") or (d.metadata or {}).get("file_path") or ""
            base = os.path.basename(str(src))
            d.metadata = dict(d.metadata or {})
            d.metadata["file_name"] = base
            if _add_doc is not None:
                try:
                    doc_id = _add_doc(base)
                    d.metadata["doc_id"] = doc_id
                except Exception:
                    pass
        except Exception:
            pass

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
        index_dir = str(get_index_dir())
        index_path = Path(index_dir)
        if index_path.exists():
            shutil.rmtree(index_path)
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    emb = _embedding()
    backend = _env("VECTORSTORE_BACKEND", "faiss").lower()

    if backend == "faiss":
        try:
            vs = FAISS.from_documents(chunks, emb)
        except Exception as exc:
            raise RuntimeError(
                "Konnte den FAISS-Index nicht aufbauen. Prüfe bitte, ob die Embedding-API "
                "erreichbar ist (z. B. Proxy-Konfiguration) oder wechsle per "
                "EMBEDDINGS_PROVIDER=huggingface auf lokale Modelle."
            ) from exc
        index_dir = _env("INDEX_DIR", "data/index/faiss")
        os.makedirs(index_dir, exist_ok=True)
        vs.save_local(index_dir)
        print(
            f"[OK] FAISS-Index gespeichert unter: {index_dir}  (Chunks: {len(chunks)})"
        )
        return

    raise ValueError(
        f"Unbekannter VECTORSTORE_BACKEND '{backend}'. Erlaubt: faiss"
    )


if __name__ == "__main__":
    build_index()
