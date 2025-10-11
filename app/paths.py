from __future__ import annotations
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent  # project root (ChatMitPDF)


def project_root() -> Path:
    return _ROOT


def resolve_project_path(p: str | os.PathLike) -> Path:
    path = Path(p)
    if not path.is_absolute():
        path = (_ROOT / path).resolve()
    return path


def get_docs_dir() -> Path:
    d = os.getenv("DOCS_DIR", "data/docs")
    return resolve_project_path(d)


def get_index_dir() -> Path:
    d = os.getenv("INDEX_DIR", "data/index/faiss")
    return resolve_project_path(d)

