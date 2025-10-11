from __future__ import annotations
import json
import os
import uuid
from pathlib import Path
from typing import Dict, List, Optional

DOCS_DIR = Path(os.getenv("DOCS_DIR", "data/docs"))
REGISTRY_PATH = DOCS_DIR / "registry.json"


def _load() -> Dict[str, str]:
    try:
        with REGISTRY_PATH.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
    except FileNotFoundError:
        pass
    return {}


def _save(mapping: Dict[str, str]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)


def ensure_registry() -> Dict[str, str]:
    mapping = _load()
    # only track PDFs for the reader
    existing = {p.name for p in DOCS_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"}
    # add missing files
    known_files = set(mapping.values())
    for fname in sorted(existing):
        if fname not in known_files and fname != REGISTRY_PATH.name:
            # new file: assign id
            doc_id = uuid.uuid4().hex
            mapping[doc_id] = fname
    # remove entries for non-existing files
    ids_to_remove = [doc_id for doc_id, fname in mapping.items() if fname not in existing]
    if ids_to_remove:
        for k in ids_to_remove:
            mapping.pop(k, None)
    _save(mapping)
    return mapping


def add_document(filename: str) -> str:
    mapping = ensure_registry()
    for k, v in mapping.items():
        if v == filename:
            return k
    doc_id = uuid.uuid4().hex
    mapping[doc_id] = filename
    _save(mapping)
    return doc_id


def get_filename(doc_id: str) -> Optional[str]:
    mapping = ensure_registry()
    return mapping.get(doc_id)


def list_documents() -> List[dict]:
    mapping = ensure_registry()
    out: List[dict] = []
    for doc_id, filename in mapping.items():
        # only expose PDFs
        if str(filename).lower().endswith(".pdf"):
            out.append({"id": doc_id, "filename": filename})
    # Stable order by filename
    out.sort(key=lambda d: d["filename"].lower())
    return out
