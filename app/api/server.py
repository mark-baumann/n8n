from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Dict, List

try:  # aiofiles ist optional – wir fallen bei Bedarf auf Sync-I/O zurück
    import aiofiles  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - fallback
    aiofiles = None  # type: ignore

from fastapi import File, FastAPI, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from app.graph import get_graph
from app.vectorstore.ingest import build_index

app = FastAPI(title="LangGraph RAG Multi-Agent API")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

DOCS_DIR = Path(os.getenv("DOCS_DIR", "data/docs"))
DOCS_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=str(DOCS_DIR)), name="uploads")

ALLOWED_SUFFIXES = {".pdf", ".md", ".txt"}
TEXT_SUFFIXES = {".md", ".txt"}

graph = get_graph()

class ChatIn(BaseModel):
    thread_id: str = "default"
    message: str

class ChatOut(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatOut)
def chat(req: ChatIn):
    config = {"configurable": {"thread_id": req.thread_id}}
    result = graph.invoke({"messages": [HumanMessage(content=req.message)]}, config=config)
    messages = result.get("messages", [])
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    answer = last_ai.content if last_ai else "No answer."
    return ChatOut(answer=answer)  # type: ignore


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def _sanitize_filename(name: str) -> str:
    base = Path(name or "upload").name
    cleaned = re.sub(r"[^A-Za-z0-9_.-]", "_", base)
    return cleaned or "upload"


async def _save_upload(file: UploadFile, destination: Path) -> int:
    size = 0

    if aiofiles is not None:
        async with aiofiles.open(destination, "wb") as buffer:  # type: ignore[attr-defined]
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                await buffer.write(chunk)
    else:  # pragma: no cover - wird nur ohne aiofiles benötigt
        with destination.open("wb") as buffer:
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                buffer.write(chunk)

    await file.close()
    return size


def _collect_preview(path: Path) -> str | None:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return None
    return text[:4000]


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> Dict[str, object]:
    original_name = file.filename or "upload"
    candidate = _sanitize_filename(original_name)
    suffix = Path(candidate).suffix.lower()

    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail="Nur PDF-, Markdown- oder Textdateien werden unterstützt.",
        )

    stem = Path(candidate).stem
    destination = DOCS_DIR / candidate
    counter = 1
    while destination.exists():
        destination = DOCS_DIR / f"{stem}_{counter}{suffix}"
        counter += 1

    size = await _save_upload(file, destination)
    await run_in_threadpool(build_index)

    preview = _collect_preview(destination)

    return {
        "filename": destination.name,
        "url": f"/uploads/{destination.name}",
        "size": size,
        "content_type": file.content_type,
        "preview": preview,
    }


@app.get("/documents")
def list_documents() -> Dict[str, List[Dict[str, object]]]:
    documents: List[Dict[str, object]] = []
    for path in sorted(DOCS_DIR.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_file():
            continue
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        stat = path.stat()
        documents.append(
            {
                "filename": path.name,
                "url": f"/uploads/{path.name}",
                "size": stat.st_size,
                "updated_at": stat.st_mtime,
                "suffix": path.suffix.lower(),
                "preview": _collect_preview(path),
            }
        )
    return {"documents": documents}
