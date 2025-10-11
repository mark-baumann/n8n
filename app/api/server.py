from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from app.graph import get_graph
from app.vectorstore.ingest import build_index
from app.logging_config import setup_logging
import logging

setup_logging()

app = FastAPI(title="LangGraph RAG Multi-Agent API")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
UPLOAD_DIR = Path("data/docs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/data", StaticFiles(directory=str(UPLOAD_DIR)), name="data")

# Load environment variables from .env early so clients (OpenAI etc.) see the keys
load_dotenv()

# Build the application graph (this will initialize ChatOpenAI which expects OPENAI_API_KEY)
graph = get_graph()

class ChatIn(BaseModel):
    message: str
    document: str | None = None
    thread_id: str | None = None

class ChatOut(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatOut)
def chat(req: ChatIn):
    try:
        # Use document-scoped thread by default
        thread_id = req.thread_id or (f"doc:{req.document}" if req.document else "default")

        # If a document is active, add a dynamic system context so agents ground responses
        state: dict = {"messages": [HumanMessage(content=req.message)]}
        if req.document:
            state["system"] = (
                "Kontext: Der Nutzer liest aktuell das Dokument '"
                + req.document
                + "'. Beantworte Fragen bevorzugt anhand dieses Dokuments.\n"
                "Wenn du das Tool 'retrieve' verwendest, gib das Argument 'source' mit dem Dateinamen des Dokuments an,\n"
                "z. B.: retrieve(query=..., k=4, source='"
                + req.document
                + "')."
            )
            # Zusätzlich den Dokument-Kontext als Feld mitgeben (Router nutzt dies)
            state["doc"] = req.document

        result = graph.invoke(
            state,
            config={"configurable": {"thread_id": thread_id}},
        )
        messages = result.get("messages", [])
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        answer = last_ai.content if last_ai else "No answer."
        return ChatOut(answer=answer)  # type: ignore
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}", exc_info=True)
        return {"answer": "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."}


@app.get("/", response_class=HTMLResponse)

def index(request: Request):
    return templates.TemplateResponse("library.html", {"request": request})

@app.get("/reader", response_class=HTMLResponse)
def reader(request: Request):
    return templates.TemplateResponse("reader.html", {"request": request})

@app.get("/documents")
def get_documents():
    files = [f.name for f in UPLOAD_DIR.iterdir() if f.is_file()]
    return {"documents": files}

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    with file_path.open("wb") as buffer:
        buffer.write(await file.read())
    # Nach Upload: Index neu aufbauen, damit das Dokument im RAG erscheint
    try:
        build_index()
    except Exception as e:
        logging.error(f"Fehler beim Neuaufbau des Index nach Upload: {e}")
    return {"filename": file.filename}
