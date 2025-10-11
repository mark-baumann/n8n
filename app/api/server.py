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

class ChatOut(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatOut)
def chat(req: ChatIn):
    try:
        result = graph.invoke({"messages": [HumanMessage(content=req.message)]})
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
    return {"filename": file.filename}
