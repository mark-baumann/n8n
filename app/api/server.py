from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request
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

# Load environment variables from .env early so clients (OpenAI etc.) see the keys
load_dotenv()

# Build the application graph (this will initialize ChatOpenAI which expects OPENAI_API_KEY)
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
