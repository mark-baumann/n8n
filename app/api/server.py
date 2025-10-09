from __future__ import annotations
import os
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from app.graph import get_graph

app = FastAPI(title="LangGraph RAG Multi-Agent API")

graph = get_graph()

class ChatIn(BaseModel):
    thread_id: str = "default"
    message: str

class ChatOut(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatOut)
def chat(req: ChatIn):
    config = {"configurable":{"thread_id": req.thread_id}}
    result = graph.invoke({"messages":[HumanMessage(content=req.message)]}, config=config)
    messages = result.get("messages", [])
    last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    answer = last_ai.content if last_ai else "No answer."
    return ChatOut(answer=answer)  # type: ignore
