from __future__ import annotations
import os
from typing import Literal
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

ROUTER_MODEL = os.getenv("ROUTER_MODEL", os.getenv("MODEL_NAME", "gpt-4o-mini"))

class RouteDecision(BaseModel):
    route: Literal["direct", "rag", "web", "clarify"] = Field(..., description="Selected path")
    reason: str = Field(..., description="Short rationale")

def route_message(user_text: str) -> RouteDecision:
    llm = ChatOpenAI(model=ROUTER_MODEL, temperature=0)
    structured = llm.with_structured_output(RouteDecision)
    system = (
        "You are a Router for a multi-agent chatbot. "
        "Decide the best route: direct | rag | web | clarify."
    )
    result = structured.invoke([{"role":"system","content":system},
                                {"role":"user","content":user_text}])
    return result
