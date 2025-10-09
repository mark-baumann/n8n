from __future__ import annotations
import os
from typing import Optional, Sequence
from pydantic import BaseModel
from langchain_openai import ChatOpenAI

class LLMConfig(BaseModel):
    model: str = os.getenv("MODEL_NAME", "gpt-4o-mini")
    temperature: float = 0.2
    streaming: bool = False

def get_chat_model(cfg: Optional[LLMConfig] = None) -> ChatOpenAI:
    cfg = cfg or LLMConfig()
    # Hinweis: output_version / responses API wird automatisch gewählt,
    # LangChain kapselt die Parameter.
    llm = ChatOpenAI(model=cfg.model, temperature=cfg.temperature)
    return llm
