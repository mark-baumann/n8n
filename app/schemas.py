from __future__ import annotations
from typing import Annotated, Literal, Optional, List, Dict, Any
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages

class AppState(TypedDict, total=False):
    # Nachrichtensequenz; add_messages hängt Messages an statt zu überschreiben
    messages: Annotated[list, add_messages]
    # Router-Entscheidung
    route: Optional[Literal["direct", "rag", "web", "clarify"]]
    # Zitationsliste (für RAG/Web-Agent)
    citations: List[Dict[str, Any]]
