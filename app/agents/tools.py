from __future__ import annotations
import os, json, time
from typing import Callable, List, Dict, Any, Optional
from langchain_core.tools import tool
from duckduckgo_search import DDGS
from app.vectorstore.retriever import get_retriever

ENABLE_WEBSEARCH = os.getenv("ENABLE_WEBSEARCH", "false").lower() == "true"
WEBSEARCH_BACKEND = os.getenv("WEBSEARCH_BACKEND", "duckduckgo").lower()

def _format_docs(docs, max_chars: int = 1800) -> str:
    out = []
    for i, d in enumerate(docs, 1):
        meta = d.metadata or {}
        src = meta.get("source") or meta.get("file_path") or "source"
        out.append(f"[{i}] ({src})\n{d.page_content[:max_chars]}")
    return "\n\n".join(out)

def _ddg_search(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    with DDGS() as ddgs:
        # news=False → allgemeine Suche
        results = list(ddgs.text(query, max_results=max_results))
    return results

@tool("retrieve", return_direct=False)
def retrieve_tool(query: str, k: int = 4, source: str | None = None, source_exact: bool = False) -> str:
    """Rufe relevante Passagen aus dem lokalen FAISS-Vektorindex ab und liefere formatierte Auszüge mit Quellen."""
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)
    if not docs:
        return "Keine Dokumente im Index. Lade zuerst ein Dokument hoch."
    if source:
        import os as _os
        src_lower = source.lower()
        def _match(d):
            meta = getattr(d, 'metadata', {}) or {}
            src = meta.get("source") or meta.get("file_path") or ""
            base = _os.path.basename(str(src))
            if source_exact:
                return base.lower() == src_lower or str(src).lower() == src_lower
            return (src_lower in str(src).lower()) or (src_lower in base.lower())
        docs = [d for d in docs if _match(d)]
        if not docs:
            return "Keine Treffer im gewählten Dokument."
    return _format_docs(docs)

@tool("web_search", return_direct=False)
def web_search_tool(query: str, max_results: int = 5) -> str:
    """Websuche (DuckDuckGo ohne API-Key). Liefert eine kompakte Ergebnisliste (Titel, Link, Snippet)."""
    if not ENABLE_WEBSEARCH:
        return "Websuche ist deaktiviert. Setze ENABLE_WEBSEARCH=true in .env"
    if WEBSEARCH_BACKEND == "duckduckgo":
        results = _ddg_search(query, max_results=max_results)
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.get('title')}\n{r.get('href')}\n{r.get('body')}")
        return "\n\n".join(lines)
    else:
        try:
            from langchain_community.tools.tavily_search import TavilySearchResults
        except Exception:
            return "Tavily nicht installiert. Installiere langchain_community Extras oder nutze duckduckgo."
        tavily = TavilySearchResults(max_results=max_results)
        return "\n\n".join(
            f"[{i}] {res['title']}\n{res['url']}\n{res.get('content','')}" 
            for i, res in enumerate(tavily.invoke({"query": query}), 1)
        )

def get_toolset(include_web: Optional[bool] = None):
    include_web = ENABLE_WEBSEARCH if include_web is None else include_web
    tools = [retrieve_tool]
    if include_web:
        tools.append(web_search_tool)
    # Mapping für manuelle Toolausführung
    tool_map: Dict[str, Callable[..., Any]] = {t.name: t for t in tools}
    return tools, tool_map
