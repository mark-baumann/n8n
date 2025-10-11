from __future__ import annotations
import os, json
from typing import Dict, Any, Literal, Callable
from typing_extensions import TypedDict
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver
try:
    # optional persistent saver
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
except Exception:
    SqliteSaver = None  # type: ignore

from app.schemas import AppState
from app.agents.tools import get_toolset
from app.vectorstore.retriever import get_retriever
try:
    # for mapping doc_id -> filename
    from app.api.docs_registry import get_filename as _get_doc_filename
except Exception:
    _get_doc_filename = lambda _x: None  # type: ignore
from app.agents.router import route_message

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")

def _make_checkpointer():
    backend = os.getenv("CHECKPOINTER_BACKEND", "memory").lower()
    if backend == "sqlite" and SqliteSaver is not None:
        path = os.getenv("SQLITE_PATH", "data/checkpoints/langgraph.sqlite")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import sqlite3
        conn = sqlite3.connect(path)
        return SqliteSaver(conn)
    return MemorySaver()

def _bind(llm: ChatOpenAI, tools):
    # Tools an das Modell binden (Tool-Calling)
    return llm.bind_tools(tools)

def _should_call_tools(messages: list[BaseMessage]) -> bool:
    if not messages:
        return False
    last = messages[-1]
    if isinstance(last, AIMessage) and (last.tool_calls or last.additional_kwargs.get("tool_calls")):
        return True
    return False

def _exec_toolcalls(tool_map: Dict[str, Callable[..., Any]], ai_msg: AIMessage, state: dict | None = None) -> list[ToolMessage]:
    out: list[ToolMessage] = []
    calls = ai_msg.tool_calls or ai_msg.additional_kwargs.get("tool_calls") or []
    for c in calls:
        name = c.get("name") or c.get("function", {}).get("name")
        args = c.get("args") or c.get("function", {}).get("arguments") or {}
        if isinstance(args, str):
            try:
                import json as _json
                args = _json.loads(args)
            except Exception:
                args = {"input": args}
        # Enforce per-document restriction for retrieve/list_docs when a document context exists
        if name in ("retrieve", "list_docs") and state is not None:
            try:
                doc_label = state.get("doc") or state.get("document")
                if doc_label:
                    src_file = _get_doc_filename(doc_label) if callable(_get_doc_filename) else None
                    if not src_file and isinstance(doc_label, str) and doc_label.lower().endswith(".pdf"):
                        src_file = doc_label
                    if src_file:
                        if not isinstance(args, dict):
                            args = {}
                        if name == "retrieve":
                            args.setdefault("source", src_file)
                            args.setdefault("source_exact", True)
                        elif name == "list_docs":
                            # apply filename filter
                            args.setdefault("filter", os.path.basename(str(src_file)))
            except Exception:
                pass
        fn = tool_map.get(name)
        if fn is None:
            result = f"Tool '{name}' nicht gefunden."
        else:
            try:
                result = fn.invoke(args) if hasattr(fn, "invoke") else fn(**args)
            except TypeError:
                result = fn.invoke({"query": args}) if hasattr(fn, "invoke") else fn(args)
            except Exception as e:
                result = f"Toolfehler: {e}"
        out.append(ToolMessage(content=str(result), tool_call_id=c.get("id","toolcall")))
    return out

def _mk_assistant(system_prompt: str, tools_enabled: bool):
    base = ChatOpenAI(model=MODEL_NAME, temperature=0.2)
    tools, tool_map = get_toolset(include_web=None)  # env entscheidet
    llm = _bind(base, tools) if tools_enabled else base

    def node(state: AppState) -> dict:
        messages = state.get("messages", [])

        # Sanitize history: remove any assistant message with tool_calls
        # that is not immediately followed by a ToolMessage. This avoids
        # OpenAI Chat API 400 errors on invalid sequences.
        cleaned: list[BaseMessage] = []
        for i, m in enumerate(messages):
            if isinstance(m, AIMessage) and (getattr(m, "tool_calls", None) or getattr(m, "additional_kwargs", {}).get("tool_calls")):
                nxt = messages[i + 1] if i + 1 < len(messages) else None
                if nxt is None or not isinstance(nxt, ToolMessage):
                    # skip unresolved tool-call assistant messages
                    continue
            cleaned.append(m)

        # prepend base system prompt + optional dynamic system context from state
        extra_sys = state.get("system") if isinstance(state, dict) else None
        sys_content = system_prompt if not extra_sys else f"{system_prompt}\n\n{extra_sys}"

        # Optionale RAG-Vorabrretrieval: Wenn ein Dokument gesetzt ist, hole Kontextextrakte
        context_msgs = []
        try:
            doc_label = state.get("doc") if isinstance(state, dict) else None
            # Ermittele Dateiname Ã¼ber Registry, falls ID Ã¼bergeben wurde
            src_file = None
            if doc_label:
                src_file = _get_doc_filename(doc_label) or (doc_label if str(doc_label).lower().endswith(".pdf") else None)

            if src_file:
                # letzte Userfrage bestimmen
                user_text = ""
                for msg in reversed(cleaned):
                    if isinstance(msg, HumanMessage):
                        user_text = msg.content  # type: ignore
                        break
                if user_text:
                    retriever = get_retriever(k=6)
                    docs = retriever.invoke(user_text)
                    # Filter nur dieses PDF
                    import os as _os
                    base_src = _os.path.basename(str(src_file)).lower()
                    def _match(d):
                        meta = getattr(d, 'metadata', {}) or {}
                        s = (meta.get('source') or meta.get('file_path') or '')
                        b = _os.path.basename(str(s)).lower()
                        return b == base_src or base_src in b
                    docs = [d for d in docs if _match(d)]
                    if docs:
                        # Kompakter Kontext
                        parts = []
                        for i, d in enumerate(docs, 1):
                            text = (getattr(d, 'page_content', '') or '').strip()
                            if not text:
                                continue
                            parts.append(f"[{i}] {text[:900]}")
                        if parts:
                            context_text = (
                                f"KontextauszÃ¼ge aus {base_src} fÃ¼r die Beantwortung der Frage:\n\n" + "\n\n".join(parts)
                            )
                            context_msgs.append({"role": "system", "content": context_text})
        except Exception:
            pass

        msgs = ([{"role":"system","content": sys_content}] + context_msgs + [m for m in cleaned])
        ai = llm.invoke(msgs)
        return {"messages": [ai]}
    def call_tools(state: AppState) -> dict:
        # fÃ¼hrt Tools aus und hÃ¤ngt ToolMessages an
        messages = state.get("messages", [])
        last = messages[-1]
        tool_messages = _exec_toolcalls(tool_map, last, state)  # type: ignore
        return {"messages": tool_messages}
    def after(state: AppState) -> Literal["tools", "__end__"]:
        return "tools" if _should_call_tools(state.get("messages", [])) else "__end__"
    return node, call_tools, after

def build_graph():
    # Router entscheidet die Route
    def router(state: AppState) -> dict:
        # Erzwinge RAG, wenn global_rag oder doc-Kontext gesetzt ist
        if isinstance(state, dict):
            if state.get("global_rag"):
                return {"route": "rag"}
            doc_ctx = state.get("doc") or state.get("document")
            if doc_ctx:
                return {"route": "rag"}

        # Sonst: Nimm die letzte User-Nachricht und route per LLM
        user_text = ""
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and getattr(msg, "type") == "human":
                user_text = msg.content  # type: ignore
                break
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_text = msg.get("content","")
                break
        decision = route_message(user_text)
        return {"route": decision.route}

    # Drei Agenten
    direct_node, direct_tools, direct_after = _mk_assistant(
        system_prompt=(
            "You are a helpful assistant. Answer directly and concisely. "
            "Do not fabricate sources. Avoid generic greetings; respond to the task immediately."
        ),
        tools_enabled=False,
    )
    rag_node, rag_tools, rag_after = _mk_assistant(
        system_prompt=(
            "You are a RAG agent. Answer strictly grounded in the indexed documents. "
            "If a document context is present, treat ALL questions as referring ONLY to that document. "
            "Do not include information from other documents. Do not rely on general knowledge. "
            "Always use the 'retrieve' tool to fetch passages (filters may be pre-filled to that document). "
            "For vague prompts like 'um was geht es hier', produce a concise summary of the current document: topic, purpose, key points, structure. "
            "For overviews across the corpus (no document context), you may use 'list_docs' and 'retrieve' without filters. "
            "Cite sources as [n] when available. Avoid generic greetings and avoid asking clarifying questions first."
        ),
        tools_enabled=True,
    )
    web_node, web_tools, web_after = _mk_assistant(
        system_prompt=(
            "You are a web search agent. Prefer the 'web_search' tool for current facts. "
            "Summarize and include short source list. Avoid generic greetings."
        ),
        tools_enabled=True,
    )

    graph = StateGraph(MessagesState)  # messages reducer aktiv
    graph.add_node("router", router)

    graph.add_node("direct", direct_node)
    graph.add_node("direct_tools", direct_tools)
    graph.add_node("rag", rag_node)
    graph.add_node("rag_tools", rag_tools)
    graph.add_node("web", web_node)
    graph.add_node("web_tools", web_tools)

    # Entry: Router
    graph.add_edge(START, "router")

    # Conditional routing von Router
    def route_edge(state: AppState) -> Literal["direct","rag","web","__end__"]:
        route = state.get("route") or "direct"
        if route not in {"direct","rag","web"}:
            return "direct"
        return route
    graph.add_conditional_edges("router", route_edge, {"direct":"direct","rag":"rag","web":"web"})

    # Loops pro Agent (Tool-Calls, dann zurÃ¼ck)
    graph.add_conditional_edges("direct", direct_after, {"tools": "direct_tools", "__end__": END})
    graph.add_edge("direct_tools", "direct")

    graph.add_conditional_edges("rag", rag_after, {"tools": "rag_tools", "__end__": END})
    graph.add_edge("rag_tools", "rag")

    graph.add_conditional_edges("web", web_after, {"tools": "web_tools", "__end__": END})
    graph.add_edge("web_tools", "web")

    checkpointer = _make_checkpointer()
    return graph.compile(checkpointer=checkpointer)

# Exponiere eine globale Fabrik
def get_graph():
    return build_graph()

