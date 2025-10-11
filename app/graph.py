from __future__ import annotations
import os
import logging
from typing import Dict, Any, Literal, Callable
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import MessagesState
from langgraph.checkpoint.memory import MemorySaver

try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore
except Exception:
    SqliteSaver = None  # type: ignore

from app.schemas import AppState
from app.agents.tools import get_toolset
from app.agents.router import route_message

# Optional: map doc_id -> filename (for source filtering)
try:
    from app.api.docs_registry import get_filename as _get_doc_filename
except Exception:
    _get_doc_filename = lambda _x: None  # type: ignore


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

        # Enforce per-document restriction when a doc context exists
        if name in ("retrieve", "list_docs") and state is not None and isinstance(state, dict):
            doc_label = state.get("doc") or state.get("document")
            doc_id_val = state.get("doc_id")
            try:
                if not isinstance(args, dict):
                    args = {}
                if name == "retrieve":
                    # prefer exact doc_id restriction
                    if doc_id_val:
                        args.setdefault("doc_id", doc_id_val)
                        logging.debug(f"tool_call retrieve enforced doc_id={doc_id_val}")
                    # and add filename filter for extra safety
                    if doc_label:
                        src = _get_doc_filename(doc_label) if callable(_get_doc_filename) else None
                        if not src and isinstance(doc_label, str) and doc_label.lower().endswith(".pdf"):
                            src = doc_label
                        if src:
                            args.setdefault("source", src)
                            args.setdefault("source_exact", True)
                            logging.debug(f"tool_call retrieve enforced source={src} label={doc_label}")
                elif name == "list_docs" and doc_label:
                    src = _get_doc_filename(doc_label) if callable(_get_doc_filename) else None
                    if not src and isinstance(doc_label, str) and doc_label.lower().endswith(".pdf"):
                        src = doc_label
                    if src:
                        import os as _os
                        args.setdefault("filter", _os.path.basename(str(src)))
                        logging.debug(f"tool_call list_docs enforced filter for {src}")
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
        out.append(ToolMessage(content=str(result), tool_call_id=c.get("id", "toolcall")))
    return out


def _mk_assistant(system_prompt: str, tools_enabled: bool):
    base = ChatOpenAI(model=MODEL_NAME, temperature=0.2)
    tools, tool_map = get_toolset(include_web=None)
    llm = _bind(base, tools) if tools_enabled else base

    def node(state: AppState) -> dict:
        messages = state.get("messages", [])

        # sanitize unresolved tool call messages
        # Accept both typed message objects (AIMessage/ToolMessage) and legacy dict-style messages
        cleaned: list[BaseMessage | dict] = []
        for i, m in enumerate(messages):
            # helper to detect assistant/tool_call intent on either typed or dict messages
            def has_tool_calls(msg):
                try:
                    if isinstance(msg, AIMessage):
                        return bool(getattr(msg, "tool_calls", None) or getattr(msg, "additional_kwargs", {}).get("tool_calls"))
                    if isinstance(msg, dict) and msg.get("role") == "assistant":
                        # dict may carry tool_calls in an extra key
                        return bool(msg.get("tool_calls") or (msg.get("additional_kwargs") or {}).get("tool_calls"))
                except Exception:
                    return False
                return False

            def is_tool_message(msg):
                return isinstance(msg, ToolMessage) or (isinstance(msg, dict) and msg.get("role") == "tool")

            if has_tool_calls(m):
                nxt = messages[i + 1] if i + 1 < len(messages) else None
                # If next message is not a ToolMessage/dict-tool, drop this assistant message (unresolved tool call)
                if nxt is None or not is_tool_message(nxt):
                    logging.debug("Dropping assistant message with unresolved tool_calls to avoid invalid LLM payload.")
                    continue
            cleaned.append(m)

        extra_sys = state.get("system") if isinstance(state, dict) else None
        sys_content = system_prompt if not extra_sys else f"{system_prompt}\n\n{extra_sys}"
        msgs = [{"role": "system", "content": sys_content}] + [m for m in cleaned]
        ai = llm.invoke(msgs)
        return {"messages": [ai]}

    def call_tools(state: AppState) -> dict:
        messages = state.get("messages", [])
        if not messages:
            return {"messages": []}
        last = messages[-1]
        if isinstance(last, AIMessage) and (last.tool_calls or last.additional_kwargs.get("tool_calls")):
            tool_messages = _exec_toolcalls(tool_map, last, state)  # type: ignore
            return {"messages": tool_messages}

        # Fallback: enforce a retrieve call in doc context
        doc_label = None
        doc_id_val = None
        if isinstance(state, dict):
            doc_label = state.get("doc") or state.get("document")
            doc_id_val = state.get("doc_id")
        if doc_label or doc_id_val:
            user_text = ""
            for m in reversed(messages):
                if isinstance(m, HumanMessage):
                    user_text = m.content  # type: ignore
                    break
            args: Dict[str, Any] = {"query": user_text}
            if doc_id_val:
                args["doc_id"] = doc_id_val
            else:
                try:
                    src = _get_doc_filename(doc_label) if callable(_get_doc_filename) else None
                    if not src and isinstance(doc_label, str) and doc_label.lower().endswith(".pdf"):
                        src = doc_label
                    if src:
                        args["source"] = src
                        args["source_exact"] = True
                except Exception:
                    pass
            try:
                retrieve = tool_map.get("retrieve")
                result = retrieve.invoke(args) if hasattr(retrieve, "invoke") else retrieve(**args)  # type: ignore
                return {"messages": [ToolMessage(content=str(result), tool_call_id="auto-retrieve")]}
            except Exception as e:  # type: ignore
                return {"messages": [ToolMessage(content=f"Toolfehler: {e}", tool_call_id="auto-retrieve")]}
        return {"messages": []}

    def after(state: AppState) -> Literal["tools", "__end__"]:
        # Doc-Modus: Erst Tools erzwingen, dann nach einer ToolMessage genau eine AI-Antwort und beenden.
        if isinstance(state, dict) and (state.get("doc") or state.get("document") or state.get("doc_id")):
            msgs = state.get("messages", []) if isinstance(state, dict) else []
            # Wenn schon mindestens eine ToolMessage in der Historie vorkommt, beenden (die AI-Antwort danach ist final)
            if any(isinstance(m, ToolMessage) for m in msgs):
                return "__end__"
            return "tools"
        # Global/ohne Doc: nur auf explizite Tool-Calls reagieren
        return "tools" if _should_call_tools(state.get("messages", [])) else "__end__"

    return node, call_tools, after


def build_graph():
    def router(state: AppState) -> dict:
        # force RAG for global or doc context
        if isinstance(state, dict):
            if state.get("global_rag"):
                return {"route": "rag"}
            doc_ctx = state.get("doc") or state.get("document")
            if doc_ctx:
                return {"route": "rag"}

        user_text = ""
        for msg in reversed(state.get("messages", [])):
            if hasattr(msg, "type") and getattr(msg, "type") == "human":
                user_text = msg.content  # type: ignore
                break
            if isinstance(msg, dict) and msg.get("role") == "user":
                user_text = msg.get("content", "")
                break
        decision = route_message(user_text)
        return {"route": decision.route}

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
            "Always call the 'retrieve' tool BEFORE answering (filters may be pre-filled), and base the answer only on retrieved text. "
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

    graph = StateGraph(MessagesState)
    graph.add_node("router", router)

    graph.add_node("direct", direct_node)
    graph.add_node("direct_tools", direct_tools)
    graph.add_node("rag", rag_node)
    graph.add_node("rag_tools", rag_tools)
    graph.add_node("web", web_node)
    graph.add_node("web_tools", web_tools)

    graph.add_edge(START, "router")

    def route_edge(state: AppState) -> Literal["direct", "rag", "web", "__end__"]:
        route = state.get("route") or "direct"
        if route not in {"direct", "rag", "web"}:
            return "direct"
        return route

    graph.add_conditional_edges("router", route_edge, {"direct": "direct", "rag": "rag", "web": "web"})

    graph.add_conditional_edges("direct", direct_after, {"tools": "direct_tools", "__end__": END})
    graph.add_edge("direct_tools", "direct")

    graph.add_conditional_edges("rag", rag_after, {"tools": "rag_tools", "__end__": END})
    graph.add_edge("rag_tools", "rag")

    graph.add_conditional_edges("web", web_after, {"tools": "web_tools", "__end__": END})
    graph.add_edge("web_tools", "web")

    checkpointer = _make_checkpointer()
    return graph.compile(checkpointer=checkpointer)


def get_graph():
    return build_graph()
