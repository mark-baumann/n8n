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

def _exec_toolcalls(tool_map: Dict[str, Callable[..., Any]], ai_msg: AIMessage) -> list[ToolMessage]:
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
        # prepend system
        msgs = [{"role":"system","content":system_prompt}] + [m for m in messages]
        ai = llm.invoke(msgs)
        return {"messages": [ai]}
    def call_tools(state: AppState) -> dict:
        # führt Tools aus und hängt ToolMessages an
        messages = state.get("messages", [])
        last = messages[-1]
        tool_messages = _exec_toolcalls(tool_map, last)  # type: ignore
        return {"messages": tool_messages}
    def after(state: AppState) -> Literal["tools", "__end__"]:
        return "tools" if _should_call_tools(state.get("messages", [])) else "__end__"
    return node, call_tools, after

def build_graph():
    # Router entscheidet die Route
    def router(state: AppState) -> dict:
        # Nimm die letzte User-Nachricht
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
            "Do not fabricate sources."
        ),
        tools_enabled=False,
    )
    rag_node, rag_tools, rag_after = _mk_assistant(
        system_prompt=(
            "You are a RAG agent. When you need external info, call the 'retrieve' tool. "
            "Cite sources as [n] if provided in tool results."
        ),
        tools_enabled=True,
    )
    web_node, web_tools, web_after = _mk_assistant(
        system_prompt=(
            "You are a web search agent. Prefer the 'web_search' tool for current facts. "
            "Summarize and include short source list."
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

    # Loops pro Agent (Tool-Calls, dann zurück)
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
