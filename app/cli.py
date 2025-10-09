from __future__ import annotations
import argparse, os, sys
from typing import Iterable
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from app.graph import get_graph

def run_cli(thread_id: str, initial: str | None):
    graph = get_graph()
    config = {"configurable":{"thread_id": thread_id}}

    def turn(user_text: str):
        state = {"messages": [HumanMessage(content=user_text)]}
        # Für Einfachheit: Wir holen nur den finalen Output (kein token-stream)
        result = graph.invoke(state, config=config)
        # Letzte AI-Nachricht extrahieren
        messages = result.get("messages", [])
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai:
            print(f"\nAssistant: {last_ai.content}\n")
        else:
            # Fallback: Alles ausgeben
            print(result)

    if initial:
        turn(initial)
    print("Tippe 'exit' zum Beenden.")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not user or user.lower() in {"exit", "quit"}:
            break
        turn(user)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--thread", default="default", help="Konversations-ID (Memory/Checkpoint Key)")
    p.add_argument("--message", default=None, help="Einmalige Nachricht (ohne Loop)")
    args = p.parse_args()
    run_cli(args.thread, args.message)
