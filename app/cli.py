from __future__ import annotations
import argparse, os, sys
from typing import Iterable
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage
from app.graph import get_graph

def run_cli(initial: str | None):
    graph = get_graph()

    def turn(user_text: str):
        state = {"messages": [HumanMessage(content=user_text)]}
        result = graph.invoke(state)
        messages = result.get("messages", [])
        last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
        if last_ai:
            print(f"\nAssistant: {last_ai.content}\n")
        else:
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
    p.add_argument("--message", default=None, help="Einmalige Nachricht (ohne Loop)")
    args = p.parse_args()
    run_cli(args.message)
