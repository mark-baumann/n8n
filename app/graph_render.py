from __future__ import annotations
import os
from app.graph import build_graph

def main():
    g = build_graph()
    try:
        os.makedirs("artifacts", exist_ok=True)
        g.get_graph(xray=True).draw_png("artifacts/graph.png")
        print("[OK] Graph gespeichert unter artifacts/graph.png")
    except Exception as e:
        print("Installiere 'graphviz' um die Visualisierung zu erzeugen. Fehler:", e)

if __name__ == "__main__":
    main()
