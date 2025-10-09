# LangGraph RAG Multi‑Agent Chatbot (von Basics → Advanced)

Ein komplettes, lokal lauffähiges Projekt, das Schritt für Schritt von einem einfachen LLM‑Call
zu einem **zustandsbehafteten** (checkpointed) **Multi‑Agent**‑Chatbot mit **RAG** (FAISS) und optionaler **Websuche**
ausgebaut wird – basierend auf **LangGraph** + **LangChain**.

## Features
- ✅ **RAG** mit FAISS (persistiert in `data/index/faiss`)
- ✅ **Multi‑Agent Routing** (`direct` | `rag` | `web`) via strukturierter LLM‑Entscheidung
- ✅ **Tool‑Aufrufe** ohne `prebuilt`-Abhängigkeit (robust gegen API‑Änderungen)
- ✅ **Memory / Checkpointing** via `MemorySaver` (optional: SQLite‑Saver)
- ✅ **CLI** (`python -m app.cli`) und **kleine FastAPI** (`uvicorn app.api.server:app --reload`)
- ✅ **Konfigurierbar** via `.env` (OpenAI / HF‑Embeddings, Websuche an/aus, Modelle)

---

## 1) Voraussetzungen
- Python **3.11**+ empfohlen
- `pip` bzw. `uv`

## 2) Setup
```bash
git clone <dieses-Archiv-oder-zip-entpacken>
cd langgraph_rag_multiagent
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# In .env API-Keys eintragen (mind. OPENAI_API_KEY für LLM + Embeddings, falls OpenAI gewählt)
```

> **Hinweis:** Standardmäßig nutzt das Projekt OpenAI‑Modelle. Alternativ kannst Du **HuggingFace**‑Embeddings wählen,
> dann brauchst Du keinen OpenAI‑Key für die Vektordatenbank (nur fürs LLM, sofern Du ein OpenAI‑LLM nutzt).

## 3) Eigene Dokumente indizieren (RAG)
Lege Deine Dateien unter `data/docs/` ab (unterstützt: `.md`, `.txt`, `.pdf`). Danach:
```bash
python -m app.vectorstore.ingest
```
Dadurch wird ein persistenter FAISS‑Index unter `data/index/faiss/` angelegt.

## 4) Chatten (CLI)
```bash
python -m app.cli --thread demo
```
- `--thread` identifiziert die Konversations-Session (wichtig für Memory/Checkpoints).

## 5) (Optional) API starten
```bash
uvicorn app.api.server:app --reload
# POST http://127.0.0.1:8000/chat  JSON: {"thread_id":"demo", "message":"<Deine Frage>"}
```

## 6) Routen & Agents
- **Router** (LLM mit strukturiertem Output) entscheidet: `direct` (direkt antworten), `rag` (Vektor‑Suche) oder `web` (Websuche).
- **RAG‑Agent**: nutzt den `retrieve`‑Tool (FAISS) iterativ, bis genug Kontext vorhanden ist, dann Antwort mit Quellen.
- **Web‑Agent**: nutzt `web_search` (DuckDuckGo ohne Key oder Tavily – wenn Key gesetzt).

## 7) Visualisierung (optional)
Wenn `graphviz` installiert ist, kannst Du den Graph als PNG exportieren:
```bash
python -m app.graph_render
```
Bild wird unter `artifacts/graph.png` abgelegt.

## 8) Testschnelllauf
```bash
# (1) Index bauen (mit den mitgelieferten Beispieldateien)
python -m app.vectorstore.ingest

# (2) Kurzer CLI-Test
python -m app.cli --thread quick --message "Worum geht es in den Beispieldokumenten?"

# (3) Websuche aktivieren (optional)
# In .env: ENABLE_WEBSEARCH=true
python -m app.cli --thread quick --message "Was ist neu in LangGraph im Jahr 2025?"
```

---

## Projektstruktur
```
app/
  api/server.py            # Kleine FastAPI für HTTP-Chat
  agents/
    tools.py               # Retriever-Tool + Websuche-Tool
    router.py              # LLM-Router (structured output)
  graph.py                 # Bau des Multi-Agent Graphen
  graph_render.py          # PNG-Export des Graphen (optional)
  models/llm.py            # LLM-Initialisierung (OpenAI, streaming-ready)
  prompts/                 # Systemprompts
  schemas.py               # State (TypedDict) + Reducer
  vectorstore/
    ingest.py              # Index erstellen
    retriever.py           # Index laden → Retriever
  cli.py                   # CLI-Einstieg
data/
  docs/                    # Deine Dokumente für RAG
  index/                   # Persistenter FAISS-Index
artifacts/                 # (optional) Graph-Bild, Dumps
.env.example               # Konfigurationsbeispiel
requirements.txt
README.md
```

---

## Hinweise
- Standardmäßig **MemorySaver** (in‑memory) als Checkpointer. Für Persistenz:
  setze `CHECKPOINTER_BACKEND=sqlite` in `.env` (benötigt `langgraph-checkpoint-sqlite`).
- Websuche ohne Key nutzt `duckduckgo_search`. Für **Tavily** setze `TAVILY_API_KEY` und `WEBSEARCH_BACKEND=tavily`.

Viel Spaß! 🚀
