# LangGraph RAG Multi‑Agent Chatbot (von Basics → Advanced)

Ein komplettes, lokal lauffähiges Projekt, das Schritt für Schritt von einem einfachen LLM‑Call
zu einem **zustandsbehafteten** (checkpointed) **Multi‑Agent**‑Chatbot mit **RAG** (Qdrant Cloud oder FAISS) und optionaler **Websuche**
ausgebaut wird – basierend auf **LangGraph** + **LangChain**.

## Features
- ✅ **RAG** via Qdrant Cloud (Standard) oder lokalem FAISS-Fallback (`VECTORSTORE_BACKEND`)
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

### Docker-Variante

Alternativ kannst Du das komplette Backend inklusive Weboberfläche per Docker starten:

```bash
cp .env.example .env  # gewünschte Keys/Settings eintragen
docker compose up --build
```

Der FastAPI-Server läuft anschließend unter http://127.0.0.1:8000 (Frontend ist im selben Container eingebettet).
Dokumente und Artefakte werden aus dem lokalen `data/`- bzw. `artifacts/`-Ordner in den Container gemountet, sodass
Du sie bequem bearbeiten kannst.

> **Hinweis:** Standardmäßig nutzt das Projekt OpenAI‑Modelle. Alternativ kannst Du **HuggingFace**‑Embeddings wählen,
> dann brauchst Du keinen OpenAI‑Key für die Vektordatenbank (nur fürs LLM, sofern Du ein OpenAI‑LLM nutzt).

## 3) Eigene Dokumente indizieren (RAG)
Lege Deine Dateien unter `data/docs/` ab (unterstützt: `.md`, `.txt`, `.pdf`). Danach:
```bash
python -m app.vectorstore.ingest
```
Standardmäßig wird dabei eine Qdrant‑Collection aufgebaut (`VECTORSTORE_BACKEND=qdrant`).
Hinterlege dazu in `.env` Deine Qdrant‑Cloud‑URL inkl. Port (`:6333`), den API‑Key und
optional einen Collection‑Namen (`QDRANT_COLLECTION`).

> **Tipp (Docker):** `docker compose run --rm app python -m app.vectorstore.ingest`

Wenn Du lieber lokal arbeiten möchtest, setze `VECTORSTORE_BACKEND=faiss`. Dann wird ein
persistenter Index unter `data/index/faiss/` abgelegt. Falls der direkte Aufruf der
OpenAI‑API durch einen Proxy blockiert ist, wechsle per `EMBEDDINGS_PROVIDER=huggingface`
auf Sentence‑Transformer.

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

### Weboberfläche nutzen
- Öffne im Browser: http://127.0.0.1:8000/
- Wähle eine Thread-ID (z. B. `demo`) oder erzeuge eine neue.
- Stelle Deine Fragen – Antworten erscheinen gemeinsam mit dem Chatverlauf.

## 6) Routen & Agents
- **Router** (LLM mit strukturiertem Output) entscheidet: `direct` (direkt antworten), `rag` (Vektor‑Suche) oder `web` (Websuche).
- **RAG‑Agent**: nutzt den `retrieve`‑Tool (Qdrant oder FAISS) iterativ, bis genug Kontext vorhanden ist, dann Antwort mit Quellen.
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
  index/                   # Persistenter FAISS-Index (nur bei VECTORSTORE_BACKEND=faiss)
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
