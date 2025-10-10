# LangGraph RAG Multi‑Agent Chatbot (von Basics → Advanced)

Ein komplettes, lokal lauffähiges Projekt, das Schritt für Schritt von einem einfachen LLM‑Call
zu einem **zustandsbehafteten** (checkpointed) **Multi‑Agent**‑Chatbot mit **RAG** (lokaler FAISS‑Index) und optionaler **Websuche**
ausgebaut wird – basierend auf **LangGraph** + **LangChain**.

## Features
- ✅ **RAG** via lokalem FAISS-Index (Voreinstellung, offline-freundlich)
- ✅ **Multi‑Agent Routing** (`direct` | `rag` | `web`) via strukturierter LLM‑Entscheidung
- ✅ **Tool‑Aufrufe** ohne `prebuilt`-Abhängigkeit (robust gegen API‑Änderungen)
- ✅ **Memory / Checkpointing** via `MemorySaver` (optional: SQLite‑Saver)
- ✅ **CLI** (`python -m app.cli`) und **kleine FastAPI** (`uvicorn app.api.server:app --reload`)
- ✅ **Browser-Oberfläche** mit fokussiertem Upload, Vorschau & Chat in einem Screen
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
# In .env API-Keys eintragen (z. B. OPENAI_API_KEY für das LLM oder Embeddings, falls OpenAI gewählt)
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

> **Hinweis:** Standardmäßig nutzt der Index `text-embedding-3-small` von OpenAI – trage dazu Deinen `OPENAI_API_KEY` in `.env` ein.
> Sollte der Aufruf scheitern (z. B. wegen Proxy), fällt das System automatisch auf den lokalen Hashing-Embedder (`local-hash-768`) zurück.

## 3) Eigene Dokumente indizieren (RAG)
Die Weboberfläche enthält einen Upload-Dialog. Jedes hochgeladene PDF/Markdown/Text-Dokument wird
automatisch nach `data/docs/` gespeichert und der FAISS-Index unmittelbar aktualisiert – Du musst
also kein separates Kommando ausführen und kannst direkt danach chatten.

> 📂 Das Repository liefert absichtlich **keine Beispiel-Dokumente** mit. Lade Deine eigenen Dateien
> hoch oder lege sie manuell in `data/docs/` ab.

> 💡 Falls Du Dokumente außerhalb der Weboberfläche in `data/docs/` ablegst, kannst Du den Index bei
> Bedarf weiterhin manuell per `python -m app.vectorstore.ingest` erneuern. Für den normalen Upload-
> Workflow ist dieser Schritt jedoch nicht nötig.

Der Index wird unter `data/index/faiss/` abgelegt. Wenn OpenAI als Embedding-Provider konfiguriert ist,
fällt die Indizierung bei Erreichbarkeitsproblemen automatisch auf den Hashing-Embedder zurück.

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
- Lade Dein Dokument über den Upload-Button oben rechts.
- Die Vorschau öffnet PDFs direkt im integrierten Browser-Viewer (andere Dateien als Textauszug).
- Stelle Deine Fragen in der Chat-Leiste am unteren Rand – der Assistent antwortet mit Kontext aus dem aktuell angezeigten Dokument.

## 6) Routen & Agents
- **Router** (LLM mit strukturiertem Output) entscheidet: `direct` (direkt antworten), `rag` (Vektor‑Suche) oder `web` (Websuche).
- **RAG‑Agent**: nutzt den `retrieve`‑Tool (lokaler FAISS-Index) iterativ, bis genug Kontext vorhanden ist, dann Antwort mit Quellen.
- **Web‑Agent**: nutzt `web_search` (DuckDuckGo ohne Key oder Tavily – wenn Key gesetzt).

## 7) Visualisierung (optional)
Wenn `graphviz` installiert ist, kannst Du den Graph als PNG exportieren:
```bash
python -m app.graph_render
```
Bild wird unter `artifacts/graph.png` abgelegt.

## 8) Testschnelllauf
```bash
# (1) Index bauen (falls bereits Dateien in data/docs/ liegen)
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
