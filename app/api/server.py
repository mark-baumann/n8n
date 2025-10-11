from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, Request, File, UploadFile
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from app.graph import get_graph
from app.vectorstore.ingest import build_index
from app.logging_config import setup_logging
from app.paths import get_docs_dir
from app.api.docs_registry import ensure_registry, add_document, get_filename, list_documents
from app.agents.tools import retrieve_tool
import logging

setup_logging()

app = FastAPI(title="LangGraph RAG Multi-Agent API")

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
UPLOAD_DIR = get_docs_dir()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
app.mount("/data", StaticFiles(directory=str(UPLOAD_DIR)), name="data")

# Load environment variables from .env early so clients (OpenAI etc.) see the keys
load_dotenv()

# Build the application graph (this will initialize ChatOpenAI which expects OPENAI_API_KEY)
graph = get_graph()

# Ensure document registry is in sync on startup
try:
    ensure_registry()
except Exception as _e:
    pass

class ChatIn(BaseModel):
    message: str
    # Backwards-compat: old param by filename
    document: str | None = None
    # Preferred: document id
    document_id: str | None = None
    thread_id: str | None = None

class ChatOut(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatOut)
def chat(req: ChatIn):
    try:
        # Resolve filename from id if provided
        file_from_id = None
        if req.document_id:
            file_from_id = get_filename(req.document_id)

        # Use document-scoped thread by default (prefer id)
        thread_id = req.thread_id or (
            f"doc:{req.document_id}" if req.document_id else (
                f"doc:{req.document}" if req.document else "default"
            )
        )

        # If a document is active, add a dynamic system context so agents ground responses
        state: dict = {"messages": [HumanMessage(content=req.message)]}
        if req.document_id or req.document:
            label = req.document_id or req.document
            preferred_source = file_from_id or req.document
            logging.debug(f"/chat doc-context label={label} resolved_source={preferred_source} thread={thread_id}")
            sys_lines = [
                f"Kontext: Der Nutzer liest aktuell das Dokument '{label}'.",
                "Beantworte Fragen AUSSCHLIESSLICH anhand dieses Dokuments.",
                "Regeln:",
                "1) Nutze das Tool 'retrieve' (gefiltert auf dieses Dokument) um Passagen zu holen.",
                "2) Keine Inhalte aus anderen Dokumenten oder Weltwissen einmischen.",
                "3) Wenn keine passenden Passagen im aktuellen Dokument gefunden werden: antworte kurz: 'Keine Treffer im aktuellen Dokument.'",
            ]
            if preferred_source:
                sys_lines.append(
                    "Wenn du das Tool 'retrieve' verwendest, gib das Argument 'source' mit dem Dateinamen des Dokuments an,"
                )
                sys_lines.append(
                    f"z. B.: retrieve(query=..., k=4, source='{preferred_source}')."
                )
            state["system"] = "\n".join(sys_lines)
            # Router-Hinweis
            state["doc"] = req.document_id or req.document
            if req.document_id:
                state["doc_id"] = req.document_id
        else:
            # Global-Dokumenten-Chat: explizit RAG über alle Dokumente bevorzugen
            state["global_rag"] = True

        # Serverseitiger Doc-RAG-Fallback: Wenn doc_id/source gesetzt sind, hole Passagen direkt und beantworte strikt daraus.
        answer: str | None = None
        if req.document_id or req.document:
            try:
                tool_args = {"query": req.message, "k": 6}
                if req.document_id:
                    tool_args["doc_id"] = req.document_id
                if file_from_id or req.document:
                    tool_args["source"] = file_from_id or req.document
                    tool_args["source_exact"] = True
                retrieved = retrieve_tool.invoke(tool_args)  # type: ignore
                # Baue den Kontext primär direkt aus dem PDF
                context_text: str | None = None
                base_from_pdf: str | None = None
                if file_from_id:
                    try:
                        from langchain_community.document_loaders import PyPDFLoader  # type: ignore
                        pdf_path = UPLOAD_DIR / file_from_id
                        pages = PyPDFLoader(str(pdf_path)).load()
                        base_from_pdf = "\n\n".join(p.page_content for p in pages[:10])
                    except Exception:
                        base_from_pdf = None
                # Ergänze optional Retrieval-Snippets aus genau diesem Dokument
                extra = None
                if isinstance(retrieved, str) and retrieved.strip() and not retrieved.strip().lower().startswith("keine treffer"):
                    extra = retrieved
                # Kontext zusammenbauen (PDF-Inhalt hat Priorität)
                parts = []
                if base_from_pdf:
                    parts.append(base_from_pdf[:6000])
                if extra:
                    parts.append(extra[:2000])
                context_text = "\n\n".join(parts) if parts else None
                # Kurze, strikte Antwort aus Kontext erzeugen
                if context_text:
                    llm = ChatOpenAI(model=os.getenv("MODEL_NAME", "gpt-4o-mini"), temperature=0)
                    sys = (
                        "Antworte ausschließlich anhand des folgenden Kontexts zum aktuellen PDF. "
                        "Erfinde nichts. Wenn der Kontext die Frage nicht beantwortet, antworte: 'Keine Treffer im aktuellen Dokument.'"
                    )
                    prompt = [
                        {"role": "system", "content": sys},
                        {"role": "user", "content": f"Kontext:\n{context_text}\n\nFrage:\n{req.message}"},
                    ]
                    ai = llm.invoke(prompt)
                    answer = ai.content if hasattr(ai, "content") else str(ai)
            except Exception as _e:
                # Fallback auf Graph unten
                pass

        if answer is None:
            result = graph.invoke(
                state,
                config={"configurable": {"thread_id": thread_id}},
            )
            messages = result.get("messages", [])
            last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
            answer = last_ai.content if last_ai else "No answer."
        return ChatOut(answer=answer)  # type: ignore
    except Exception as e:
        logging.error(f"Error in chat endpoint: {e}", exc_info=True)
        return {"answer": "Es ist ein Fehler aufgetreten. Bitte versuchen Sie es später erneut."}


@app.get("/", response_class=HTMLResponse)

def index(request: Request):
    return templates.TemplateResponse("library.html", {"request": request})

@app.get("/reader", response_class=HTMLResponse)
def reader(request: Request):
    return templates.TemplateResponse("reader.html", {"request": request})

@app.get("/documents")
def get_documents():
    docs = list_documents()
    return {"documents": docs}

@app.get("/document/{doc_id}")
def get_document(doc_id: str):
    fname = get_filename(doc_id)
    if not fname:
        return JSONResponse(status_code=404, content={"error": "Dokument nicht gefunden"})
    return {"id": doc_id, "filename": fname}

@app.get("/data_by_id/{doc_id}")
def data_by_id(doc_id: str):
    fname = get_filename(doc_id)
    if not fname:
        return JSONResponse(status_code=404, content={"error": "Dokument nicht gefunden"})
    file_path = UPLOAD_DIR / fname
    if not file_path.exists():
        return JSONResponse(status_code=404, content={"error": "Datei nicht gefunden"})
    # Serve inline so browsers can render PDFs in <embed> instead of forcing download
    return FileResponse(path=str(file_path), media_type="application/pdf")

@app.post("/reindex")
def reindex():
    try:
        build_index()
        ensure_registry()
        return {"status": "ok"}
    except Exception as e:
        logging.error(f"Reindex failed: {e}")
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    file_path = UPLOAD_DIR / file.filename
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    with file_path.open("wb") as buffer:
        buffer.write(await file.read())
    if not file_path.exists() or file_path.stat().st_size == 0:
        logging.error(f"Upload failed or empty file: {file_path}")
        return JSONResponse(status_code=500, content={"error": "Upload fehlgeschlagen."})
    # Nach Upload: Index neu aufbauen, damit das Dokument im RAG erscheint
    try:
        doc_id = add_document(file.filename)
        logging.info(f"Uploaded PDF saved at: {file_path}")
        build_index()
    except Exception as e:
        logging.error(f"Fehler beim Neuaufbau des Index nach Upload: {e}")
    return {"filename": file.filename, "id": doc_id}
