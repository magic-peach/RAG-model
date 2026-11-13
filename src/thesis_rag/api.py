"""FastAPI application: ingest, ask (SSE), analyze, status, health; serves web/.

    uvicorn thesis_rag.api:app            # or: python -m thesis_rag.api

Endpoints
---------
GET  /healthz   liveness for Docker; never touches Ollama or the index
GET  /status    document name, index counts, ingest progress, Ollama state (ok/busy/offline)
POST /ingest    upload a PDF (or name a server-side path); builds the index in a
                background thread and reports progress through /status
POST /ask       {"question", "chat_id"?} -> Server-Sent Events: "chat" (when a new
                chat was created), "mode" (thesis | general), "token" and
                "reasoning" while streaming, one "answer" with sources, "done"
POST /analyze   CSV upload -> statistics, then the grounded explanation
GET/POST/PATCH/DELETE /chats[/{id}]
                saved chats, scoped to the browser's X-Client-Id header

Heavy objects (index, reranker, LLM client) are created on first use and
rebuilt after an ingest, so the server starts instantly and tests can swap
them for fakes via :func:`create_app`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .config import Settings, get_settings

log = logging.getLogger(__name__)

WEB_DIR = Path(os.environ.get("RAG_WEB_DIR", Path(__file__).resolve().parents[2] / "web"))
MAX_PDF_BYTES = 100_000_000


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    chat_id: str | None = None


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=80)


_CLIENT_ID = re.compile(r"^[A-Za-z0-9-]{8,64}$")


def _owner(client_id: str | None, required: bool = True) -> str | None:
    """The browser's random id (see store.py); malformed or missing ids are refused."""
    if client_id and _CLIENT_ID.match(client_id):
        return client_id
    if required:
        raise HTTPException(400, "Missing or invalid X-Client-Id header")
    return None


class IngestJob:
    """State of the (single) background ingestion, polled by GET /status."""

    def __init__(self):
        self.lock = threading.Lock()
        self.state = "idle"  # idle | running | done | error
        self.message = ""
        self.progress = 0.0
        self.started_at: float | None = None
        self.finished_at: float | None = None
        self.error = ""

    def to_dict(self) -> dict:
        return {
            "state": self.state,
            "message": self.message,
            "progress": round(self.progress, 3),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "error": self.error,
        }


class AppState:
    def __init__(
        self,
        settings: Settings,
        qa_factory: Callable[[Settings], object] | None = None,
        build_fn: Callable | None = None,
    ):
        self.settings = settings
        self._qa_factory = qa_factory
        self._build_fn = build_fn
        self._qa = None
        self._qa_lock = threading.Lock()
        self._store = None
        self.job = IngestJob()

    @property
    def store(self):
        if self._store is None:
            from .store import ChatStore

            location = self.settings.chats_db or self.settings.data_dir / "chats.db"
            self._store = ChatStore(location, self.settings.chats_db_token)
        return self._store

    def warm_up(self) -> None:
        """Load the index, reranker and model before the first question arrives."""
        if not self.indexed():
            return
        t0 = time.perf_counter()
        try:
            qa = self.qa()
            qa.retriever.retrieve("warm up")
            qa.llm.invoke("Reply with OK.")
            log.info("Warm-up finished in %.1f s", time.perf_counter() - t0)
        except Exception:  # warm-up is best effort; the first question will retry
            log.warning("Warm-up failed", exc_info=True)

    # --- lazy QA pipeline -------------------------------------------------
    def indexed(self) -> bool:
        from .index import Index

        return Index.exists(self.settings)

    def qa(self):
        with self._qa_lock:
            if self._qa is None:
                if self._qa_factory is not None:
                    self._qa = self._qa_factory(self.settings)
                else:
                    from .generate import ThesisQA

                    self._qa = ThesisQA(self.settings)
            return self._qa

    def reset_qa(self) -> None:
        with self._qa_lock:
            self._qa = None

    def manifest(self) -> dict | None:
        path = self.settings.index_dir / "manifest.json"
        return json.loads(path.read_text()) if path.is_file() else None

    # --- ingestion --------------------------------------------------------
    def start_ingest(self, pdf_path: Path, force: bool) -> None:
        job = self.job
        with job.lock:
            if job.state == "running":
                raise HTTPException(409, "An ingestion is already running")
            job.state, job.message, job.progress, job.error = "running", "Starting", 0.0, ""
            job.started_at, job.finished_at = time.time(), None

        def progress(msg: str, frac: float) -> None:
            job.message, job.progress = msg, frac

        def run() -> None:
            build = self._build_fn
            if build is None:
                from .index import build_index as build
            try:
                build(pdf_path, self.settings, force=force, progress=progress)
                self.reset_qa()  # next question loads the new index
                job.state, job.message, job.progress = "done", "Indexed", 1.0
            except Exception as exc:  # report any failure through /status
                log.exception("Ingestion failed")
                job.state, job.error = "error", str(exc)
            finally:
                job.finished_at = time.time()

        threading.Thread(target=run, name="ingest", daemon=True).start()


def _ollama_status(base_url: str, timeout: float = 4.0) -> str:
    """'ok', 'busy' (no reply in time: usually mid-generation on a slow machine) or 'offline'."""
    try:
        with urllib.request.urlopen(f"{base_url}/api/tags", timeout=timeout) as resp:
            return "ok" if resp.status == 200 else "offline"
    except TimeoutError:
        return "busy"
    except urllib.error.URLError as exc:
        return "busy" if isinstance(exc.reason, TimeoutError) else "offline"
    except Exception:
        return "offline"


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def create_app(
    settings: Settings | None = None,
    qa_factory: Callable[[Settings], object] | None = None,
    build_fn: Callable | None = None,
    check_ollama: Callable[[str], str] = _ollama_status,
    warm_up: bool | str = False,
) -> FastAPI:
    state = AppState(settings or get_settings(), qa_factory, build_fn)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if warm_up == "startup":
            # Finish loading before accepting traffic. On Cloud Run the CPU is
            # throttled outside requests, so a background warm-up would barely
            # run and the first question would pay for it; during startup the
            # instance gets full (boosted) CPU, and the port only opens once
            # the models are ready.
            await asyncio.to_thread(state.warm_up)
        elif warm_up:  # in a thread, so the server accepts requests immediately
            threading.Thread(target=state.warm_up, name="warm-up", daemon=True).start()
        yield

    app = FastAPI(title="Thesis RAG", version="2.1.0", lifespan=lifespan)
    app.state.rag = state

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/status")
    def status() -> dict:
        from .analyze import load_knowledge_card

        s = state.settings
        manifest = state.manifest()
        card = load_knowledge_card(s)
        return {
            "indexed": manifest is not None,
            "document": None
            if manifest is None
            else {
                "title": manifest.get("title"),
                "source": manifest.get("source"),
                "built_at": manifest.get("built_at"),
                "chunk_counts": manifest.get("chunk_counts", {}),
            },
            "ingest": state.job.to_dict(),
            "knowledge_card": None if card is None else {"reviewed": bool(card.get("reviewed"))},
            # The hosted demo has no Ollama; its model is reached through Groq.
            "ollama": check_ollama(s.ollama_base_url) if s.llm_provider == "ollama" else "ok",
            "ingest_enabled": s.allow_ingest,
            "settings": {
                "llm_provider": s.llm_provider,
                "llm_model": s.groq_model if s.llm_provider == "groq" else s.llm_model,
                "embedding_model": s.embedding_model,
                "use_bm25": s.use_bm25,
                "use_reranker": s.use_reranker,
                "use_multi_query": s.use_multi_query,
                "top_k": s.top_k,
            },
        }

    @app.post("/ingest", status_code=202)
    async def ingest(
        file: Annotated[UploadFile | None, File()] = None,
        path: Annotated[str | None, Form()] = None,
        force: Annotated[bool, Form()] = False,
    ) -> dict:
        if not state.settings.allow_ingest:
            raise HTTPException(403, "Uploading documents is disabled on this deployment")
        if file is not None:
            data = await file.read()
            if len(data) > MAX_PDF_BYTES:
                raise HTTPException(413, "PDF is larger than 100 MB")
            if not data.startswith(b"%PDF"):
                raise HTTPException(400, "The uploaded file is not a PDF")
            name = _SAFE_NAME.sub("_", Path(file.filename or "upload.pdf").name)
            if not name.lower().endswith(".pdf"):
                name += ".pdf"
            pdf_path = state.settings.data_dir / "raw" / name
            pdf_path.parent.mkdir(parents=True, exist_ok=True)
            pdf_path.write_bytes(data)
        elif path:
            pdf_path = Path(path).expanduser()
            if not pdf_path.is_file():
                raise HTTPException(400, f"No file at {path}")
        else:
            raise HTTPException(400, "Send a PDF as 'file' or a server path as 'path'")
        state.start_ingest(pdf_path, force)
        return {"status": "started", "source": pdf_path.name}

    @app.post("/ask")
    def ask(
        req: AskRequest, x_client_id: Annotated[str | None, Header()] = None
    ) -> StreamingResponse:
        if not state.indexed():
            raise HTTPException(409, "No document is indexed yet; POST /ingest first")
        owner = _owner(x_client_id, required=False)
        question = req.question.strip()
        store = state.store if owner else None
        chat_id, new_chat = req.chat_id, None
        if store is not None:
            if chat_id and store.get_chat(owner, chat_id) is None:
                raise HTTPException(404, "Chat not found")
            if not chat_id:
                from .store import title_from

                new_chat = store.create_chat(owner, title_from(question))
                chat_id = new_chat["id"]
        history = store.history(owner, chat_id, state.settings.history_turns) if store else []
        try:
            qa = state.qa()
        except Exception as exc:  # e.g. the hosted model's API key is not configured
            log.exception("Could not start the QA pipeline")
            raise HTTPException(503, _friendly_error(exc)) from exc

        def events() -> Iterator[str]:
            try:
                if new_chat:
                    yield _sse("chat", {"id": new_chat["id"], "title": new_chat["title"]})
                for kind, value in qa.ask_stream(question, history):
                    if kind == "done":
                        answer = value.to_dict()
                        if store is not None:
                            store.add_exchange(owner, chat_id, question, answer)
                        yield _sse("answer", {**answer, "chat_id": chat_id})
                    elif kind == "mode":
                        yield _sse("mode", {"mode": value})
                    else:  # "token" (answer text) or "reasoning"
                        yield _sse(kind, {"text": value})
                yield _sse("done", {})
            except Exception as exc:  # the HTTP status is already sent; report in-band
                log.exception("Answer failed")
                yield _sse("error", {"message": _friendly_error(exc)})

        # Sync generator: Starlette iterates it in a worker thread, so the blocking
        # model stream does not stall the event loop.
        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # --- saved chats -------------------------------------------------------
    @app.get("/chats")
    def list_chats(x_client_id: Annotated[str | None, Header()] = None, q: str = "") -> list[dict]:
        return state.store.list_chats(_owner(x_client_id), q.strip())

    @app.post("/chats", status_code=201)
    def create_chat(x_client_id: Annotated[str | None, Header()] = None) -> dict:
        return state.store.create_chat(_owner(x_client_id))

    @app.get("/chats/{chat_id}")
    def get_chat(chat_id: str, x_client_id: Annotated[str | None, Header()] = None) -> dict:
        chat = state.store.get_chat(_owner(x_client_id), chat_id)
        if chat is None:
            raise HTTPException(404, "Chat not found")
        return chat

    @app.patch("/chats/{chat_id}")
    def rename_chat(
        chat_id: str, req: RenameRequest, x_client_id: Annotated[str | None, Header()] = None
    ) -> dict:
        if not state.store.rename_chat(_owner(x_client_id), chat_id, req.title):
            raise HTTPException(404, "Chat not found")
        return {"id": chat_id, "title": " ".join(req.title.split())}

    @app.delete("/chats/{chat_id}", status_code=204)
    def delete_chat(chat_id: str, x_client_id: Annotated[str | None, Header()] = None) -> None:
        if not state.store.delete_chat(_owner(x_client_id), chat_id):
            raise HTTPException(404, "Chat not found")

    @app.post("/analyze")
    def analyze(
        file: Annotated[UploadFile, File()],
        group_col: Annotated[str | None, Form()] = None,
        explain: Annotated[bool, Form()] = True,
    ) -> dict:
        from .analyze import analyze_csv

        data = file.file.read()
        qa = state.qa() if explain and state.indexed() else None
        try:
            result = analyze_csv(data, state.settings, qa, group_col or None)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:
            log.exception("Analysis failed")
            raise HTTPException(502, _friendly_error(exc)) from exc
        if explain and qa is None:
            result["note"] = "No document is indexed, so only statistics are shown."
        return result

    if WEB_DIR.is_dir():

        @app.get("/", include_in_schema=False)
        def index() -> FileResponse:
            return FileResponse(WEB_DIR / "index.html")

        app.mount("/", StaticFiles(directory=WEB_DIR), name="web")

    return app


def _friendly_error(exc: Exception) -> str:
    text = str(exc)
    if "Connection refused" in text or "ConnectError" in type(exc).__name__:
        return "Cannot reach the language model. Is Ollama running (ollama serve)?"
    if "api_key" in text.lower() or "api key" in text.lower():
        return "The language model is not configured on this server (missing API key)."
    if "rate limit" in text.lower() or "429" in text:
        return "The model is rate-limited right now; please try again in a minute."
    return text or type(exc).__name__


def _warm_up_mode(value: str) -> bool | str:
    """RAG_WARMUP: "0" off, "startup" load before serving, anything else in the background."""
    value = value.strip().lower()
    if value == "startup":
        return "startup"
    return value != "0"


app = create_app(warm_up=_warm_up_mode(os.environ.get("RAG_WARMUP", "1")))


def main() -> None:
    import uvicorn

    uvicorn.run(
        "thesis_rag.api:app",
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
