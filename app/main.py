"""FastAPI app: routes, wiring and error mapping.

`create_app(settings, services)` builds the app; tests pass fake services (embedder, LLM, reranker)
so no test needs an API key. `app` at the bottom is what uvicorn serves.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import psycopg
from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse

from app.answer import AnthropicLLM, AskOutcome, LLMClient, LLMUnavailable, answer_question
from app.config import Settings, get_settings
from app.demo import NOT_RECORDED, DemoBackend
from app.embeddings import Embedder, EmbeddingError, build_embedder, voyage_limiter
from app.guards import enforce_usage_limit, require_api_key
from app.ingest import ingest_document
from app.ingest_pdf import PdfRejected, check_pdf_upload, extract_pages
from app.limits import UsageLimiter
from app.logs import log_event, setup_logging
from app.rerank import Reranker, VoyageReranker
from app.retrieve import retrieve
from app.schemas import (
    ID_PATTERN,
    AskIn,
    AskOut,
    CitationOut,
    DocumentIn,
    DocumentInfo,
    DocumentList,
    HealthOut,
    IngestOut,
    ReadyOut,
    RetrievedChunkOut,
    RetrieveIn,
    RetrieveOut,
)
from app.store import Store, StoreError

log = logging.getLogger("evidence_rag")
UI_FILE = Path(__file__).parent / "ui" / "index.html"


@dataclass
class Services:
    """Everything a request may need. Built lazily so a missing optional key only affects the
    routes that need it (e.g. no ANTHROPIC_API_KEY: /ask returns 503, /retrieve still works)."""

    settings: Settings
    store: Store | None = None
    embedder: Embedder | None = None
    llm: LLMClient | None = None
    reranker: Reranker | None = None
    demo: DemoBackend | None = None

    def get_store(self) -> Store:
        if self.store is None:
            raise HTTPException(503, "database is not configured")
        return self.store

    def get_embedder(self) -> Embedder:
        if self.embedder is None:
            try:
                self.embedder = build_embedder(self.settings)
            except EmbeddingError as exc:
                raise HTTPException(503, str(exc)) from exc
        return self.embedder

    def get_llm(self) -> LLMClient:
        if self.llm is None:
            if not self.settings.anthropic_api_key:
                raise HTTPException(
                    503, "ANTHROPIC_API_KEY is not set: /ask cannot call Claude (try /retrieve)"
                )
            s = self.settings
            self.llm = AnthropicLLM(
                s.anthropic_api_key,
                s.answer_model,
                s.answer_effort,
                s.answer_max_tokens,
                s.answer_refusal_fallback,
                s.answer_api_base_url,
            )
        return self.llm

    def get_reranker(self) -> Reranker | None:
        if self.reranker is None and self.settings.voyage_api_key:
            s = self.settings
            self.reranker = VoyageReranker(
                s.voyage_api_key,
                s.rerank_model,
                voyage_limiter(s.voyage_api_key, s.voyage_rpm, s.voyage_tpm),
            )
        return self.reranker


def build_services(settings: Settings) -> Services:
    if settings.is_demo:
        demo = DemoBackend(
            settings.corpus_dir,
            settings.recordings_path,
            settings.demo_collection,
            settings.chunk_max_chars,
        )
        return Services(settings=settings, demo=demo)
    store = Store(settings.database_url, settings.embedding_dim)
    store.init_schema()
    return Services(settings=settings, store=store)


def _cors_origins(settings: Settings | None) -> list[str]:
    """Read at app creation (middleware cannot change later). Never raises: a broken setting is
    reported properly at startup, by the lifespan."""
    try:
        return (settings or Settings()).cors_origin_list
    except Exception:  # noqa: BLE001
        return []


def _note(request: Request, **fields) -> None:
    """Add fields to this request's log line (counts and ids only, never text or keys)."""
    request.state.log_fields.update(fields)


def create_app(settings: Settings | None = None, services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        s = settings or get_settings()
        setup_logging(s.log_format, s.log_level)
        app.state.settings = s
        # Validated here, so a malformed API_KEYS stops the start instead of locking callers out.
        app.state.callers = {} if s.is_demo else s.callers
        app.state.limiter = UsageLimiter(s.ask_per_minute, s.ask_per_day)
        app.state.demo_limiter = UsageLimiter(s.demo_per_minute, 0)
        app.state.services = services or build_services(s)
        log_event(
            log,
            "started",
            mode=s.evidence_rag_mode,
            callers=sorted(app.state.callers),
            ask_per_minute=s.ask_per_minute,
            ask_per_day=s.ask_per_day,
        )
        yield

    app = FastAPI(
        title="evidence-rag",
        version="0.1.0",
        summary="Answers about your documents with exact-quote citations, checked in code.",
        description=(
            "Every answer must quote the exact text it relies on. Code checks each quote against "
            "the retrieved chunks; failures get one retry, then `needs_human`. Questions the "
            "documents cannot answer are refused (`abstained`). Synthetic data only."
        ),
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def request_log(request: Request, call_next):
        """One log line per request: id, caller, route, status, duration (+ fields from _note)."""
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        request.state.log_fields = {}
        started = time.perf_counter()

        def emit(status_code: int) -> None:
            route = request.scope.get("route")
            log_event(
                log,
                "request",
                logging.WARNING if status_code >= 500 else logging.INFO,
                request_id=request_id,
                caller=getattr(request.state, "caller", None),
                method=request.method,
                route=getattr(route, "path", "unmatched"),  # the template, not the raw URL
                status=status_code,
                duration_ms=int((time.perf_counter() - started) * 1000),
                **request.state.log_fields,
            )

        try:
            response = await call_next(request)
        except Exception:
            emit(500)
            raise
        response.headers["X-Request-ID"] = request_id
        emit(response.status_code)
        return response

    origins = _cors_origins(settings)
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "Content-Type"],
            expose_headers=[
                "Retry-After",
                "X-Request-ID",
                "X-RateLimit-Remaining-Minute",
                "X-RateLimit-Remaining-Day",
            ],
        )

    def svc(request: Request) -> Services:
        return request.app.state.services

    # ---------- public ----------

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/ui")

    @app.get("/health", response_model=HealthOut, tags=["meta"])
    def health(request: Request) -> HealthOut:
        return HealthOut(status="ok", mode=request.app.state.settings.evidence_rag_mode)

    @app.get(
        "/ready",
        response_model=ReadyOut,
        responses={503: {"description": "the database is unreachable or has no schema"}},
        tags=["meta"],
    )
    def ready(request: Request, s: Services = Depends(svc)) -> ReadyOut:
        """Readiness: can this instance serve requests right now? /health only says it is alive."""
        mode = request.app.state.settings.evidence_rag_mode
        if s.store is None:
            return ReadyOut(status="ready", mode=mode, database="not_used")
        try:
            s.store.ping()
        except (StoreError, psycopg.Error) as exc:
            log.warning("not ready: %s", type(exc).__name__)
            raise HTTPException(503, "database is not ready") from exc
        return ReadyOut(status="ready", mode=mode, database="ok")

    @app.get("/ui", include_in_schema=False)
    def ui() -> FileResponse:
        return FileResponse(UI_FILE, media_type="text/html")

    # ---------- documents ----------

    def _live_only(s: Services) -> None:
        if s.demo is not None:
            raise HTTPException(409, "demo mode is read-only: run in live mode to add documents")

    @app.post(
        "/documents",
        response_model=IngestOut,
        tags=["documents"],
        dependencies=[Depends(require_api_key)],
    )
    def add_document(body: DocumentIn, s: Services = Depends(svc)) -> IngestOut:
        _live_only(s)
        try:
            result = ingest_document(
                s.get_store(),
                s.get_embedder(),
                collection_id=body.collection_id,
                doc_id=body.doc_id,
                title=body.title,
                version=body.version,
                source_type="text",
                pages=[(None, body.text)],
                raw_content=body.text.encode("utf-8"),
                chunk_max_chars=s.settings.chunk_max_chars,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        return IngestOut(**result.__dict__)

    @app.post(
        "/documents/pdf",
        response_model=IngestOut,
        tags=["documents"],
        dependencies=[Depends(require_api_key)],
    )
    async def add_pdf(
        file: UploadFile = File(...),
        doc_id: str = Form(..., pattern=ID_PATTERN),
        title: str = Form(..., min_length=1, max_length=200),
        version: str = Form("", max_length=64),
        collection_id: str = Form("default", pattern=ID_PATTERN),
        s: Services = Depends(svc),
    ) -> IngestOut:
        _live_only(s)
        limit = s.settings.max_pdf_mb * 1024 * 1024
        data = await file.read(limit + 1)
        try:
            check_pdf_upload(data, file.content_type, s.settings.max_pdf_mb)
            pages = extract_pages(data, s.settings.max_pdf_pages)
            result = ingest_document(
                s.get_store(),
                s.get_embedder(),
                collection_id=collection_id,
                doc_id=doc_id,
                title=title,
                version=version,
                source_type="pdf",
                pages=[(n, t) for n, t in pages],
                raw_content=data,
                chunk_max_chars=s.settings.chunk_max_chars,
            )
        except (PdfRejected, ValueError) as exc:
            raise HTTPException(422, str(exc)) from exc
        return IngestOut(**result.__dict__)

    @app.get(
        "/documents",
        response_model=DocumentList,
        tags=["documents"],
        dependencies=[Depends(require_api_key)],
    )
    def list_documents(
        collection_id: str | None = Query(None, pattern=ID_PATTERN), s: Services = Depends(svc)
    ) -> DocumentList:
        if s.demo is not None:
            counts: dict[str, int] = {}
            for c in s.demo.chunks.values():
                counts[c.doc_id] = counts.get(c.doc_id, 0) + 1
            docs = [
                DocumentInfo(
                    collection_id=s.demo.collection_id,
                    doc_id=d.doc_id,
                    title=d.title,
                    version=d.version,
                    source_type=d.source_type,
                    pages=len(d.pages) if d.source_type == "pdf" else None,
                    chunks=counts.get(d.doc_id, 0),
                )
                for d in s.demo.docs
                if collection_id in (None, s.demo.collection_id)
            ]
            return DocumentList(documents=docs)
        rows = s.get_store().list_documents(collection_id)
        return DocumentList(documents=[DocumentInfo(**r) for r in rows])

    # ---------- ask / retrieve ----------

    def _demo_collection(demo: DemoBackend, collection_id: str) -> None:
        if collection_id not in demo.accepted_collections:
            raise HTTPException(
                404, f"demo mode only serves the recorded collection {demo.collection_id!r}"
            )

    def _to_out(outcome: AskOutcome, mode: str, rerank: str | None, started: float, run_mode: str):
        return AskOut(
            status=outcome.status,
            answer=outcome.answer,
            citations=[CitationOut(**c.__dict__) for c in outcome.citations],
            retried=outcome.retried,
            mode=mode,
            errors=outcome.errors,
            rerank=rerank,
            latency_ms=int((time.perf_counter() - started) * 1000),
            run_mode=run_mode,
        )

    def _note_outcome(request: Request, body: AskIn, outcome: AskOutcome, run_mode: str) -> None:
        _note(
            request,
            question_len=len(body.question),
            retrieval_mode=body.mode,
            run_mode=run_mode,
            answer_status=outcome.status,
            retried=outcome.retried,
            citations=len(outcome.citations),
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
        )

    @app.post(
        "/ask",
        response_model=AskOut,
        tags=["ask"],
        dependencies=[Depends(enforce_usage_limit)],
        responses={429: {"description": "per-minute or daily limit reached; see Retry-After"}},
    )
    def ask(request: Request, body: AskIn, s: Services = Depends(svc)) -> AskOut:
        started = time.perf_counter()
        _note(request, question_len=len(body.question), retrieval_mode=body.mode)
        if s.demo is not None:
            _demo_collection(s.demo, body.collection_id)
            outcome = s.demo.ask(body.question, body.mode)
            if outcome is None:
                raise HTTPException(404, NOT_RECORDED)
            _note_outcome(request, body, outcome, "demo")
            return _to_out(outcome, body.mode, None, started, "demo")

        store = s.get_store()
        if store.count_chunks(body.collection_id) == 0:
            raise HTTPException(409, f"no documents in collection {body.collection_id!r}")
        llm = s.get_llm()
        result = retrieve(
            store,
            s.get_embedder(),
            collection_id=body.collection_id,
            question=body.question,
            mode=body.mode,
            k=s.settings.retrieval_k,
            candidates=s.settings.rerank_candidates,
            reranker=s.get_reranker() if body.mode == "hybrid_rerank" else None,
        )
        outcome = answer_question(llm, body.question, result.chunks)
        _note_outcome(request, body, outcome, "live")
        return _to_out(outcome, body.mode, result.rerank, started, "live")

    @app.post(
        "/retrieve",
        response_model=RetrieveOut,
        tags=["ask"],
        dependencies=[Depends(enforce_usage_limit)],
        summary="Show the chunks retrieval returns for a question (no Claude call)",
        responses={429: {"description": "per-minute or daily limit for this caller"}},
    )
    def retrieve_route(
        request: Request, body: RetrieveIn, s: Services = Depends(svc)
    ) -> RetrieveOut:
        _note(request, question_len=len(body.question), retrieval_mode=body.mode)
        if s.demo is not None:
            _demo_collection(s.demo, body.collection_id)
            rec = s.demo.recordings.get((" ".join(body.question.split()).lower(), body.mode))
            if rec is None:
                raise HTTPException(404, NOT_RECORDED)
            if any(c not in s.demo.chunks for c in rec.retrieved_chunk_ids):
                raise HTTPException(409, "demo mode: the recording does not match the corpus")
            chunks = [s.demo.chunks[c] for c in rec.retrieved_chunk_ids]
            return RetrieveOut(
                mode=body.mode,
                chunks=[
                    RetrievedChunkOut(
                        chunk_id=c.chunk_id,
                        doc_id=c.doc_id,
                        title=c.title,
                        page=c.page,
                        score=0.0,
                        text=c.text,
                    )
                    for c in chunks
                ],
            )
        store = s.get_store()
        if store.count_chunks(body.collection_id) == 0:
            raise HTTPException(409, f"no documents in collection {body.collection_id!r}")
        result = retrieve(
            store,
            s.get_embedder(),
            collection_id=body.collection_id,
            question=body.question,
            mode=body.mode,
            k=s.settings.retrieval_k,
            candidates=s.settings.rerank_candidates,
            reranker=s.get_reranker() if body.mode == "hybrid_rerank" else None,
        )
        return RetrieveOut(
            mode=body.mode,
            rerank=result.rerank,
            chunks=[
                RetrievedChunkOut(
                    chunk_id=c.chunk_id,
                    doc_id=c.doc_id,
                    title=c.title,
                    page=c.page,
                    score=c.score,
                    text=c.text,
                )
                for c in result.chunks
            ],
        )

    @app.get("/demo/questions", tags=["meta"], summary="Recorded questions (demo mode)")
    def demo_questions(s: Services = Depends(svc)) -> dict:
        if s.demo is None:
            return {"mode": "live", "questions": []}
        return {"mode": "demo", "meta": s.demo.meta, "questions": s.demo.recorded_questions()}

    @app.get(
        "/metrics",
        tags=["meta"],
        summary="Measured results (results.json, written by the scripts), read-only",
    )
    def metrics(s: Services = Depends(svc)) -> dict:
        path = Path(s.settings.corpus_dir).parent.parent / "results.json"
        if not path.exists():
            return {"note": "not measured yet"}
        return json.loads(path.read_text("utf-8"))

    @app.get("/samples", tags=["meta"], summary="Example questions from the test set, by kind")
    def samples(s: Services = Depends(svc)) -> dict:
        path = Path(s.settings.corpus_dir).parent / "questions" / "questions.json"
        if not path.exists():
            return {"groups": []}
        questions = json.loads(path.read_text("utf-8"))
        if s.demo is not None:  # demo mode can only answer what was recorded
            recorded = {q["question"] for q in s.demo.recorded_questions()}
            questions = [q for q in questions if q["question"] in recorded]
        groups = [
            (
                "Decoys",
                "a superseded or look-alike document says something else",
                lambda q: "decoy" in q.get("notes", "").lower(),
            ),
            (
                "PDF, page 2+",
                "the answer sits deep inside a PDF",
                lambda q: any(p and p >= 2 for p in q["expected_pages"]),
            ),
            (
                "Paraphrased",
                "the question uses other words than the document",
                lambda q: "paraphrase" in q["tags"],
            ),
            (
                "Exact terms",
                "codes, rates and limits that must match exactly",
                lambda q: "exact-term" in q["tags"],
            ),
            (
                "Two documents",
                "the answer needs two documents together",
                lambda q: q["type"] == "multi",
            ),
            (
                "Not in the documents",
                "the right answer is a refusal",
                lambda q: q["type"] == "unanswerable",
            ),
        ]
        out = []
        for name, hint, rule in groups:
            items = [{"id": q["id"], "question": q["question"]} for q in questions if rule(q)]
            if items:
                out.append({"group": name, "hint": hint, "items": items[:5]})
        return {"groups": out}

    # ---------- error mapping ----------

    @app.exception_handler(EmbeddingError)
    async def _embedding_down(_: Request, exc: EmbeddingError):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(LLMUnavailable)
    async def _llm_down(_: Request, exc: LLMUnavailable):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(psycopg.Error)
    async def _db_error(_: Request, exc: psycopg.Error):
        log.warning("database error: %s", type(exc).__name__)
        return JSONResponse({"detail": "database error, try again"}, status_code=503)

    @app.exception_handler(StoreError)
    async def _db_down(_: Request, exc: StoreError):
        return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.exception_handler(Exception)
    async def _unexpected(request: Request, exc: Exception):
        """Anything not mapped above: log it with the request id, tell the caller only the id.
        No stack trace, text or key ever reaches the response."""
        request_id = getattr(request.state, "request_id", None)
        log.error("unhandled %s (request_id=%s)", type(exc).__name__, request_id)
        return JSONResponse(
            {"detail": "internal error", "request_id": request_id}, status_code=500
        )

    return app


app = create_app()
