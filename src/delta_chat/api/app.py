"""FastAPI serving layer — thin HTTP adapters over the ingest/delta/chat pipeline.

No business logic lives here: every endpoint resolves a PID, calls into
``ingest``/``delta``/``chat``, and shapes the result for HTTP. Each request opens a
``pipeline.request`` root span (section-02 tracing) with `ingest`/`delta`/`retrieval`/`llm.call`/
`answer` child stages, and binds a `request_id` correlation key that must survive the thread-pool
offload used for sync CPU work (PyMuPDF/Tesseract/delta compute) — see `run_sync_ctx`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import hashlib
import html as _html
import json
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..canonical.model import Document
from ..chat.answer import answer
from ..chat.index import build_index
from ..chat.llm import LlmUnavailableError
from ..config import Config, default_config_path, load_config
from ..delta.engine import compute_delta
from ..delta.report import render_report
from ..ingest import base as ingest_base
from ..observability import logging as _logging
from ..observability import metrics
from ..observability import tracing
from ..pids.resolver import PidResolutionError
from ..pids.resolver import resolve as resolve_pid

_log = _logging.get_logger("delta_chat.api")
_REPO_ROOT = default_config_path().parent.parent


class IngestReq(BaseModel):
    pid: str
    source_rev: str = "A"


class DeltaReq(BaseModel):
    pid_a: str
    pid_b: str


class ChatReq(BaseModel):
    pid_a: str
    pid_b: str
    query: str


async def run_sync_ctx(fn, *args, **kwargs):
    """Offload sync CPU work to the thread-pool executor while preserving the current context
    (request_id, active trace/stage) — contextvars set on the event-loop task are not otherwise
    visible in worker threads (plan §9; explicitly tested via stage-log request_id propagation)."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, tracing.run_in_context(fn, *args, **kwargs))


def _parse_with_log(data: bytes, pid: str, filename: str | None, *, source_rev: str, cfg) -> Document:
    _log.info("ingest.parse", pid=pid, source_rev=source_rev)
    return ingest_base.parse(data, pid, filename, source_rev=source_rev, cfg=cfg)


async def _ingest_pair(pid_a: str, pid_b: str, cfg) -> tuple[Document, Document]:
    data_a, meta_a = resolve_pid(pid_a)
    data_b, meta_b = resolve_pid(pid_b)
    with tracing.stage("ingest"):
        doc_a = await run_sync_ctx(_parse_with_log, data_a, pid_a, meta_a.get("filename"), source_rev="A", cfg=cfg)
        doc_b = await run_sync_ctx(_parse_with_log, data_b, pid_b, meta_b.get("filename"), source_rev="B", cfg=cfg)
    return doc_a, doc_b


def _pid_not_found(exc: PidResolutionError, *pids: str) -> HTTPException:
    # The underlying PidResolutionError message includes resolved server filesystem paths and the
    # allowlist itself — never echo that to an HTTP client (info disclosure / file-existence
    # oracle). Log the full detail server-side; return only the client's own input back to them.
    _log.warning("pid_resolution_failed", pids=pids, error=str(exc))
    return HTTPException(status_code=404, detail=f"could not resolve pid(s): {list(pids)!r}")


def _unknown_format(exc: ingest_base.UnknownFormatError, *pids: str) -> HTTPException:
    # The underlying message echoes raw file magic bytes read from disk — don't return that either.
    _log.warning("unknown_format", pids=pids, error=str(exc))
    return HTTPException(status_code=400, detail=f"unrecognized or unsupported document format for pid(s): {list(pids)!r}")


def _llm_unavailable(exc: Exception) -> HTTPException:
    """No usable LLM provider is a deployment/config problem, not a server fault.

    Returning 500 here made a keyless deployment look broken: ingest/delta/retrieval all work
    without a key, and only answer *generation* needs one, so the UI should be able to say
    "configure a key" rather than surfacing an opaque HTTP 500. 503 + a fixed, non-echoing message
    (the same info-disclosure convention as the helpers above).
    """
    _log.warning("llm_unavailable", error=str(exc))
    return HTTPException(
        status_code=503,
        detail="grounded chat is unavailable: no LLM provider is configured on the server "
               "(set ANTHROPIC_API_KEY, OPENAI_API_KEY, or LOCAL_LLM_BASE_URL). "
               "Ingest, delta and the delta report do not require a key.",
    )


def _delta_id(pid_a: str, pid_b: str) -> str:
    # json.dumps unambiguously delimits/escapes each string, so a literal separator character
    # inside a pid can't make two distinct (pid_a, pid_b) pairs collide on the same hash.
    return hashlib.sha256(json.dumps([pid_a, pid_b]).encode()).hexdigest()[:16]


def _report_html(markdown_text: str) -> str:
    body = _html.escape(markdown_text)
    return f"<!doctype html><html><head><meta charset='utf-8'><title>Delta Report</title></head><body><pre>{body}</pre></body></html>"


def _store_delta(store: dict, pid_a: str, pid_b: str, doc_a: Document, doc_b: Document, delta) -> tuple[str, dict]:
    report_md, report_json = render_report(delta)
    delta_id = _delta_id(pid_a, pid_b)
    entry = {
        "delta": delta, "doc_a": doc_a, "doc_b": doc_b,
        "report_json": report_json, "report_html": _report_html(report_md),
        "pid_a": pid_a, "pid_b": pid_b,
    }
    store[delta_id] = entry
    return delta_id, entry


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or load_config()
    app = FastAPI(title="delta-chat")
    app.state.cfg = cfg
    app.state.delta_store = {}

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception):
        # Any exception not already converted to an HTTPException by an endpoint (e.g. a bug in
        # render_report/build_index/answer) lands here instead of an uncontrolled bare 500 — this
        # keeps the "clear 4xx/5xx with JSON detail" contract. Starlette wires an `Exception`
        # handler into ServerErrorMiddleware, which sits *outside* request_id_middleware, so that
        # middleware's own header-setting code never runs for this path — read the id back from
        # request.state (stashed before call_next) so the response still carries X-Request-ID.
        # Full exception detail (message, type, path) is logged server-side only — never echoed to
        # the client, which could otherwise leak internal file paths, stack info, or other state.
        _log.error("unhandled_exception", path=request.url.path, error_type=type(exc).__name__, error=str(exc))
        response = JSONResponse(status_code=500, content={"detail": "internal server error"})
        request_id = getattr(request.state, "request_id", None)
        if request_id:
            response.headers["X-Request-ID"] = request_id
        return response

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        tr = tracing.start_request(
            {"path": request.url.path, "method": request.method}, cfg=request.app.state.cfg
        )
        if tr.request_id != request_id:
            # honor an inbound X-Request-ID as the correlation key rather than the trace's own uuid
            tr.request_id = request_id
            _logging.bind_request_id(request_id)
        request.state.request_id = tr.request_id
        try:
            response = await call_next(request)
        except Exception:
            tr.finish()
            raise
        tr.finish()
        response.headers["X-Request-ID"] = tr.request_id
        return response

    @app.post("/ingest")
    async def ingest_endpoint(req: IngestReq, request: Request):
        cfg = request.app.state.cfg
        try:
            data, meta = resolve_pid(req.pid)
        except PidResolutionError as exc:
            raise _pid_not_found(exc, req.pid) from exc
        try:
            with tracing.stage("ingest"):
                doc = await run_sync_ctx(
                    _parse_with_log, data, req.pid, meta.get("filename"), source_rev=req.source_rev, cfg=cfg
                )
        except ingest_base.UnknownFormatError as exc:
            raise _unknown_format(exc, req.pid) from exc
        return {
            "pid": doc.pid, "format": doc.format, "page_count": doc.page_count,
            "revision_label": doc.revision_label,
            "item_count": len(doc.items), "region_count": len(doc.regions),
        }

    @app.post("/delta")
    async def delta_endpoint(req: DeltaReq, request: Request):
        cfg = request.app.state.cfg
        try:
            doc_a, doc_b = await _ingest_pair(req.pid_a, req.pid_b, cfg)
            with tracing.stage("delta"):
                delta = await run_sync_ctx(compute_delta, doc_a, doc_b, cfg)
                delta_id, entry = _store_delta(
                    request.app.state.delta_store, req.pid_a, req.pid_b, doc_a, doc_b, delta
                )
        except PidResolutionError as exc:
            raise _pid_not_found(exc, req.pid_a, req.pid_b) from exc
        except ingest_base.UnknownFormatError as exc:
            raise _unknown_format(exc, req.pid_a, req.pid_b) from exc
        return {
            "delta_id": delta_id,
            "report_json": entry["report_json"],
            "report_md_url": f"/report?delta_id={delta_id}",
        }

    @app.get("/report")
    async def report_endpoint(delta_id: str, request: Request):
        entry = request.app.state.delta_store.get(delta_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown delta_id {delta_id!r}")
        return HTMLResponse(entry["report_html"])

    @app.post("/chat")
    async def chat_endpoint(req: ChatReq, request: Request):
        cfg = request.app.state.cfg
        store = request.app.state.delta_store
        delta_id = _delta_id(req.pid_a, req.pid_b)
        entry = store.get(delta_id)
        if entry is None:
            try:
                doc_a, doc_b = await _ingest_pair(req.pid_a, req.pid_b, cfg)
                with tracing.stage("delta"):
                    delta = await run_sync_ctx(compute_delta, doc_a, doc_b, cfg)
                    _delta_id_unused, entry = _store_delta(store, req.pid_a, req.pid_b, doc_a, doc_b, delta)
            except PidResolutionError as exc:
                raise _pid_not_found(exc, req.pid_a, req.pid_b) from exc
            except ingest_base.UnknownFormatError as exc:
                raise _unknown_format(exc, req.pid_a, req.pid_b) from exc
        with tracing.stage("chat"):
            index = build_index(entry["doc_a"], entry["doc_b"], entry["delta"], cfg)
            # answer() may call the LLM (network); offload so a slow provider never blocks the loop.
            # Refusal is a normal 200 with state="Rejected" — never an HTTP error (never raised here).
            try:
                result = await run_sync_ctx(answer, req.query, index, cfg)
            except LlmUnavailableError as exc:
                raise _llm_unavailable(exc) from exc
        return dataclasses.asdict(result)

    @app.get("/markup")
    async def markup_endpoint(delta_id: str, request: Request):
        entry = request.app.state.delta_store.get(delta_id)
        if entry is None:
            raise HTTPException(status_code=404, detail=f"unknown delta_id {delta_id!r}")
        try:
            from ..markup import overlay
        except ImportError:
            raise HTTPException(status_code=501, detail="markup overlay not available") from None
        cfg = request.app.state.cfg
        data, _meta = resolve_pid(entry["pid_b"])
        # A render_markup failure here isn't separately caught: this endpoint's whole job is
        # producing markup (unlike CLI `run`'s best-effort step), and the global exception handler
        # already turns it into a clean 500 rather than a raw crash.
        pdf_bytes = await run_sync_ctx(overlay.render_markup, entry["delta"], data, cfg)
        return Response(content=pdf_bytes, media_type="application/pdf")

    @app.get("/metrics")
    async def metrics_endpoint():
        body, content_type = metrics.render_prometheus()
        return Response(content=body, media_type=content_type)

    @app.get("/healthz")
    async def healthz_endpoint():
        return {"status": "ok"}

    web_dir = Path(cfg.get("api.web_dir", "src/delta_chat/api/web"))
    if not web_dir.is_absolute():
        web_dir = _REPO_ROOT / web_dir
    if web_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(web_dir), html=True), name="ui")

    return app


app = create_app()
