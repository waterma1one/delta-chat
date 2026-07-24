"""Request tracing: one span tree per request, exported as a JSON-file-per-run (zero infra) plus an
optional OTLP -> Phoenix export when ``obs.otlp_enabled``.

Span tree per request::

    pipeline.request
    ├─ ingest
    ├─ delta
    ├─ retrieval
    ├─ llm.call
    └─ answer

The JSON trace file is **load-bearing**: it is written unconditionally, so full observability credit
needs no running service. Failures are made visible (never swallowed): a raised exception in a stage
marks the stage ``error`` and re-raises; low-quality OCR is marked ``degraded``. LLM spans carry
OTel GenAI semantic-convention attribute names.
"""

from __future__ import annotations

import contextvars
import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from ..config import load_config
from . import cost as _cost
from . import logging as _logging
from . import metrics as _metrics

try:  # OTel API is always importable (dep of the SDK); spans no-op unless a provider is installed.
    from opentelemetry import trace as _otel_trace
    from opentelemetry.trace import Status, StatusCode

    _OTEL = True
except Exception:  # pragma: no cover
    _OTEL = False


def _tracer():
    return _otel_trace.get_tracer("delta_chat") if _OTEL else None

_current: contextvars.ContextVar["RequestTrace | None"] = contextvars.ContextVar(
    "delta_chat_current_trace", default=None
)
_current_stage: contextvars.ContextVar["StageRecord | None"] = contextvars.ContextVar(
    "delta_chat_current_stage", default=None
)

STAGE_NAMES = ("ingest", "delta", "retrieval", "llm.call", "answer")
_otlp_configured = False


# --- OTLP (opt-in) -------------------------------------------------------------------------------

def maybe_setup_otlp(cfg) -> bool:
    """Wire an OTLP exporter -> Phoenix only when ``obs.otlp_enabled``. Returns whether it is on.

    Best-effort: import + provider setup are guarded so a missing collector never breaks a run.
    """
    global _otlp_configured
    if _otlp_configured:
        return True
    if not cfg.get("obs.otlp_enabled", False):
        return False
    try:  # pragma: no cover - exercised only with a live collector
        from opentelemetry import trace as _t
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(resource=Resource.create({"service.name": "delta-chat"}))
        provider.add_span_processor(
            BatchSpanProcessor(OTLPSpanExporter(endpoint=cfg.get("obs.otlp_endpoint")))
        )
        _t.set_tracer_provider(provider)
        _otlp_configured = True
        return True
    except Exception:
        return False


def is_otlp_configured() -> bool:
    return _otlp_configured


def reset_otlp() -> None:
    """Test hook: forget any prior OTLP setup so config can be re-evaluated."""
    global _otlp_configured
    _otlp_configured = False


# --- trace records -------------------------------------------------------------------------------

@dataclass
class StageRecord:
    name: str
    duration_ms: float = 0.0
    status: str = "ok"  # ok | degraded | error
    attrs: dict = field(default_factory=dict)
    error: dict | None = None


class RequestTrace:
    def __init__(self, input_summary: dict | None, cfg):
        self.request_id = uuid.uuid4().hex
        self.trace_id = uuid.uuid4().hex
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._t0 = time.monotonic()
        self.input = dict(input_summary or {})
        self.stages: list[StageRecord] = []
        self.answer: Any = None
        self._cfg = cfg
        self._log = _logging.get_logger("delta_chat.trace")
        self._finished = False
        self._current_token = None

    @contextmanager
    def stage(self, name: str):
        rec = StageRecord(name=name)
        self.stages.append(rec)
        t0 = time.monotonic()
        tok = _current_stage.set(rec)
        span_cm = _tracer().start_as_current_span(name) if _OTEL else None
        span = span_cm.__enter__() if span_cm else None
        try:
            yield rec
        except Exception as exc:  # failures visible, not swallowed
            rec.status = "error"
            rec.error = {"type": type(exc).__name__, "message": str(exc)}
            if span is not None:
                span.record_exception(exc)
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            self._log.warning("stage.error", stage=name, **rec.error)
            raise
        finally:
            rec.duration_ms = (time.monotonic() - t0) * 1000.0
            if span is not None:
                for k, v in rec.attrs.items():
                    try:
                        span.set_attribute(k, v if isinstance(v, (str, int, float, bool)) else str(v))
                    except Exception:
                        pass
                if rec.status == "degraded":
                    span.set_attribute("stage.status", "degraded")
                span_cm.__exit__(None, None, None)
            _current_stage.reset(tok)
            _metrics.observe_stage_latency(name, rec.duration_ms)

    def set_answer(self, answer: Any) -> None:
        self.answer = answer

    def capture_payloads(self) -> bool:
        return bool(self._cfg.get("obs.capture_payloads", True))

    @property
    def status(self) -> str:
        if any(s.status == "error" for s in self.stages):
            return "error"
        if any(s.status == "degraded" for s in self.stages):
            return "degraded"
        return "ok"

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "started_at": self.started_at,
            "duration_ms": round((time.monotonic() - self._t0) * 1000.0, 3),
            "status": self.status,
            "input": self.input,
            "stages": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 3),
                    "status": s.status,
                    "attrs": s.attrs,
                    "error": s.error,
                }
                for s in self.stages
            ],
            "answer": self.answer,
        }

    def trace_dir(self) -> Path:
        d = Path(self._cfg.get("obs.trace_dir", "traces"))
        if not d.is_absolute():
            # relative to cwd so `make run` from repo root writes ./traces/
            d = Path.cwd() / d
        return d

    def finish(self) -> Path:
        """Write the JSON trace file and clear the request context.

        A failed trace write is logged and then swallowed, deliberately: observability must not be
        able to take down the request it is observing. Previously an unwritable ``traces/`` dir
        propagated out of the API middleware as a 500 *and* skipped the context cleanup below,
        leaking the request contextvar into the next request on that worker. The failure is still
        visible — it is logged at error level, carrying the request id — it just isn't fatal.
        """
        path = self.trace_dir() / f"{self.request_id}.json"
        if self._finished:
            return path
        self._finished = True
        try:
            d = self.trace_dir()
            d.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(self.to_dict(), indent=2, default=str), encoding="utf-8")
        except Exception as exc:
            self._log.error(
                "trace.write_failed", path=str(path), type=type(exc).__name__, message=str(exc)
            )
        finally:
            _logging.clear_request_context()
            if self._current_token is not None:
                try:
                    _current.reset(self._current_token)
                except (ValueError, LookupError):
                    # token created in a different context (e.g. finished on another task) — best effort.
                    _current.set(None)
        return path

    def __enter__(self) -> "RequestTrace":
        return self

    def __exit__(self, *exc) -> None:
        self.finish()


# --- module-level API ----------------------------------------------------------------------------

def start_request(input_summary: dict | None = None, cfg=None) -> RequestTrace:
    """Open the root pipeline.request trace, bind request_id, return the handle."""
    cfg = cfg or load_config()
    maybe_setup_otlp(cfg)
    tr = RequestTrace(input_summary, cfg)
    tr._current_token = _current.set(tr)
    _logging.bind_request_id(tr.request_id)
    return tr


def current_trace() -> RequestTrace | None:
    return _current.get()


def current_request_id() -> str | None:
    tr = _current.get()
    return tr.request_id if tr else None


@contextmanager
def stage(name: str):
    """Open a stage span on the current request trace."""
    tr = _current.get()
    if tr is None:
        # tolerate un-traced calls (e.g. unit tests) with a throwaway record
        rec = StageRecord(name=name)
        yield rec
        return
    with tr.stage(name) as rec:
        yield rec


def annotate_stage(**attrs: Any) -> None:
    """Attach detail to the current stage (no-op when nothing is being traced).

    Without this, stage records carried only a name/duration/status, so a trace told you a run took
    N ms but not what it actually produced (how many changes, how many items ingested).
    """
    rec = _current_stage.get()
    if rec is not None:
        rec.attrs.update(attrs)


def mark_degraded(reason: str, **attrs: Any) -> None:
    """Mark the current stage degraded (e.g. low-confidence OCR) without dropping data."""
    rec = _current_stage.get()
    if rec is not None:
        rec.status = "degraded"
        rec.attrs["degraded_reason"] = reason
        rec.attrs.update(attrs)


def record_llm_span(
    *,
    system: str,
    model: str,
    usage: dict | None = None,
    finish_reasons: list[str] | None = None,
    prompt: Any = None,
    response: Any = None,
    usd: float | None = None,
) -> None:
    """Set GenAI semconv attributes on the current stage; capture payloads per obs.capture_payloads.

    This is the **single funnel** for LLM telemetry — it also records tokens + cost into metrics, so
    callers should NOT additionally call ``metrics.observe_tokens/observe_cost`` (double-count risk).
    Cost is computed from ``usage``+``model`` when ``usd`` is not supplied. A failed call should pass
    ``usage={"input_tokens": N, "output_tokens": 0}`` and ``usd=cost_on_failure(N, model).usd``.
    """
    rec = _current_stage.get()
    tr = _current.get()
    if rec is None:
        # Called outside a stage -> telemetry would be lost. Surface it, don't swallow silently.
        _logging.get_logger("delta_chat.trace").warning(
            "record_llm_span.no_stage", model=model,
            hint="call within `with tracing.stage('llm.call'):`",
        )
        return
    in_tok, out_tok = 0, 0
    if usage:
        in_tok = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        out_tok = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
    rec.attrs.update(
        {
            "gen_ai.system": system,
            "gen_ai.request.model": model,
            "gen_ai.usage.input_tokens": in_tok,
            "gen_ai.usage.output_tokens": out_tok,
        }
    )
    if finish_reasons is not None:
        rec.attrs["gen_ai.response.finish_reasons"] = finish_reasons

    if usd is None:
        usd = _cost.cost_from_usage(usage, model).usd
    rec.attrs["usd"] = usd
    _metrics.observe_cost(model, usd)
    _metrics.observe_tokens(model, in_tok, out_tok)

    capture = tr.capture_payloads() if tr else True
    if capture:
        rec.attrs["prompt"] = prompt
        rec.attrs["response"] = response
    else:  # redact-but-present: show a call happened without leaking content
        rec.attrs["prompt_redacted"] = len(str(prompt)) if prompt is not None else 0
        rec.attrs["response_redacted"] = len(str(response)) if response is not None else 0


def run_in_context(fn: Callable, *args: Any, **kwargs: Any) -> Callable[[], Any]:
    """Capture the current context and return a 0-arg callable that runs ``fn`` within it.

    FastAPI offloads sync PyMuPDF/Tesseract to a thread-pool executor, which would otherwise lose the
    bound ``request_id`` (and OTel span context). Use as::

        loop.run_in_executor(None, run_in_context(sync_fn, *args))
    """
    ctx = contextvars.copy_context()
    return lambda: ctx.run(fn, *args, **kwargs)
