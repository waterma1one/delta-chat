"""Structured JSON logging with a request_id correlation key.

``request_id`` is bound once at request entry via ``structlog.contextvars`` and then rides along on
every log line for that request — the same id is written into the trace file, so logs and traces join
on one key.
"""

from __future__ import annotations

import structlog

_configured = False


def configure_logging(json: bool = True) -> None:
    """Configure structlog to emit JSON lines carrying contextvars (idempotent)."""
    global _configured
    if _configured:
        return
    renderer = structlog.processors.JSONRenderer() if json else structlog.dev.ConsoleRenderer()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            renderer,
        ],
        cache_logger_on_first_use=True,
    )
    _configured = True


def bind_request_id(request_id: str) -> None:
    """Bind request_id into structlog contextvars for the current (async) context."""
    structlog.contextvars.bind_contextvars(request_id=request_id)


def clear_request_context() -> None:
    structlog.contextvars.clear_contextvars()


def get_logger(name: str | None = None):
    """Return a structlog logger; every line carries the bound request_id."""
    configure_logging()
    return structlog.get_logger(name)
