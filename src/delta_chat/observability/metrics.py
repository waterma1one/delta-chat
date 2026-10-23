"""Prometheus metrics: per-stage latency, tokens, cost, delta counts, retrieval hits.

Metrics live on a dedicated ``CollectorRegistry`` (not the global default) so tests can ``reset()`` to
a clean registry. The API mounts :func:`render_prometheus` at ``GET /metrics``.
"""

from __future__ import annotations

from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest


class _Metrics:
    def __init__(self) -> None:
        self.registry = CollectorRegistry()
        self.stage_latency = Histogram(
            "delta_chat_stage_latency_ms",
            "Per-stage latency in milliseconds",
            ["stage"],
            registry=self.registry,
            # ms-scale buckets (default prometheus buckets are seconds -> everything >10ms would
            # fall into +Inf and the histogram would be useless for percentiles).
            buckets=(5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, float("inf")),
        )
        self.tokens = Counter(
            "delta_chat_tokens_total",
            "LLM tokens",
            ["model", "direction"],
            registry=self.registry,
        )
        self.cost_usd = Counter(
            "delta_chat_cost_usd_total",
            "Estimated LLM cost in USD",
            ["model"],
            registry=self.registry,
        )
        self.delta_count = Counter(
            "delta_chat_delta_changes_total",
            "Number of delta changes produced",
            registry=self.registry,
        )
        self.retrieval_hits = Counter(
            "delta_chat_retrieval_hits_total",
            "Retrieval hits",
            registry=self.registry,
        )


_M = _Metrics()


def reset() -> None:
    """Recreate all metrics on a fresh registry (test isolation)."""
    global _M
    _M = _Metrics()


def observe_stage_latency(stage: str, ms: float) -> None:
    _M.stage_latency.labels(stage=stage).observe(ms)


def observe_tokens(model: str, input_tokens: int, output_tokens: int) -> None:
    if input_tokens:
        _M.tokens.labels(model=model, direction="input").inc(input_tokens)
    if output_tokens:
        _M.tokens.labels(model=model, direction="output").inc(output_tokens)


def observe_cost(model: str, usd: float) -> None:
    if usd:
        _M.cost_usd.labels(model=model).inc(usd)


def observe_delta_count(n: int) -> None:
    if n:
        _M.delta_count.inc(n)


def observe_retrieval_hits(n: int) -> None:
    if n:
        _M.retrieval_hits.inc(n)


def render_prometheus() -> tuple[str, str]:
    """Return (body, content_type) for the /metrics endpoint."""
    return generate_latest(_M.registry).decode("utf-8"), CONTENT_TYPE_LATEST
