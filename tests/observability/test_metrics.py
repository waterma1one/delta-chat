"""Prometheus metrics increment and render as text exposition."""

from __future__ import annotations

from delta_chat.observability import metrics


def test_metrics_increment_and_render():
    metrics.reset()
    metrics.observe_stage_latency("delta", 12.5)
    metrics.observe_tokens("claude-sonnet-5", 100, 40)
    metrics.observe_cost("claude-sonnet-5", 0.0021)
    metrics.observe_delta_count(7)
    metrics.observe_retrieval_hits(5)

    body, content_type = metrics.render_prometheus()
    assert "text/plain" in content_type
    assert "delta_chat_stage_latency_ms" in body
    assert 'delta_chat_tokens_total{direction="input",model="claude-sonnet-5"} 100.0' in body
    assert "delta_chat_delta_changes_total 7.0" in body
    assert "delta_chat_retrieval_hits_total 5.0" in body


def test_reset_clears_counters():
    metrics.reset()
    body, _ = metrics.render_prometheus()
    assert "delta_chat_delta_changes_total 7.0" not in body
