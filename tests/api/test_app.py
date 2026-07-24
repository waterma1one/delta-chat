"""Tests for api/app.py — FastAPI endpoints. LLM is mocked at the `answer()` call site (canned
`ChatAnswer`), ingest/delta run for real against the golden native sample pair (section-04)."""

from __future__ import annotations

import json
from pathlib import Path

from delta_chat.chat.types import ChatAnswer, Citation, Claim

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_delta_returns_report_json_with_counts(client, registered_pair):
    pid_a, pid_b = registered_pair
    resp = client.post("/delta", json={"pid_a": pid_a, "pid_b": pid_b})
    assert resp.status_code == 200
    body = resp.json()
    assert "counts" in body["report_json"]
    assert body["report_json"]["changes"]
    assert all(k.startswith("delta:CHG-") for k in body["report_json"]["changes"])
    assert body["report_md_url"].startswith("/report?delta_id=")


def test_report_endpoint_renders_html_with_change_count(client, registered_pair):
    pid_a, pid_b = registered_pair
    resp = client.post("/delta", json={"pid_a": pid_a, "pid_b": pid_b})
    body = resp.json()
    delta_id = body["delta_id"]

    r2 = client.get(f"/report?delta_id={delta_id}")

    assert r2.status_code == 200
    assert r2.headers["content-type"].startswith("text/html")
    total = sum(body["report_json"]["counts"].values())
    assert str(total) in r2.text


def test_report_unknown_delta_id_is_404(client):
    resp = client.get("/report?delta_id=does-not-exist")
    assert resp.status_code == 404


def test_chat_returns_grounded_answer_with_resolvable_citations(client, registered_pair, mock_answer):
    pid_a, pid_b = registered_pair
    mock_answer.queue(ChatAnswer(
        state="Grounded", text="the setpoint changed",
        claims=[Claim(
            text="the setpoint changed",
            citations=[Citation(citation_id="doc_a:0:notes:1", source_type="doc_a", page=0, sheet="1", bbox=None)],
            verified=True,
        )],
        used_chunks=["doc_a:0:notes:1"], refusal_reason=None,
    ))

    resp = client.post("/chat", json={"pid_a": pid_a, "pid_b": pid_b, "query": "what changed?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "Grounded"
    assert body["claims"]
    for claim in body["claims"]:
        assert len(claim["citations"]) >= 1
        for cit in claim["citations"]:
            assert set(cit) >= {"citation_id", "page", "sheet", "bbox", "source_type"}
    assert len(mock_answer.calls) == 1
    assert mock_answer.calls[0]["query"] == "what changed?"


def test_chat_rejected_is_200_not_an_error(client, registered_pair, mock_answer):
    pid_a, pid_b = registered_pair
    mock_answer.queue(ChatAnswer(
        state="Rejected", text="", claims=[], used_chunks=[],
        refusal_reason="INSUFFICIENT_EVIDENCE: nothing on that topic was retrieved",
    ))

    resp = client.post("/chat", json={"pid_a": pid_a, "pid_b": pid_b, "query": "unanswerable?"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["state"] == "Rejected"
    assert body["refusal_reason"]


def test_chat_without_a_configured_provider_is_503_not_500(client, registered_pair, monkeypatch):
    """A keyless deployment is a config state, not a server fault: ingest/delta/report all work
    without a key and only answer generation needs one, so the UI must be able to say "configure a
    key" instead of surfacing an opaque 500."""
    from delta_chat.api import app as app_module
    from delta_chat.chat.llm import LlmUnavailableError

    def _no_provider(*a, **k):
        raise LlmUnavailableError("no LLM provider configured: set ANTHROPIC_API_KEY, ...")

    monkeypatch.setattr(app_module, "answer", _no_provider)
    pid_a, pid_b = registered_pair

    resp = client.post("/chat", json={"pid_a": pid_a, "pid_b": pid_b, "query": "what changed?"})

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "no LLM provider" in detail
    # ingest/delta still work without a key, and the client is told so
    assert "do not require a key" in detail


def test_metrics_exposes_prometheus_text(client):
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert "delta_chat_stage_latency_ms" in resp.text


def test_healthz_ok(client):
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_request_id_header_present_on_every_response(client, registered_pair):
    pid_a, _pid_b = registered_pair
    resp = client.post("/ingest", json={"pid": pid_a})
    headers = {k.lower(): v for k, v in resp.headers.items()}
    assert "x-request-id" in headers
    assert headers["x-request-id"]


def test_inbound_request_id_is_honored(client, registered_pair):
    pid_a, _pid_b = registered_pair
    resp = client.post("/ingest", json={"pid": pid_a}, headers={"X-Request-ID": "my-fixed-id"})
    headers = {k.lower(): v for k, v in resp.headers.items()}
    assert headers["x-request-id"] == "my-fixed-id"


def test_request_id_propagates_into_executor_thread_stage_logs(client, registered_pair, capsys):
    pid_a, _pid_b = registered_pair
    resp = client.post("/ingest", json={"pid": pid_a})
    request_id = resp.headers.get("x-request-id")
    assert request_id

    out = capsys.readouterr().out
    log_lines = []
    for line in out.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            log_lines.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    # `_parse_with_log` logs "ingest.parse" from inside the thread-pool-offloaded call; its
    # request_id must match the response header, proving contextvars propagated into the thread.
    ingest_lines = [ln for ln in log_lines if ln.get("event") == "ingest.parse"]
    assert ingest_lines
    assert all(ln.get("request_id") == request_id for ln in ingest_lines)


def test_unknown_pid_returns_4xx_with_clear_detail(client):
    resp = client.post("/ingest", json={"pid": "totally-unknown-pid-xyz"})
    assert 400 <= resp.status_code < 500
    assert resp.json().get("detail")


def test_pid_error_detail_does_not_leak_server_filesystem_paths(client):
    # A bare relative pid resolves against the resolver's allowed-roots (repo root) and fails to
    # exist; the response must not echo the resolved absolute server path or the allowlist itself.
    resp = client.post("/ingest", json={"pid": "does-not-exist-anywhere.pdf"})
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "/Users/" not in detail
    assert str(REPO_ROOT) not in detail


def test_pid_outside_allowed_roots_error_does_not_leak_allowlist(client, tmp_path):
    # Echoing the client's own submitted pid back is fine (they already know it); what must NOT
    # leak is the server's allowlist configuration ("outside the allowed roots [...]").
    outside_file = tmp_path / "secret.pdf"
    outside_file.write_bytes(b"%PDF-1.4\n")
    resp = client.post("/ingest", json={"pid": str(outside_file)})
    assert resp.status_code == 404
    detail = resp.json()["detail"]
    assert "allowed roots" not in detail
    assert str(REPO_ROOT) not in detail


def test_unhandled_exception_response_body_is_generic(app_cfg, registered_pair, monkeypatch):
    from fastapi.testclient import TestClient

    from delta_chat.api.app import create_app

    pid_a, pid_b = registered_pair

    def _boom(query, index, cfg):
        raise RuntimeError("sensitive internal detail: /etc/passwd or similar")

    monkeypatch.setattr("delta_chat.api.app.answer", _boom)
    client = TestClient(create_app(cfg=app_cfg), raise_server_exceptions=False)

    resp = client.post("/chat", json={"pid_a": pid_a, "pid_b": pid_b, "query": "what changed?"})

    assert resp.status_code == 500
    detail = resp.json()["detail"]
    assert "sensitive internal detail" not in detail
    assert detail == "internal server error"


def test_markup_returns_annotated_pdf(client, registered_pair):
    # markup.overlay (section-09) now exists — /markup renders a real annotated PDF rather than
    # the 501 fallback this test asserted before section-09 was built.
    pid_a, pid_b = registered_pair
    resp = client.post("/delta", json={"pid_a": pid_a, "pid_b": pid_b})
    delta_id = resp.json()["delta_id"]

    r2 = client.get(f"/markup?delta_id={delta_id}")

    assert r2.status_code == 200
    assert r2.headers["content-type"] == "application/pdf"
    assert r2.content.startswith(b"%PDF-")


def test_markup_returns_501_when_overlay_unavailable(client, registered_pair, monkeypatch):
    # Simulate section-09 being absent: `from ..markup import overlay` inside the endpoint only
    # raises ImportError if the submodule isn't already cached as a package attribute AND a fresh
    # import of it fails — so both the cached attribute and sys.modules entry must be cleared.
    import sys

    import delta_chat.markup as markup_pkg

    pid_a, pid_b = registered_pair
    resp = client.post("/delta", json={"pid_a": pid_a, "pid_b": pid_b})
    delta_id = resp.json()["delta_id"]

    monkeypatch.delattr(markup_pkg, "overlay", raising=False)
    monkeypatch.setitem(sys.modules, "delta_chat.markup.overlay", None)

    r2 = client.get(f"/markup?delta_id={delta_id}")

    assert r2.status_code == 501
    assert r2.json().get("detail")


def test_markup_unknown_delta_id_is_404_before_checking_overlay(client):
    resp = client.get("/markup?delta_id=does-not-exist")
    assert resp.status_code == 404


def test_ingest_returns_compact_document_summary(client, registered_pair):
    pid_a, _pid_b = registered_pair
    resp = client.post("/ingest", json={"pid": pid_a})
    assert resp.status_code == 200
    body = resp.json()
    assert set(body) == {"pid", "format", "page_count", "revision_label", "item_count", "region_count"}
    assert body["item_count"] > 0


def test_unhandled_exception_in_answer_becomes_clean_500_with_request_id(app_cfg, registered_pair, monkeypatch):
    # Starlette's ServerErrorMiddleware always re-raises after sending its response (so log/debug
    # tooling upstream can see the crash) — the default TestClient re-raises that to the caller
    # too. `raise_server_exceptions=False` is the standard way to instead inspect the Response
    # the app actually sent, which is what we're asserting on here.
    from fastapi.testclient import TestClient

    from delta_chat.api.app import create_app

    pid_a, pid_b = registered_pair

    def _boom(query, index, cfg):
        raise RuntimeError("boom from inside answer()")

    monkeypatch.setattr("delta_chat.api.app.answer", _boom)
    client = TestClient(create_app(cfg=app_cfg), raise_server_exceptions=False)

    resp = client.post("/chat", json={"pid_a": pid_a, "pid_b": pid_b, "query": "what changed?"})

    assert resp.status_code == 500
    assert resp.json().get("detail")
    headers = {k.lower(): v for k, v in resp.headers.items()}
    assert headers.get("x-request-id")


def test_delta_id_does_not_collide_on_unescaped_separator(client):
    from delta_chat.api.app import _delta_id

    id_1 = _delta_id("a|b", "c")
    id_2 = _delta_id("a", "b|c")
    assert id_1 != id_2
