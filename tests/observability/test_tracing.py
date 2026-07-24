"""Tracing: span tree, unconditional JSON trace file, failure visibility, payload capture, ctx-copy."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest
import structlog

from delta_chat.config import load_config
from delta_chat.observability import tracing


@pytest.fixture
def cfg(tmp_path, monkeypatch):
    monkeypatch.setenv("DELTA_CHAT__OBS__TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("DELTA_CHAT__OBS__OTLP_ENABLED", "false")
    return load_config()


def test_span_tree_and_trace_file_written_unconditionally(cfg):
    tr = tracing.start_request({"query": "what changed?"}, cfg)
    for name in tracing.STAGE_NAMES:
        with tracing.stage(name):
            pass
    tr.set_answer({"state": "Grounded"})
    path = tr.finish()

    assert path.exists()  # written with zero services running
    data = json.loads(path.read_text())
    assert {"request_id", "trace_id", "started_at", "duration_ms", "status", "input", "stages", "answer"} <= data.keys()
    assert [s["name"] for s in data["stages"]] == list(tracing.STAGE_NAMES)
    assert data["status"] == "ok"
    assert tracing.is_otlp_configured() is False


def test_exception_in_stage_marks_error_and_reraises(cfg):
    tr = tracing.start_request({}, cfg)
    with pytest.raises(ValueError):
        with tracing.stage("delta"):
            raise ValueError("boom")
    path = tr.finish()
    stage = json.loads(path.read_text())["stages"][0]
    assert stage["status"] == "error"
    assert stage["error"]["type"] == "ValueError" and "boom" in stage["error"]["message"]
    assert json.loads(path.read_text())["status"] == "error"


def test_degraded_stage_is_not_error_and_keeps_data(cfg):
    tr = tracing.start_request({}, cfg)
    with tracing.stage("ingest"):
        tracing.mark_degraded("low OCR confidence", **{"ocr.mean_word_conf": 0.31})
    data = json.loads(tr.finish().read_text())
    st = data["stages"][0]
    assert st["status"] == "degraded" and data["status"] == "degraded"
    assert st["attrs"]["ocr.mean_word_conf"] == 0.31


def test_llm_payload_captured_when_enabled(cfg):
    tr = tracing.start_request({}, cfg)  # capture_payloads default true
    with tracing.stage("llm.call"):
        tracing.record_llm_span(
            system="anthropic",
            model="claude-sonnet-5",
            usage={"input_tokens": 100, "output_tokens": 20},
            finish_reasons=["end_turn"],
            prompt="the prompt",
            response="the answer",
            usd=0.001,
        )
    attrs = json.loads(tr.finish().read_text())["stages"][0]["attrs"]
    assert attrs["gen_ai.request.model"] == "claude-sonnet-5"
    assert attrs["gen_ai.usage.input_tokens"] == 100
    assert attrs["prompt"] == "the prompt" and attrs["response"] == "the answer"


def test_llm_payload_redacted_when_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("DELTA_CHAT__OBS__TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("DELTA_CHAT__OBS__CAPTURE_PAYLOADS", "false")
    cfg = load_config()
    tr = tracing.start_request({}, cfg)
    with tracing.stage("llm.call"):
        tracing.record_llm_span(system="openai", model="gpt-4o", prompt="secret", response="hi")
    attrs = json.loads(tr.finish().read_text())["stages"][0]["attrs"]
    assert "prompt" not in attrs
    assert attrs["prompt_redacted"] == len("secret") and attrs["response_redacted"] == len("hi")


def test_current_trace_reset_after_finish(cfg):
    tr = tracing.start_request({}, cfg)
    assert tracing.current_trace() is tr
    tr.finish()
    assert tracing.current_trace() is None


def test_llm_failure_records_error_zero_output_and_partial_cost(cfg):
    from delta_chat.observability import cost

    tr = tracing.start_request({}, cfg)
    model = "claude-sonnet-5"
    with pytest.raises(TimeoutError):
        with tracing.stage("llm.call"):
            c = cost.cost_on_failure(1000, model)  # request sent -> input tokens billed
            tracing.record_llm_span(
                system="anthropic",
                model=model,
                usage={"input_tokens": 1000, "output_tokens": 0},
                usd=c.usd,
            )
            raise TimeoutError("llm timed out")
    st = json.loads(tr.finish().read_text())["stages"][0]
    assert st["status"] == "error"
    assert st["attrs"]["gen_ai.usage.output_tokens"] == 0
    assert st["attrs"]["usd"] > 0


def test_run_in_context_propagates_request_id_into_worker_thread(cfg):
    tr = tracing.start_request({}, cfg)

    def worker() -> str | None:
        return structlog.contextvars.get_contextvars().get("request_id")

    wrapped = tracing.run_in_context(worker)
    with ThreadPoolExecutor(max_workers=1) as ex:
        got = ex.submit(wrapped).result()
    assert got == tr.request_id
    tr.finish()
