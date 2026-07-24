"""Tests for chat/llm.py — provider-agnostic client. `_call_provider` (the network-touching call)
is monkeypatched throughout, so these tests exercise provider selection, retry, and cost/tracing
behavior with no network access."""

from __future__ import annotations

import pytest

from delta_chat.chat import llm as llm_mod
from delta_chat.chat.types import LlmResult
from delta_chat.observability import tracing


def test_provider_selected_from_anthropic_key(monkeypatch, cfg):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)
    assert llm_mod.select_provider(cfg) == "anthropic"


def test_provider_selected_from_openai_key_when_no_anthropic_key(monkeypatch, cfg):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("LOCAL_LLM_BASE_URL", raising=False)
    assert llm_mod.select_provider(cfg) == "openai"


def test_provider_selected_from_local_base_url_when_no_keys(monkeypatch, cfg):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
    assert llm_mod.select_provider(cfg) == "local"


def test_llm_result_exposes_usage_model_and_finish_reason(monkeypatch, cfg):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(
        llm_mod, "_call_provider",
        lambda *a, **k: LlmResult(
            text="hi", blocks=None, usage={"input_tokens": 10, "output_tokens": 5},
            model="claude-sonnet-5", finish_reason="stop",
        ),
    )
    result = llm_mod.complete([{"role": "user", "content": "hi"}], cfg=cfg)
    assert result.usage == {"input_tokens": 10, "output_tokens": 5}
    assert result.model == "claude-sonnet-5"
    assert result.finish_reason == "stop"


def test_timeout_retries_bounded_then_raises_and_bills_input_tokens(monkeypatch, tmp_path, cfg):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(llm_mod.time, "sleep", lambda _s: None)
    cfg.data["llm"]["max_retries"] = 3
    cfg.data["obs"]["trace_dir"] = str(tmp_path / "traces")
    cfg.data["obs"]["otlp_enabled"] = False

    call_count = {"n": 0}

    def _boom(*_a, **_k):
        call_count["n"] += 1
        raise TimeoutError("provider timed out")

    monkeypatch.setattr(llm_mod, "_call_provider", _boom)

    tr = tracing.start_request({"query": "test"}, cfg=cfg)
    with pytest.raises(TimeoutError):
        llm_mod.complete([{"role": "user", "content": "hi " * 50}], cfg=cfg)
    tr.finish()

    assert call_count["n"] == 3

    llm_stage = next(s for s in tr.stages if s.name == "llm.call")
    assert llm_stage.status == "error"
    assert llm_stage.attrs.get("gen_ai.usage.output_tokens") == 0
    assert llm_stage.attrs.get("gen_ai.usage.input_tokens", 0) > 0
    assert llm_stage.attrs.get("usd") is not None


def test_non_retryable_error_fails_fast_without_backoff(monkeypatch, cfg):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setattr(llm_mod.time, "sleep", lambda _s: None)
    cfg.data["llm"]["max_retries"] = 3

    call_count = {"n": 0}

    def _bad_request(*_a, **_k):
        call_count["n"] += 1
        raise ValueError("400 bad request")

    monkeypatch.setattr(llm_mod, "_call_provider", _bad_request)

    with pytest.raises(ValueError):
        llm_mod.complete([{"role": "user", "content": "hi"}], cfg=cfg)

    assert call_count["n"] == 1
