"""Tests for chat/llm.py — provider-agnostic client. `_call_provider` (the network-touching call)
is monkeypatched throughout, so these tests exercise provider selection, retry, and cost/tracing
behavior with no network access."""

from __future__ import annotations

import pytest

from delta_chat.chat import llm as llm_mod
from delta_chat.chat.types import LlmResult
from delta_chat.observability import tracing


def _clear_provider_env(monkeypatch, cfg):
    """Make a provider-selection test hermetic regardless of the ambient environment. Clears every
    provider env var (a developer may have a real GEMINI_API_KEY exported) AND forces
    llm.provider=auto — litellm's `load_dotenv()` on import leaks a repo `.env`'s
    DELTA_CHAT__LLM__PROVIDER override into os.environ, which load_config would otherwise honor."""
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "GEMINI_API_KEY", "LOCAL_LLM_BASE_URL",
                "DELTA_CHAT__LLM__PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    cfg.data["llm"]["provider"] = "auto"


def test_provider_selected_from_anthropic_key(monkeypatch, cfg):
    _clear_provider_env(monkeypatch, cfg)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    assert llm_mod.select_provider(cfg) == "anthropic"


def test_provider_selected_from_openai_key_when_no_anthropic_key(monkeypatch, cfg):
    _clear_provider_env(monkeypatch, cfg)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    assert llm_mod.select_provider(cfg) == "openai"


def test_provider_selected_from_local_base_url_when_no_keys(monkeypatch, cfg):
    _clear_provider_env(monkeypatch, cfg)
    monkeypatch.setenv("LOCAL_LLM_BASE_URL", "http://localhost:11434")
    assert llm_mod.select_provider(cfg) == "local"


def test_provider_selected_from_gemini_key(monkeypatch, cfg):
    _clear_provider_env(monkeypatch, cfg)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-key")
    assert llm_mod.select_provider(cfg) == "gemini"


def test_gemini_provider_prefixes_model_for_litellm(monkeypatch, cfg):
    """litellm routes to Gemini via a `gemini/<model>` model string; the raw config model
    (e.g. 'gemini-2.0-flash') must be prefixed so the call actually reaches Gemini."""
    captured = {}

    def _fake_completion(**kwargs):
        captured.update(kwargs)
        raise RuntimeError("stop after capturing kwargs")

    monkeypatch.setattr(llm_mod._litellm, "completion", _fake_completion)
    with pytest.raises(RuntimeError):
        llm_mod._call_provider(
            "gemini", "gemini-2.0-flash", [{"role": "user", "content": "hi"}],
            documents=None, response_format=None, cfg=cfg, timeout_s=5.0,
        )
    assert captured["model"] == "gemini/gemini-2.0-flash"


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


def test_litellm_drops_unsupported_params():
    """Regression: claude-sonnet-5 rejects temperature=0 (only 1 is allowed). A provider-agnostic
    client must drop the unsupported param, not 500. `drop_params` is set at import time."""
    assert llm_mod._litellm.drop_params is True
