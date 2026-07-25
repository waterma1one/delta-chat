"""Provider-agnostic completion client — one interface behind which the provider (Anthropic /
OpenAI / local) is chosen by whichever API key env var is present. Wraps every call in a
`llm.call` span, bounded exponential-backoff retry, and cost recording (including on failure —
a timed-out call still billed its input tokens).

litellm is the unifying call layer so provider switching is config, not code. The Anthropic
Citations API (custom-content blocks) is passed through best-effort when `documents` is supplied
against an Anthropic model; it is mutually exclusive with `response_format` (strict JSON) per the
Anthropic API — never pass both in the same call.
"""

from __future__ import annotations

import os
import time
from typing import Any

from ..config import load_config
from ..observability import cost as _cost
from ..observability import tracing
from .types import LlmResult

try:
    import litellm as _litellm

    # Different models accept different sampling params — e.g. claude-sonnet-5 rejects any
    # temperature other than 1, while most models accept temperature=0. For a genuinely
    # provider-agnostic client, an unsupported param must be dropped, not turned into a hard
    # UnsupportedParamsError (which otherwise 500s a chat request the moment the pinned model
    # disagrees with `llm.temperature`). This is the one place that promise is enforced.
    _litellm.drop_params = True

    _RETRYABLE: tuple[type[BaseException], ...] = (
        TimeoutError,
        ConnectionError,
        _litellm.exceptions.Timeout,
        _litellm.exceptions.RateLimitError,
        _litellm.exceptions.APIConnectionError,
        _litellm.exceptions.ServiceUnavailableError,
    )
except Exception:  # pragma: no cover - litellm is a hard dependency; guard is defensive only
    _RETRYABLE = (TimeoutError, ConnectionError)


class LlmUnavailableError(RuntimeError):
    """No usable LLM provider is configured.

    A distinct type (rather than a bare RuntimeError) so callers — notably the API layer — can
    map *this* condition to a 503 "configure a key" response without also swallowing unrelated
    RuntimeErrors raised from inside a provider call.
    """


def select_provider(cfg=None) -> str:
    """Pick a provider by config override, else by whichever API key env var is present.

    Order: ANTHROPIC_API_KEY -> anthropic; OPENAI_API_KEY -> openai; GEMINI_API_KEY -> gemini;
    LOCAL_LLM_BASE_URL -> local. Keys come from env only (never config), per the security
    convention in .env.example.
    """
    cfg = cfg or load_config()
    configured = cfg.get("llm.provider", "auto")
    if configured and configured != "auto":
        return configured
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("GEMINI_API_KEY"):
        return "gemini"
    if os.environ.get("LOCAL_LLM_BASE_URL"):
        return "local"
    raise LlmUnavailableError(
        "no LLM provider configured: set ANTHROPIC_API_KEY, OPENAI_API_KEY, GEMINI_API_KEY, "
        "or LOCAL_LLM_BASE_URL"
    )


def _flatten_messages(messages: list[dict]) -> str:
    return "\n".join(str(m.get("content", "")) for m in messages)


def _backoff_seconds(attempt: int) -> float:
    return min(0.5 * (2 ** (attempt - 1)), 8.0)


def _with_documents(messages: list[dict], documents: list[dict] | None, provider: str) -> list[dict]:
    """Best-effort Anthropic custom-content blocks for the Citations API. Other providers ignore
    `documents` entirely (JSON-schema grounding is the provider-agnostic default path)."""
    if not documents or provider != "anthropic":
        return messages
    doc_blocks = [
        {
            "type": "document",
            "source": {"type": "text", "media_type": "text/plain", "data": d["text"]},
            "title": d.get("citation_id", ""),
            "citations": {"enabled": True},
        }
        for d in documents
    ]
    out = list(messages)
    for i, m in enumerate(out):
        if m.get("role") == "user":
            content = m["content"]
            content_list = content if isinstance(content, list) else [{"type": "text", "text": content}]
            out[i] = {**m, "content": doc_blocks + content_list}
            break
    return out


def _call_provider(
    provider: str,
    model: str,
    messages: list[dict],
    *,
    documents: list[dict] | None,
    response_format: dict | None,
    cfg,
    timeout_s: float,
) -> LlmResult:
    """The single network-touching call. Isolated so tests can monkeypatch it and exercise the
    retry/cost/tracing logic in `complete()` without any network access."""
    if response_format is not None and documents:
        raise ValueError("response_format and documents (Citations API) are mutually exclusive")

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": _with_documents(messages, documents, provider),
        "timeout": timeout_s,
        "temperature": cfg.get("llm.temperature", 0.0),
        "max_tokens": cfg.get("llm.max_tokens", 1024),
    }
    if provider == "local":
        kwargs["model"] = f"openai/{model}"
        kwargs["api_base"] = os.environ.get("LOCAL_LLM_BASE_URL")
        kwargs["api_key"] = "local"
    elif provider == "gemini":
        # litellm routes to Google Gemini via a `gemini/<model>` model string and reads
        # GEMINI_API_KEY from the env itself. Config carries the bare model id (e.g.
        # "gemini-2.0-flash") so the same value stays provider-neutral.
        kwargs["model"] = f"gemini/{model}"
    if response_format is not None:
        kwargs["response_format"] = response_format

    resp = _litellm.completion(**kwargs)
    choice = resp.choices[0]
    text = choice.message.content or ""
    usage = resp.usage
    usage_dict = {
        "input_tokens": getattr(usage, "prompt_tokens", 0) or 0,
        "output_tokens": getattr(usage, "completion_tokens", 0) or 0,
    }
    blocks = None
    if documents and provider == "anthropic":
        fields = getattr(choice.message, "provider_specific_fields", None) or {}
        blocks = fields.get("citations")
    return LlmResult(
        text=text, blocks=blocks, usage=usage_dict, model=model, finish_reason=choice.finish_reason
    )


def _record_failure(provider: str, model: str, prompt_text: str) -> None:
    estimated = _cost.cost_from_usage(None, model, prompt_text=prompt_text)
    failure_cost = _cost.cost_on_failure(estimated.input_tokens, model)
    tracing.record_llm_span(
        system=provider,
        model=model,
        usage={"input_tokens": failure_cost.input_tokens, "output_tokens": 0},
        usd=failure_cost.usd,
        prompt=prompt_text,
        response=None,
    )


def complete(
    messages: list[dict],
    *,
    model: str | None = None,
    response_format: dict | None = None,
    documents: list[dict] | None = None,
    cfg=None,
) -> LlmResult:
    """One call behind one interface. Never returns a silent empty answer: on timeout/rate-limit
    exhaustion the failure is traced (span ERROR + exception recorded), cost is billed for the
    input tokens actually sent, and the exception is re-raised."""
    cfg = cfg or load_config()
    model = model or cfg.get("llm.model")
    provider = select_provider(cfg)
    max_retries = max(1, int(cfg.get("llm.max_retries", 3)))
    timeout_s = cfg.get("llm.timeout_s", 30)
    # Include `documents` text in the failure-cost estimate too — on the Citations API path they
    # are the bulk of what's actually sent, and _with_documents() only merges them into a local
    # copy inside _call_provider, so they'd otherwise be invisible to cost-on-failure accounting.
    prompt_text = _flatten_messages(messages)
    if documents:
        prompt_text += "\n" + "\n".join(str(d.get("text", "")) for d in documents)

    with tracing.stage("llm.call"):
        for attempt in range(1, max_retries + 1):
            try:
                result = _call_provider(
                    provider, model, messages,
                    documents=documents, response_format=response_format,
                    cfg=cfg, timeout_s=timeout_s,
                )
            except _RETRYABLE:
                if attempt < max_retries:
                    time.sleep(_backoff_seconds(attempt))
                    continue
                _record_failure(provider, model, prompt_text)
                raise
            except Exception:
                # non-retryable (e.g. bad request, auth) -> fail fast, no backoff loop
                _record_failure(provider, model, prompt_text)
                raise
            else:
                tracing.record_llm_span(
                    system=provider,
                    model=model,
                    usage=result.usage,
                    finish_reasons=[result.finish_reason] if result.finish_reason else None,
                    prompt=prompt_text,
                    response=result.text,
                )
                return result
    raise AssertionError("unreachable")  # pragma: no cover - loop always returns or raises
