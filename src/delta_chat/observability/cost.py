"""Token counts + USD cost per LLM call, behind one interface.

Prefers the provider's authoritative ``usage`` fields (OpenAI: ``prompt_tokens``/``completion_tokens``;
Anthropic: ``input_tokens``/``output_tokens``). Prices via a small maintained table (extended from
litellm when available). Local models cost 0.0. A **failed/timeout** call still bills input tokens, so
:func:`cost_on_failure` records a partial cost with ``output_tokens == 0``. When usage is absent, tokens
are **estimated** from the prompt/response text (tiktoken if installed, else a chars/4 heuristic).
"""

from __future__ import annotations

from dataclasses import dataclass

# USD per 1K tokens (input, output). Illustrative mid-2026 public rates — pull live values from
# litellm/tokencost in production. Matched by longest substring so version suffixes still resolve.
_PRICE_PER_1K: dict[str, tuple[float, float]] = {
    "claude-opus": (0.015, 0.075),
    "claude-sonnet": (0.003, 0.015),
    "claude-haiku": (0.001, 0.005),
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.005, 0.015),
    "gpt-4": (0.03, 0.06),
}


@dataclass
class CostRecord:
    input_tokens: int
    output_tokens: int
    usd: float
    model: str
    estimated: bool  # True if tokens were estimated (no provider usage)


def _is_local(model: str) -> bool:
    m = model.lower()
    return m.startswith("local") or "ollama" in m or "vllm" in m or m.startswith("llama")


def _price(model: str) -> tuple[float, float] | None:
    m = model.lower()
    # longest matching key wins, so "gpt-4o-mini" beats "gpt-4o" regardless of dict order.
    matches = [(k, v) for k, v in _PRICE_PER_1K.items() if k in m]
    if matches:
        return max(matches, key=lambda kv: len(kv[0]))[1]
    try:  # pragma: no cover - optional dependency path
        import litellm  # type: ignore

        info = litellm.model_cost.get(model)  # type: ignore[attr-defined]
        if info:
            return (
                float(info.get("input_cost_per_token", 0.0)) * 1000.0,
                float(info.get("output_cost_per_token", 0.0)) * 1000.0,
            )
    except Exception:
        pass
    return None


def _estimate_tokens(text: str | None, model: str) -> int:
    if not text:
        return 0
    try:  # pragma: no cover - optional dependency path
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(model)
        except Exception:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)  # rough chars/4 fallback


def _extract_tokens(usage: dict | None) -> tuple[int, int]:
    usage = usage or {}
    in_tok = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    out_tok = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    return int(in_tok), int(out_tok)


def compute_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    if _is_local(model):
        return 0.0
    price = _price(model)
    if price is None:
        return 0.0
    in_rate, out_rate = price
    return round((input_tokens / 1000.0) * in_rate + (output_tokens / 1000.0) * out_rate, 6)


def cost_from_usage(
    usage: dict | None,
    model: str,
    *,
    prompt_text: str | None = None,
    completion_text: str | None = None,
) -> CostRecord:
    """Normalize OpenAI/Anthropic usage shapes -> tokens; price for ``model`` (local -> 0.0).

    Tolerates ``usage=None``. When usage is empty/absent, estimates tokens from the supplied
    ``prompt_text``/``completion_text`` (tiktoken or chars/4) and marks ``estimated=True``.
    """
    in_tok, out_tok = _extract_tokens(usage)
    estimated = False
    if in_tok == 0 and out_tok == 0 and (prompt_text or completion_text):
        in_tok = _estimate_tokens(prompt_text, model)
        out_tok = _estimate_tokens(completion_text, model)
        estimated = True
    elif not usage:
        estimated = True
    return CostRecord(in_tok, out_tok, compute_usd(model, in_tok, out_tok), model, estimated)


def cost_on_failure(input_tokens: int, model: str) -> CostRecord:
    """Partial cost for a failed/timeout call: output_tokens=0, USD from input tokens only."""
    usd = compute_usd(model, input_tokens, 0)
    return CostRecord(int(input_tokens), 0, usd, model, estimated=False)
