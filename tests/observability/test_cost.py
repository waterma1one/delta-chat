"""Cost/token extraction across provider usage shapes, incl. failed calls + local models."""

from __future__ import annotations

from delta_chat.observability import cost


def test_openai_usage_tokens_and_cost():
    rec = cost.cost_from_usage({"prompt_tokens": 1000, "completion_tokens": 500}, "gpt-4o")
    assert (rec.input_tokens, rec.output_tokens) == (1000, 500)
    # 1000/1000*0.005 + 500/1000*0.015 = 0.005 + 0.0075
    assert rec.usd == 0.0125


def test_anthropic_usage_tokens_and_cost():
    rec = cost.cost_from_usage({"input_tokens": 2000, "output_tokens": 1000}, "claude-sonnet-5")
    assert (rec.input_tokens, rec.output_tokens) == (2000, 1000)
    assert rec.usd == round(2 * 0.003 + 1 * 0.015, 6)


def test_failed_call_records_input_cost_only():
    rec = cost.cost_on_failure(1000, "claude-sonnet-5")
    assert rec.output_tokens == 0
    assert rec.usd == round(1000 / 1000 * 0.003, 6) > 0


def test_local_model_is_free():
    rec = cost.cost_from_usage({"input_tokens": 5000, "output_tokens": 5000}, "local/llama-3")
    assert rec.usd == 0.0


def test_missing_usage_flags_estimated_and_does_not_crash():
    rec = cost.cost_from_usage({}, "gpt-4o")
    assert rec.estimated is True and rec.usd == 0.0


def test_usage_none_does_not_crash():
    rec = cost.cost_from_usage(None, "gpt-4o")
    assert rec.estimated is True and rec.usd == 0.0


def test_estimates_tokens_from_text_when_usage_absent():
    rec = cost.cost_from_usage({}, "gpt-4o", prompt_text="a " * 200, completion_text="b " * 40)
    assert rec.input_tokens > 0 and rec.output_tokens > 0
    assert rec.estimated is True and rec.usd > 0
