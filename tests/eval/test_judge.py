"""Tests for eval/judge.py — order-swap averaging, Cohen's kappa, cross-family config assertion.
The LLM client is monkeypatched throughout; no network."""

from __future__ import annotations

import copy
import json

import pytest

from delta_chat.chat.types import LlmResult
from delta_chat.config import Config, load_config

from eval.judge import assert_cross_family, cohen_kappa, judge_pairwise


def test_judge_pairwise_runs_both_orderings_and_averages(monkeypatch):
    calls = []

    def _fake_complete(messages, *, model=None, response_format=None, documents=None, cfg=None):
        user_content = messages[-1]["content"]
        calls.append(user_content)
        # first candidate presented always scores 4, second always scores 2 (deliberate bias)
        return LlmResult(
            text=json.dumps({"first_score": 4, "second_score": 2, "rationale": "biased toward first"}),
            blocks=None, usage={"input_tokens": 10, "output_tokens": 5}, model="m", finish_reason="stop",
        )

    monkeypatch.setattr("delta_chat.chat.llm.complete", _fake_complete)

    result = judge_pairwise("Candidate A text", "Candidate B text", model="judge-model")

    assert len(calls) == 2
    # (A,B) call and (B,A) call both happened -> position-bias-inducing scores average out to a tie
    assert result["scores"]["a"] == 3.0
    assert result["scores"]["b"] == 3.0
    assert result["winner"] == "tie"


def test_cohen_kappa_matches_hand_computed_value_on_known_fixture():
    # 10 items, 2 categories (0/1). Agree on 7, disagree on 3.
    judge_labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
    human_labels = [0, 0, 0, 1, 1, 1, 1, 1, 0, 1]
    # po = 7/10 = 0.7
    # judge: P(0)=5/10=0.5, P(1)=5/10=0.5; human: P(0)=4/10=0.4, P(1)=6/10=0.6
    # pe = 0.5*0.4 + 0.5*0.6 = 0.2 + 0.3 = 0.5
    # kappa = (0.7 - 0.5) / (1 - 0.5) = 0.4
    expected = 0.4

    result = cohen_kappa(judge_labels, human_labels)

    assert abs(result - expected) < 1e-9


def test_cohen_kappa_perfect_agreement_is_one():
    labels = [0, 1, 2, 1, 0]
    assert cohen_kappa(labels, labels) == 1.0


def test_cross_family_assertion_passes_when_families_differ():
    data = copy.deepcopy(load_config().data)
    data["llm"]["model"] = "claude-sonnet-5"
    data["eval"]["judge"]["model"] = "gpt-4o-mini"
    cfg = Config(data)

    assert_cross_family(cfg)  # must not raise


def test_cross_family_assertion_raises_when_same_family():
    data = copy.deepcopy(load_config().data)
    data["llm"]["model"] = "claude-sonnet-5"
    data["eval"]["judge"]["model"] = "claude-haiku-4-5"
    cfg = Config(data)

    with pytest.raises(ValueError):
        assert_cross_family(cfg)
