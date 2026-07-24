"""Tests for `--offline` mode (section-11): `make eval` / the CI deterministic gate must never
need a live model or an API key. `_offline_llm_complete` is a deterministic stand-in for
`delta_chat.chat.llm.complete`, and `_OfflineJudge` a deterministic stand-in for the live judge —
both wired in by `eval.run_eval.main(["--offline", ...])`, mirroring the same mocked-LLM-fixture
convention used elsewhere in this codebase (tests/chat/conftest.py's `_MockLlm`).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from eval.run_eval import _offline_llm_complete, _OfflineJudge, main

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_offline_llm_complete_answers_yes_for_sufficiency_probe():
    messages = [
        {"role": "system", "content": "Answer only 'yes' or 'no': does the context suffice?"},
        {"role": "user", "content": "Question: what changed?\n\nContext:\n[c1] the setpoint changed"},
    ]
    result = _offline_llm_complete(messages, model="offline")
    assert result.text.strip().lower() == "yes"


def test_offline_llm_complete_builds_resolvable_claims_for_json_grounding():
    user_content = (
        "Question: what changed?\n\nSources:\n"
        "[delta:CHG-0001] setpoint changed to 175 psi\n"
        "[doc_a:0:notes:2] unrelated note text"
    )
    messages = [{"role": "system", "content": "..."}, {"role": "user", "content": user_content}]
    result = _offline_llm_complete(messages, model="offline", response_format={"type": "json_object"})

    payload = json.loads(result.text)
    assert payload["claims"], "offline stub must produce at least one claim when sources exist"
    cited_ids = {cid for c in payload["claims"] for cid in c["citations"]}
    assert cited_ids <= {"delta:CHG-0001", "doc_a:0:notes:2"}
    # every claim's text overlaps its cited chunk's text, so chat/answer.py's lexical entailment
    # check (rapidfuzz partial_ratio) passes deterministically without a real model call.
    for c in payload["claims"]:
        assert c["text"]


def test_offline_llm_complete_reports_insufficient_when_no_sources():
    messages = [{"role": "system", "content": "..."}, {"role": "user", "content": "Question: x\n\nSources:\n"}]
    result = _offline_llm_complete(messages, model="offline", response_format={"type": "json_object"})
    payload = json.loads(result.text)
    assert payload["insufficient_evidence"] is True
    assert payload["claims"] == []


def test_offline_judge_never_touches_the_network():
    judge = _OfflineJudge()
    assert judge.entails("any claim", "any-citation-id") is True
    assert judge.relevancy("any answer", "any question") == 1.0


def test_main_offline_runs_end_to_end_with_no_api_keys_set(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    pair01 = REPO_ROOT / "data" / "samples" / "pair_01_native"
    if not (pair01 / "rev_a.pdf").exists():
        pytest.skip(f"sample pair missing: {pair01}")

    exit_code = main(["--offline", "--out-dir", str(tmp_path)])

    assert exit_code == 0
    scorecard = json.loads((tmp_path / "scorecard.json").read_text())
    assert "deterministic" in scorecard
    assert (tmp_path / "budget.md").exists()
