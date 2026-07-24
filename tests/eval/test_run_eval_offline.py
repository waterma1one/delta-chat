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

from eval.run_eval import _offline_llm_complete, _OfflineJudge, main, resolve_pair_files

REPO_ROOT = Path(__file__).resolve().parents[2]


# --- sample-pair file resolution ------------------------------------------------------------------
# Regression: hardcoded rev_a.pdf/rev_b.pdf silently skipped the scanned and DXF pairs, so 8 of the
# 12 held-out gold changes were scored as misses against zero predictions.


def test_resolves_plain_pdf_pair(tmp_path):
    (tmp_path / "rev_a.pdf").write_bytes(b"a")
    (tmp_path / "rev_b.pdf").write_bytes(b"b")
    a, b = resolve_pair_files(tmp_path)
    assert a.name == "rev_a.pdf" and b.name == "rev_b.pdf"


def test_resolves_dxf_pair(tmp_path):
    (tmp_path / "rev_a.dxf").write_bytes(b"a")
    (tmp_path / "rev_b.dxf").write_bytes(b"b")
    a, b = resolve_pair_files(tmp_path)
    assert a.name == "rev_a.dxf" and b.name == "rev_b.dxf"


def test_prefers_scanned_rev_b_over_native_variant(tmp_path):
    """pair_03_scanned ships both; the scanned one is the whole point of that pair, so evaluating
    the native variant would quietly skip the OCR path it exists to exercise."""
    (tmp_path / "rev_a.pdf").write_bytes(b"a")
    (tmp_path / "rev_b_native.pdf").write_bytes(b"n")
    (tmp_path / "rev_b_scanned.pdf").write_bytes(b"s")
    _a, b = resolve_pair_files(tmp_path)
    assert b.name == "rev_b_scanned.pdf"


def test_returns_none_for_missing_or_incomplete_pair(tmp_path):
    assert resolve_pair_files(tmp_path / "does_not_exist") is None
    (tmp_path / "rev_a.pdf").write_bytes(b"a")  # no rev_b at all
    assert resolve_pair_files(tmp_path) is None


def test_every_shipped_sample_pair_resolves():
    """Guards the actual repo layout: if a pair stops resolving, the eval silently scores it as a
    total miss rather than failing, so assert on the real directories."""
    samples = REPO_ROOT / "data" / "samples"
    pairs = sorted(p for p in samples.glob("pair_*") if p.is_dir())
    if not pairs:
        pytest.skip("samples not synthesized")
    for p in pairs:
        assert resolve_pair_files(p) is not None, f"{p.name} no longer resolves"


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
