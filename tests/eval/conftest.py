"""Shared fixtures for the eval-harness test suite: tiny hand-built delta/chat fixtures, a mocked
judge, synthetic trace files, and a baseline scorecard — no LLM, no network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from delta_chat.canonical.model import BBox, Change
from delta_chat.chat.types import ChatAnswer, Citation, Claim
from eval.gold import GoldChange, QAGold


def _change(id_, type_, *, page=0, sheet=None, bbox=(0.1, 0.1, 0.2, 0.15), description="", old=None, new=None,
            change_set_id="CS-0-1", hard_slice=False) -> Change:
    x0, y0, x1, y1 = bbox
    return Change(
        id=id_, type=type_, item_type="tag", page=page, sheet=sheet, bbox=BBox(x0, y0, x1, y1),
        description=description, old=old, new=new, rev_a_ref=None, rev_b_ref=None,
        match_conf=0.9, class_conf=0.9, change_set_id=change_set_id, hard_slice=hard_slice,
    )


def _gold_change(change_id, type_, *, page=0, sheet=None, bbox=(0.1, 0.1, 0.2, 0.15), description="",
                  old=None, new=None, pair_id="pair_01", format="pdf_native", hard_slice=False) -> GoldChange:
    x0, y0, x1, y1 = bbox
    return GoldChange(
        change_id=change_id, type=type_, page=page, sheet=sheet, bbox=BBox(x0, y0, x1, y1),
        old=old, new=new, description=description, hard_slice=hard_slice, pair_id=pair_id, format=format,
    )


@pytest.fixture
def delta_gold() -> list[GoldChange]:
    return [
        _gold_change("GOLD-1", "modified", bbox=(0.1, 0.1, 0.2, 0.15), old="A", new="B",
                      description="tag changed", pair_id="pair_01", format="pdf_native"),
        _gold_change("GOLD-2", "modified", bbox=(0.3, 0.1, 0.4, 0.15), old="C", new="D",
                      description="dimension changed", pair_id="pair_01", format="pdf_native"),
        _gold_change("GOLD-3", "removed", bbox=(0.5, 0.1, 0.6, 0.15), description="note removed",
                      pair_id="pair_02", format="pdf_scanned", hard_slice=True),
    ]


@pytest.fixture
def delta_pred() -> list[Change]:
    return [
        # exact match of GOLD-1
        _change("CHG-1", "modified", bbox=(0.1, 0.1, 0.2, 0.15), old="A", new="B", description="tag changed"),
        # same location+description as GOLD-2 but a mistyped type -> lands in the confusion matrix
        _change("CHG-2", "added", bbox=(0.3, 0.1, 0.4, 0.15), old="C", new="D", description="dimension changed"),
        # spurious, no matching gold at all -> FP
        _change("CHG-3", "moved", page=1, bbox=(0.7, 0.7, 0.8, 0.75), description="unrelated spurious change"),
        # GOLD-3 is never predicted -> FN
    ]


@pytest.fixture
def qa_gold() -> list[QAGold]:
    return [
        QAGold(question="What changed on FV-9040?", reference_answer="Setpoint changed to 175 psi.",
               expected_citation_ids=["delta:CHG-1"], answerable=True, citation_class="delta", pair_id="pair_01"),
        QAGold(question="What is the hallucinated-citation question?", reference_answer="X changed.",
               expected_citation_ids=["doc_a:0:notes:1"], answerable=True, citation_class="doc_span", pair_id="pair_01"),
        QAGold(question="What is the wrong-span question?", reference_answer="Y changed.",
               expected_citation_ids=["doc_a:0:notes:2"], answerable=True, citation_class="doc_span", pair_id="pair_01"),
        QAGold(question="What does the drawing say about warp drives?", reference_answer="",
               expected_citation_ids=[], answerable=False, citation_class="doc_span", pair_id="pair_01"),
    ]


@pytest.fixture
def chat_preds() -> list[ChatAnswer]:
    return [
        # 1. fully grounded: cited id is in used_chunks, judge will say it entails.
        ChatAnswer(
            state="Grounded", text="Setpoint changed to 175 psi.",
            claims=[Claim(text="Setpoint changed to 175 psi.",
                          citations=[Citation(citation_id="delta:CHG-1", source_type="delta", page=0, sheet=None, bbox=None)],
                          verified=True)],
            used_chunks=["delta:CHG-1"], refusal_reason=None,
        ),
        # 2. hallucinated citation id: NOT present in used_chunks -> validity < 1.
        ChatAnswer(
            state="Grounded", text="X changed.",
            claims=[Claim(text="X changed.",
                          citations=[Citation(citation_id="doc_a:0:notes:HALLUCINATED", source_type="doc_a", page=0, sheet=None, bbox=None)],
                          verified=True)],
            used_chunks=["doc_a:0:notes:1"], refusal_reason=None,
        ),
        # 3. wrong-span cite: id DOES resolve (in used_chunks) but judge will say it doesn't entail.
        ChatAnswer(
            state="Grounded", text="Y changed (wrong span).",
            claims=[Claim(text="Y changed (wrong span).",
                          citations=[Citation(citation_id="doc_a:0:notes:2", source_type="doc_a", page=0, sheet=None, bbox=None)],
                          verified=True)],
            used_chunks=["doc_a:0:notes:2"], refusal_reason=None,
        ),
        # 4. correctly abstains on the unanswerable question.
        ChatAnswer(
            state="Rejected", text="", claims=[], used_chunks=[],
            refusal_reason="INSUFFICIENT_EVIDENCE: nothing about warp drives was retrieved",
        ),
    ]


class _MockJudge:
    """Deterministic canned verdicts keyed by citation_id, so chat-metric tests never hit a
    network. `entails` returns False only for the deliberately-wrong-span citation."""

    def entails(self, claim_text: str, citation_id: str) -> bool:
        return citation_id != "doc_a:0:notes:2"

    def relevancy(self, answer_text: str, question: str) -> float:
        return 1.0


@pytest.fixture
def mock_judge() -> _MockJudge:
    return _MockJudge()


@pytest.fixture
def trace_files(tmp_path) -> list[Path]:
    blobs = [
        {
            "request_id": "r1", "trace_id": "t1", "started_at": "2026-01-01T00:00:00Z",
            "duration_ms": 100.0, "status": "ok", "input": {}, "answer": None,
            "stages": [
                {"name": "ingest", "duration_ms": 10.0, "status": "ok", "attrs": {}, "error": None},
                {"name": "llm.call", "duration_ms": 200.0, "status": "ok",
                 "attrs": {"gen_ai.usage.input_tokens": 100, "gen_ai.usage.output_tokens": 50, "usd": 0.01},
                 "error": None},
            ],
        },
        {
            "request_id": "r2", "trace_id": "t2", "started_at": "2026-01-01T00:01:00Z",
            "duration_ms": 300.0, "status": "ok", "input": {}, "answer": None,
            "stages": [
                {"name": "ingest", "duration_ms": 30.0, "status": "ok", "attrs": {}, "error": None},
                {"name": "llm.call", "duration_ms": 400.0, "status": "ok",
                 "attrs": {"gen_ai.usage.input_tokens": 300, "gen_ai.usage.output_tokens": 60, "usd": 0.02},
                 "error": None},
            ],
        },
    ]
    paths = []
    for i, blob in enumerate(blobs):
        p = tmp_path / f"trace-{i}.json"
        p.write_text(json.dumps(blob), encoding="utf-8")
        paths.append(p)
    return paths


@pytest.fixture
def baseline_scorecard() -> dict:
    return {
        "dataset_hash": "deadbeef",
        "split": {"train": ["pair_01"], "test": ["pair_02"]},
        "deterministic": {"delta_f1_test": 0.90, "citation_validity": 1.0},
    }
