"""Tests for delta/engine.py — orchestration + determinism, against the synthesized native pair."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from unittest.mock import Mock

from delta_chat.delta.engine import compute_delta
from delta_chat.delta.report import render_report
from delta_chat.ingest.pdf_native import parse_native_pdf


def _mock_llm() -> Mock:
    # pair_01_native's note edits don't land in the ambiguous band (verified: equal/delete/insert
    # opcodes only, no replace), so this should never actually be called -- but the section plan
    # requires the LLM path be mocked for these tests regardless, since it's the pipeline's one
    # intentionally non-deterministic seam and must not silently go unexercised/unverified.
    return Mock(return_value={"is_meaningful": True, "summary": "reworded", "severity": "info"})


def _load_pair(repo_root: Path):
    pair_dir = repo_root / "data" / "samples" / "pair_01_native"
    a = parse_native_pdf((pair_dir / "rev_a.pdf").read_bytes(), "pid:pair_01_native:A", source_rev="A")
    b = parse_native_pdf((pair_dir / "rev_b.pdf").read_bytes(), "pid:pair_01_native:B", source_rev="B")
    return a, b


def _load_gold(repo_root: Path) -> list[dict]:
    gold_path = repo_root / "eval" / "datasets" / "delta_gold.jsonl"
    rows = [json.loads(line) for line in gold_path.read_text().splitlines() if line.strip()]
    return [r for r in rows if r["pair_id"] == "pair_01_native"]  # train split only


def test_compute_delta_reproduces_known_edit_gold(repo_root, cfg):
    doc_a, doc_b = _load_pair(repo_root)
    delta = compute_delta(doc_a, doc_b, cfg, llm=_mock_llm())

    # Compared on the substantive fields (type/item_type/old/new), as a multiset so a wrong *count*
    # of otherwise-identical changes would still be caught: descriptions are generated human-readable
    # text, not extracted content, so they aren't part of the gold contract.
    got = Counter((ch.type, ch.item_type, ch.old, ch.new) for ch in delta.changes)
    expected = Counter((g["type"], g["item_type"], g["old"], g["new"]) for g in _load_gold(repo_root))

    assert got == expected


def test_compute_delta_is_deterministic(repo_root, cfg):
    doc_a, doc_b = _load_pair(repo_root)

    delta1 = compute_delta(doc_a, doc_b, cfg, llm=_mock_llm())
    delta2 = compute_delta(doc_a, doc_b, cfg, llm=_mock_llm())

    _, json1 = render_report(delta1)
    _, json2 = render_report(delta2)

    assert json.dumps(json1, sort_keys=True) == json.dumps(json2, sort_keys=True)


# --- instrumentation (regression: the delta counter and stage attrs were never populated) --------


def test_compute_delta_increments_the_prometheus_delta_counter(cfg, item_factory):
    """The counter was defined and unit-tested but no production path ever called it, so
    `delta_chat_delta_changes_total` stayed at 0.0 no matter what a real run produced."""
    from delta_chat.canonical.model import Document
    from delta_chat.observability import metrics

    metrics.reset()
    doc_a = Document(pid="A", format="pdf_native", page_count=1, revision_label="A",
                     sheet_size=(1.0, 1.0), items=[item_factory("a1", "26-FV-9038")])
    doc_b = Document(pid="B", format="pdf_native", page_count=1, revision_label="B",
                     sheet_size=(1.0, 1.0), items=[item_factory("b1", "26-FV-9040")])

    delta = compute_delta(doc_a, doc_b, cfg)

    assert delta.changes  # the fixture genuinely differs, so the assertion below is meaningful
    body, _ = metrics.render_prometheus()
    assert f"delta_chat_delta_changes_total {float(len(delta.changes))}" in body


def test_delta_stage_records_change_counts_in_the_trace(cfg, item_factory, tmp_path):
    """Traces carried timing but no detail — every stage's `attrs` was an empty dict, so a trace
    told you a run took N ms but not what it produced."""
    from delta_chat.canonical.model import Document
    from delta_chat.config import Config
    from delta_chat.observability import tracing

    trace_cfg = Config({**cfg.data, "obs": {**cfg.data["obs"], "trace_dir": str(tmp_path),
                                            "otlp_enabled": False}})
    doc_a = Document(pid="A", format="pdf_native", page_count=1, revision_label="A",
                     sheet_size=(1.0, 1.0), items=[item_factory("a1", "26-FV-9038")])
    doc_b = Document(pid="B", format="pdf_native", page_count=1, revision_label="B",
                     sheet_size=(1.0, 1.0), items=[item_factory("b1", "26-FV-9040")])

    tr = tracing.start_request({"command": "test"}, cfg=trace_cfg)
    with tracing.stage("delta"):
        compute_delta(doc_a, doc_b, trace_cfg)
    tr.finish()

    trace = json.loads(next(Path(tmp_path).glob("*.json")).read_text())
    delta_stage = next(s for s in trace["stages"] if s["name"] == "delta")
    assert delta_stage["attrs"]["change_count"] > 0
    assert delta_stage["attrs"]["counts_by_type"]
