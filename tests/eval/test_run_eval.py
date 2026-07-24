"""Tests for eval/run_eval.py — scorecard + budget writing, byte-stability, and the CI regression
gate. All predictions/judge are injected; no LLM, no network, no real ingest/delta pipeline run."""

from __future__ import annotations

import copy
import json

import pytest

from delta_chat.config import Config, load_config

from eval.run_eval import build_scorecard, check_regression, run_eval


@pytest.fixture
def eval_cfg():
    data = copy.deepcopy(load_config().data)
    data["eval"]["judge"]["n_samples"] = 2
    # match the fixture gold's pair_ids (pair_01/pair_02) rather than the real dataset's
    # pair_01_native/pair_02_hard/... so the split partition actually selects fixture data.
    data["eval"]["datasets"]["split"] = {"train": ["pair_01"], "test": ["pair_02"]}
    return Config(data)


def _dataset_paths():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    return [root / "eval" / "datasets" / "delta_gold.jsonl", root / "eval" / "datasets" / "qa_gold.jsonl"]


def test_run_eval_writes_scorecard_and_budget(
    eval_cfg, tmp_path, delta_pred, delta_gold, chat_preds, qa_gold, mock_judge, trace_files
):
    exit_code = run_eval(
        eval_cfg, out_dir=tmp_path, judge=mock_judge,
        delta_pred_by_pair={"pair_01": delta_pred, "pair_02": []}, delta_gold=delta_gold,
        chat_preds=chat_preds, chat_gold=qa_gold, retrieved={q.question: [] for q in qa_gold},
        trace_files=trace_files,
    )

    assert (tmp_path / "scorecard.json").exists()
    assert (tmp_path / "budget.md").exists()
    scorecard = json.loads((tmp_path / "scorecard.json").read_text())
    assert "deterministic" in scorecard
    assert "dataset_hash" in scorecard
    assert exit_code == 0  # no baseline supplied -> nothing to regress against


def test_scorecard_deterministic_sections_are_byte_stable_across_runs(
    eval_cfg, tmp_path, delta_pred, delta_gold, chat_preds, qa_gold, mock_judge
):
    kwargs = dict(
        cfg=eval_cfg,
        delta_pred_by_pair={"pair_01": delta_pred, "pair_02": []},
        delta_gold=delta_gold, chat_preds=chat_preds, chat_gold=qa_gold, judge=mock_judge,
        retrieved={q.question: [] for q in qa_gold}, dataset_paths=_dataset_paths(),
    )
    sc1 = build_scorecard(**kwargs)
    sc2 = build_scorecard(**kwargs)

    assert json.dumps(sc1["deterministic"], sort_keys=True) == json.dumps(sc2["deterministic"], sort_keys=True)
    assert json.dumps(sc1["delta"], sort_keys=True) == json.dumps(sc2["delta"], sort_keys=True)


def test_ci_gate_returns_nonzero_when_deterministic_metric_regresses(
    eval_cfg, tmp_path, delta_pred, delta_gold, chat_preds, qa_gold, mock_judge
):
    scorecard = build_scorecard(
        eval_cfg, delta_pred_by_pair={"pair_01": delta_pred, "pair_02": []}, delta_gold=delta_gold,
        chat_preds=chat_preds, chat_gold=qa_gold, judge=mock_judge,
        retrieved={q.question: [] for q in qa_gold}, dataset_paths=_dataset_paths(),
    )
    # baseline claims a much higher delta_f1_test than this run actually achieved
    baseline = {"deterministic": {"delta_f1_test": scorecard["deterministic"]["delta_f1_test"] + 0.5}}
    thresholds = {"delta_f1_test": 0.02}

    problems = check_regression(scorecard, baseline, thresholds)

    assert problems
    assert any("delta_f1_test" in p for p in problems)


def test_ci_gate_is_direction_aware_for_lower_is_better_metrics():
    # latency/cost regress on an INCREASE, unlike every other deterministic metric (higher-is-
    # better) — a naive "baseline - current > threshold" check would never fire for these.
    scorecard = {"deterministic": {"latency_p95_ms": 500.0, "cost_usd_per_request": 0.05}}
    baseline = {"deterministic": {"latency_p95_ms": 200.0, "cost_usd_per_request": 0.05}}
    thresholds = {"latency_p95_ms": 50.0, "cost_usd_per_request": 0.01}

    problems = check_regression(scorecard, baseline, thresholds)

    assert any("latency_p95_ms" in p for p in problems)
    assert not any("cost_usd_per_request" in p for p in problems)  # unchanged -> no regression


def test_ci_gate_lower_is_better_does_not_fire_when_metric_improves():
    scorecard = {"deterministic": {"latency_p95_ms": 100.0}}
    baseline = {"deterministic": {"latency_p95_ms": 500.0}}  # got FASTER -> not a regression
    thresholds = {"latency_p95_ms": 50.0}

    assert check_regression(scorecard, baseline, thresholds) == []


def test_ci_gate_does_not_fire_on_model_graded_metric_noise(
    eval_cfg, tmp_path, delta_pred, delta_gold, chat_preds, qa_gold, mock_judge
):
    scorecard = build_scorecard(
        eval_cfg, delta_pred_by_pair={"pair_01": delta_pred, "pair_02": []}, delta_gold=delta_gold,
        chat_preds=chat_preds, chat_gold=qa_gold, judge=mock_judge,
        retrieved={q.question: [] for q in qa_gold}, dataset_paths=_dataset_paths(),
    )
    # a wildly different "baseline" faithfulness (model-graded) must never affect the gate, since
    # check_regression only ever inspects the `deterministic` block.
    baseline = {
        "deterministic": dict(scorecard["deterministic"]),
        "chat": {"faithfulness": {"mean": 0.0, "stdev": 0.0}},
    }
    thresholds = {"delta_f1_test": 0.02, "citation_validity": 0.0}

    problems = check_regression(scorecard, baseline, thresholds)

    assert problems == []


def test_scorecard_records_dataset_hash_and_split_provenance(
    eval_cfg, delta_pred, delta_gold, chat_preds, qa_gold, mock_judge
):
    scorecard = build_scorecard(
        eval_cfg, delta_pred_by_pair={"pair_01": delta_pred, "pair_02": []}, delta_gold=delta_gold,
        chat_preds=chat_preds, chat_gold=qa_gold, judge=mock_judge,
        retrieved={q.question: [] for q in qa_gold}, dataset_paths=_dataset_paths(),
    )

    assert isinstance(scorecard["dataset_hash"], str) and len(scorecard["dataset_hash"]) == 64
    assert scorecard["split"]["train"] and scorecard["split"]["test"]
    # every deterministic number is drawn from the TEST split's delta score
    assert scorecard["deterministic"]["delta_f1_test"] == scorecard["delta"]["test"]["per_tau"][str(max(eval_cfg.get("eval.taus")))]["f1"]
