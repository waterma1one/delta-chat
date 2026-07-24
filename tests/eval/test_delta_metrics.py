"""Tests for eval/delta_metrics.py — Hungarian-matched P/R/F1, multi-tau, type confusion, soft-F1,
segmentation. No LLM, no network."""

from __future__ import annotations

from delta_chat.canonical.model import BBox, Change

from eval.delta_metrics import score_delta
from eval.gold import GoldChange


def _change(id_, type_, *, page=0, bbox=(0.1, 0.1, 0.2, 0.15), description="", old=None, new=None) -> Change:
    x0, y0, x1, y1 = bbox
    return Change(
        id=id_, type=type_, item_type="tag", page=page, sheet=None, bbox=BBox(x0, y0, x1, y1),
        description=description, old=old, new=new, rev_a_ref=None, rev_b_ref=None,
        match_conf=0.9, class_conf=0.9, change_set_id="CS-0-1",
    )


def _gold(change_id, type_, *, page=0, bbox=(0.1, 0.1, 0.2, 0.15), description="", old=None, new=None,
          pair_id="p1", format="pdf_native", hard_slice=False) -> GoldChange:
    x0, y0, x1, y1 = bbox
    return GoldChange(
        change_id=change_id, type=type_, page=page, sheet=None, bbox=BBox(x0, y0, x1, y1),
        old=old, new=new, description=description, hard_slice=hard_slice, pair_id=pair_id, format=format,
    )


def test_perfect_prediction_gives_p_r_f1_of_one():
    gold = [_gold("G1", "modified", description="tag changed", old="A", new="B")]
    pred = [_change("C1", "modified", description="tag changed", old="A", new="B")]

    score = score_delta(pred, gold, taus=[0.5])

    assert score.per_tau[0.5]["precision"] == 1.0
    assert score.per_tau[0.5]["recall"] == 1.0
    assert score.per_tau[0.5]["f1"] == 1.0


def test_extra_predicted_change_lowers_precision():
    gold = [_gold("G1", "modified", description="tag changed", old="A", new="B")]
    pred = [
        _change("C1", "modified", description="tag changed", old="A", new="B"),
        _change("C2", "added", page=5, bbox=(0.9, 0.9, 0.95, 0.95), description="totally unrelated spurious"),
    ]

    score = score_delta(pred, gold, taus=[0.5])

    assert score.per_tau[0.5]["precision"] < 1.0
    assert score.per_tau[0.5]["recall"] == 1.0


def test_missed_gold_lowers_recall():
    gold = [
        _gold("G1", "modified", description="tag changed", old="A", new="B"),
        _gold("G2", "removed", page=5, bbox=(0.9, 0.9, 0.95, 0.95), description="missed change"),
    ]
    pred = [_change("C1", "modified", description="tag changed", old="A", new="B")]

    score = score_delta(pred, gold, taus=[0.5])

    assert score.per_tau[0.5]["recall"] < 1.0
    assert score.per_tau[0.5]["precision"] == 1.0


def test_hungarian_prevents_double_count_when_gold_split_into_two():
    gold = [_gold("G1", "modified", bbox=(0.1, 0.1, 0.3, 0.15), description="one change spanning a region")]
    pred = [
        _change("C1", "modified", bbox=(0.1, 0.1, 0.2, 0.15), description="one change spanning a region"),
        _change("C2", "modified", bbox=(0.2, 0.1, 0.3, 0.15), description="one change spanning a region"),
    ]

    score = score_delta(pred, gold, taus=[0.5])

    # exactly one of the two predictions can match the single gold — never two TPs from one gold.
    assert score.per_tau[0.5]["tp"] <= 1
    assert score.counts == {"n_pred": 2, "n_gold": 1}


def test_type_mismatch_on_localized_change_lands_in_confusion_not_silent_fp_fn():
    gold = [_gold("G1", "modified", description="dimension changed", old="C", new="D")]
    pred = [_change("C1", "added", description="dimension changed", old="C", new="D")]

    score = score_delta(pred, gold, taus=[0.5])

    assert score.type_confusion.get("modified", {}).get("added") == 1
    assert "MISSED" not in score.type_confusion.get("modified", {})
    assert "SPURIOUS" not in score.type_confusion


def test_missed_and_spurious_land_in_dedicated_confusion_cells():
    gold = [_gold("G1", "removed", description="a gold change nothing predicts")]
    pred = [_change("C1", "added", page=9, bbox=(0.9, 0.9, 0.95, 0.95), description="a totally spurious prediction")]

    score = score_delta(pred, gold, taus=[0.9])  # score between these two is far below any real match

    assert score.type_confusion.get("removed", {}).get("MISSED") == 1
    assert score.type_confusion.get("SPURIOUS", {}).get("added") == 1


def test_multi_tau_sweep_is_monotonic_non_increasing_tp():
    gold = [
        _gold("G1", "modified", bbox=(0.1, 0.1, 0.2, 0.15), description="tag changed", old="A", new="B"),
        _gold("G2", "modified", bbox=(0.3, 0.1, 0.4, 0.15), description="dimension changed", old="C", new="D"),
    ]
    pred = [
        _change("C1", "modified", bbox=(0.1, 0.1, 0.2, 0.15), description="tag changed", old="A", new="B"),
        _change("C2", "added", bbox=(0.3, 0.1, 0.4, 0.15), description="dimension changed", old="C", new="D"),
    ]
    taus = [0.5, 0.7, 0.85, 0.95]

    score = score_delta(pred, gold, taus=taus)

    tps = [score.per_tau[t]["tp"] for t in taus]
    assert all(tps[i] >= tps[i + 1] for i in range(len(tps) - 1))
    assert tps[0] == 2  # both match at the lenient end (a type mistype only costs 0.2 of the score)
    assert tps[-1] < 2  # the mistyped pair (s=0.8) drops out once tau exceeds 0.8


def test_soft_f1_is_at_least_hard_f1_on_partial_matches():
    gold = [_gold("G1", "modified", description="dimension changed", old="C", new="D")]
    pred = [_change("C1", "added", description="dimension changed", old="C", new="D")]  # mistyped -> s=0.8

    score = score_delta(pred, gold, taus=[0.95])  # strict: hard-F1 drops this pair entirely (0.8 < 0.95)

    assert score.soft_f1 >= score.per_tau[0.95]["f1"]


def test_empty_pred_and_gold_gives_perfect_score():
    score = score_delta([], [], taus=[0.5, 0.7])
    assert score.per_tau[0.5]["f1"] == 1.0
    assert score.per_tau[0.7]["f1"] == 1.0
    assert score.soft_f1 == 1.0


def test_metrics_segmented_by_pair_format_and_hard_slice(delta_pred, delta_gold):
    score = score_delta(delta_pred, delta_gold, taus=[0.5])

    assert "pair:pair_01" in score.by_segment
    assert "pair:pair_02" in score.by_segment
    assert "format:pdf_native" in score.by_segment
    assert "format:pdf_scanned" in score.by_segment
    assert "hard_slice:True" in score.by_segment
    assert "hard_slice:False" in score.by_segment

    # pair_02's only gold item is the hard-slice, entirely-missed change -> recall 0 there;
    # pair_01's two gold items are both matched (one mistyped, still a match) -> recall 1.
    assert score.by_segment["pair:pair_02"]["per_tau"][0.5]["recall"] == 0.0
    assert score.by_segment["pair:pair_01"]["per_tau"][0.5]["recall"] == 1.0
    assert score.by_segment["hard_slice:True"]["per_tau"][0.5]["recall"] == 0.0
