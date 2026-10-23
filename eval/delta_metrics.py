"""Delta-quality metrics — Hungarian-matched P/R/F1 at multiple τ, soft-F1, a type-
confusion matrix, and metrics segmented by pair/format/hard_slice. This is the deterministic,
CI-gated core metric: the held-out test-split delta P/R/F1 must be pure code and byte-stable.

Matching: pred<->gold Hungarian assignment maximizing pairwise score
``s = 0.5*loc + 0.2*type + 0.3*desc_sim`` (loc = bbox IoU when same page/sheet else 0; type = exact
match; desc_sim = RapidFuzz WRatio over description+old+new). A matched pair with s>=tau is a TP;
an unmatched (or below-tau) prediction is an FP; an unmatched gold is an FN.

Segmentation caveat: ``Change`` (the prediction type) carries no pair_id/format — only gold does.
So ``by_segment`` groups matched pairs and missed-gold FNs by the GOLD side's pair_id/format/
hard_slice; a spurious (unmatched) prediction has no gold to attribute a segment from and is
counted only in the top-level, unsegmented totals. Each segment then re-runs its own Hungarian
assignment on just that segment's subset rather than reusing the global pairing — with near-tied
candidates that happen to straddle a segment boundary, local re-optimization could in principle
pick a different pairing than the global match did. Accepted as a known limitation (the tests don't
construct that ambiguous a fixture) rather than restructuring `_score_flat` to accept a
pre-computed assignment for a rare edge case.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from delta_chat.canonical.model import Change
from delta_chat.delta.align import normalize_text, spatial_similarity

from .gold import GoldChange

_LOC_WEIGHT = 0.5
_TYPE_WEIGHT = 0.2
_DESC_WEIGHT = 0.3


def _desc_sim(pred: Change, gold: GoldChange) -> float:
    a = normalize_text(f"{pred.description} {pred.old or ''} {pred.new or ''}")
    b = normalize_text(f"{gold.description} {gold.old or ''} {gold.new or ''}")
    return fuzz.WRatio(a, b) / 100.0


def _pair_score(pred: Change, gold: GoldChange) -> float:
    same_location = pred.page == gold.page and pred.sheet == gold.sheet
    loc = spatial_similarity(pred.bbox, gold.bbox, metric="iou") if same_location else 0.0
    type_match = 1.0 if pred.type == gold.type else 0.0
    return _LOC_WEIGHT * loc + _TYPE_WEIGHT * type_match + _DESC_WEIGHT * _desc_sim(pred, gold)


def _hungarian_matches(pred: list[Change], gold: list[GoldChange]) -> list[tuple[int, int, float]]:
    """Return (pred_idx, gold_idx, score) for every real<->real pairing in the optimal assignment
    (dummy-padded so pred/gold counts may differ; dummy pairings are dropped)."""
    n_pred, n_gold = len(pred), len(gold)
    if not n_pred or not n_gold:
        return []
    size = max(n_pred, n_gold)
    cost = np.ones((size, size))  # dummy rows/cols cost 1.0 (score 0)
    for i, p in enumerate(pred):
        for j, g in enumerate(gold):
            cost[i, j] = 1.0 - _pair_score(p, g)
    row_ind, col_ind = linear_sum_assignment(cost)
    return [(r, c, 1.0 - cost[r, c]) for r, c in zip(row_ind, col_ind) if r < n_pred and c < n_gold]


def _prf1(tp: float, fp: float, fn: float) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def _per_tau(matches: list[tuple[int, int, float]], n_pred: int, n_gold: int, taus: list[float]) -> dict[float, dict]:
    per_tau: dict[float, dict] = {}
    for tau in taus:
        matched_pred = {r for r, _c, s in matches if s >= tau}
        matched_gold = {c for _r, c, s in matches if s >= tau}
        tp, fp, fn = len(matched_pred), n_pred - len(matched_pred), n_gold - len(matched_gold)
        per_tau[tau] = {**_prf1(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn}
    return per_tau


def _soft_f1(matches: list[tuple[int, int, float]], n_pred: int, n_gold: int) -> float:
    """Fractional TP credit = matched score `s`, unconditional on any tau — a partial match a
    strict-enough tau would drop entirely (0 credit) still contributes `s` here.

    Guarantee (only holds against a tau the pair does NOT clear): for a matched pair with score s,
    soft-F1 >= hard-F1 at any tau > s, since hard drops that pair to 0 credit while soft still
    gives it `s`. This can NOT hold universally across every tau: at a tau <= s, hard scoring gives
    the same pair a FULL 1.0 TP credit, which is strictly more generous than soft's fractional `s`
    — no fractional-credit scheme can beat a hard threshold lenient enough to already max out. So
    "soft-F1 >= hard-F1" is meaningful when comparing against the strictest configured tau (where
    partial matches are most likely to be dropped by hard scoring), not literally every tau in a
    sweep."""
    soft_tp = sum(s for _r, _c, s in matches)
    soft_fp = n_pred - len(matches)
    soft_fn = n_gold - len(matches)
    return _prf1(soft_tp, soft_fp, soft_fn)["f1"]


def _type_confusion(
    matches: list[tuple[int, int, float]], pred: list[Change], gold: list[GoldChange], ref_tau: float
) -> dict[str, dict[str, int]]:
    """Built once at the most lenient configured tau (not swept): a matched-but-mistyped pair is
    recorded as a `gold_type -> pred_type` cell, never as a silent FP+FN. A wholly missed gold
    lands in `gold_type -> "MISSED"`; a spurious prediction lands in `"SPURIOUS" -> pred_type`."""
    confusion: dict[str, dict[str, int]] = {}

    def _bump(row: str, col: str) -> None:
        confusion.setdefault(row, {}).setdefault(col, 0)
        confusion[row][col] += 1

    matched_pred = {r for r, _c, s in matches if s >= ref_tau}
    matched_gold = {c for _r, c, s in matches if s >= ref_tau}
    for r, c, s in matches:
        if s >= ref_tau:
            _bump(gold[c].type, pred[r].type)
    for j, g in enumerate(gold):
        if j not in matched_gold:
            _bump(g.type, "MISSED")
    for i, p in enumerate(pred):
        if i not in matched_pred:
            _bump("SPURIOUS", p.type)
    return confusion


def _score_flat(pred: list[Change], gold: list[GoldChange], taus: list[float]) -> dict:
    """Score one (pred, gold) list pair with no segmentation — the shared core used both for the
    overall totals and for each segment's own sub-lists."""
    if not pred and not gold:
        empty_tau = {tau: {**_prf1(0, 0, 0), "tp": 0, "fp": 0, "fn": 0} for tau in taus}
        return {"per_tau": empty_tau, "soft_f1": 1.0, "type_confusion": {}, "counts": {"n_pred": 0, "n_gold": 0}}
    matches = _hungarian_matches(pred, gold)
    ref_tau = min(taus) if taus else 0.5
    return {
        "per_tau": _per_tau(matches, len(pred), len(gold), taus),
        "soft_f1": _soft_f1(matches, len(pred), len(gold)),
        "type_confusion": _type_confusion(matches, pred, gold, ref_tau),
        "counts": {"n_pred": len(pred), "n_gold": len(gold)},
    }


def _segment(gold: list[GoldChange], key_fn) -> dict[str, list[GoldChange]]:
    by_key: dict[str, list[GoldChange]] = {}
    for g in gold:
        by_key.setdefault(str(key_fn(g)), []).append(g)
    return by_key


@dataclass
class DeltaScore:
    per_tau: dict[float, dict]
    soft_f1: float
    type_confusion: dict
    by_segment: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)


def score_delta(pred: list[Change], gold: list[GoldChange], taus: list[float]) -> DeltaScore:
    """Hungarian-match pred<->gold on s=0.5*loc+0.2*type+0.3*desc_sim; report P/R/F1 at each tau, a
    soft-F1 (fractional TP credit = matched s), and a type-confusion matrix. Also compute the same
    metrics segmented by pair_id, by format, and by hard_slice (never one averaged headline F1)."""
    overall = _score_flat(pred, gold, taus)
    matches = _hungarian_matches(pred, gold)
    ref_tau = min(taus) if taus else 0.5
    # Only pairs clearing the lenient reference tau count as a "real" match for segment
    # attribution — the raw (unconditional) Hungarian assignment always produces a full
    # permutation when len(pred) == len(gold), which would otherwise glue an unrelated spurious
    # prediction to whatever gold happened to be left over, misattributing it to that gold's
    # pair/format/hard_slice segment instead of being excluded as documented above.
    matched_gold_idx = {c: r for r, c, s in matches if s >= ref_tau}

    def _matched_pred_for(gold_subset_idx: set[int]) -> list[Change]:
        # only pred items Hungarian-matched to a gold in this segment are attributable to it —
        # a spurious (unmatched) prediction has no gold, so no segment to attribute it to.
        return [pred[matched_gold_idx[j]] for j in gold_subset_idx if j in matched_gold_idx]

    gold_index_by: dict[str, dict[str, list[int]]] = {"pair": {}, "format": {}, "hard_slice": {}}
    for idx, g in enumerate(gold):
        gold_index_by["pair"].setdefault(g.pair_id, []).append(idx)
        gold_index_by["format"].setdefault(g.format, []).append(idx)
        gold_index_by["hard_slice"].setdefault(str(g.hard_slice), []).append(idx)

    by_segment: dict[str, dict] = {}
    for dimension, buckets in gold_index_by.items():
        for value, idxs in buckets.items():
            gold_subset = [gold[i] for i in idxs]
            pred_subset = _matched_pred_for(set(idxs))
            by_segment[f"{dimension}:{value}"] = _score_flat(pred_subset, gold_subset, taus)

    return DeltaScore(
        per_tau=overall["per_tau"], soft_f1=overall["soft_f1"], type_confusion=overall["type_confusion"],
        by_segment=by_segment, counts=overall["counts"],
    )
