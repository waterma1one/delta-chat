"""Alignment: two-stage Hungarian matching between two revisions' items.

Stage 1 runs a Hungarian assignment **per region** (drawing / title / revtable / ...). Stage 2 runs one
more Hungarian pass over the **combined** leftover unmatched-A + unmatched-B pool, ignoring region
scoping, to recover boundary / region-misassigned items before declaring add/remove (region still
contributes as a soft similarity term, so it nudges correct pairs together without hard-blocking a
cross-region move).

Text similarity dominates the blend on purpose: a tag that moved across the page but reads identically
must still be matched, not split into a remove+add pair. Spatial similarity defaults to normalized
centroid distance rather than IoU — real P&ID item boxes are small enough that IoU collapses to ~0 for
any genuine move and can't disambiguate two far-apart duplicate labels.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from math import sqrt

import numpy as np
from rapidfuzz import fuzz
from scipy.optimize import linear_sum_assignment

from ..canonical.model import BBox, Item

_QUOTE_DASH_MAP = str.maketrans(
    {
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        "–": "-",
        "—": "-",
        "−": "-",
    }
)
# "150 mm" -> "150mm" so a stray space around a unit never reads as a real edit.
_UNIT_SPACE_RE = re.compile(r"(?<=\d)\s+(?=(?:mm|cm|m|in|deg)\b)", re.IGNORECASE)
_WS_RE = re.compile(r"\s+")

# Normalized [0,1] page space -> the diagonal of the unit square, used to express a centroid shift
# as a scale-invariant fraction of the page regardless of physical sheet size.
PAGE_DIAGONAL = sqrt(2.0)


def normalize_text(text: str) -> str:
    """Fold whitespace/quote/dash/unit-spacing variants together before any text comparison."""
    t = text.translate(_QUOTE_DASH_MAP)
    t = _UNIT_SPACE_RE.sub("", t)
    t = _WS_RE.sub(" ", t).strip()
    return t.lower()


def centroid_distance(a: BBox, b: BBox) -> float:
    ax, ay = a.centroid
    bx, by = b.centroid
    return sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def centroid_shift_frac(a: BBox, b: BBox) -> float:
    """Centroid distance as a fraction of the page diagonal, clamped to [0,1]."""
    return min(centroid_distance(a, b) / PAGE_DIAGONAL, 1.0)


def _iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def spatial_similarity(a: BBox, b: BBox, metric: str = "centroid") -> float:
    if metric == "iou":
        return _iou(a, b)
    return 1.0 - centroid_shift_frac(a, b)


@dataclass(frozen=True)
class SimWeights:
    text: float = 0.55
    spatial: float = 0.25
    type: float = 0.10
    region: float = 0.10

    @classmethod
    def from_config(cls, cfg) -> "SimWeights":
        w = cfg.get("delta.weights")
        return cls(
            text=float(w["text"]),
            spatial=float(w["spatial"]),
            type=float(w["type"]),
            region=float(w["region"]),
        )

    @property
    def total(self) -> float:
        return self.text + self.spatial + self.type + self.region


def similarity(a: Item, b: Item, w: SimWeights, spatial_metric: str = "centroid") -> float:
    """Weighted blend, text dominant: 0.55*text + 0.25*spatial + 0.10*type + 0.10*region (weights
    sum-normalized). Returns a value in [0,1]."""
    text_sim = fuzz.WRatio(normalize_text(a.text), normalize_text(b.text)) / 100.0
    sp = spatial_similarity(a.bbox, b.bbox, spatial_metric)
    type_match = 1.0 if a.type == b.type else 0.0
    region_agree = 1.0 if a.region == b.region else 0.0
    total = w.total or 1.0
    return (w.text * text_sim + w.spatial * sp + w.type * type_match + w.region * region_agree) / total


@dataclass
class MatchResult:
    matched: list[tuple[Item, Item, float]] = field(default_factory=list)
    unmatched_a: list[Item] = field(default_factory=list)
    unmatched_b: list[Item] = field(default_factory=list)
    # (item.source_rev, item.id) -> runner-up margin (best-rejected-candidate similarity minus tau).
    # Item.id is only stable *within* a Document (ingestion assigns it positionally, e.g.
    # "0:drawing:3"), so rev-A and rev-B items routinely share an id string -- keying on id alone
    # would let one side's margin silently clobber the other's.
    margins: dict[tuple[str, str], float] = field(default_factory=dict)


def _solve(
    items_a: list[Item], items_b: list[Item], w: SimWeights, tau: float, spatial_metric: str
) -> tuple[list[tuple[Item, Item, float]], list[Item], list[Item]]:
    """One Hungarian solve, square-padded with a dummy cost of ``1 - tau`` so unequal A/B counts
    resolve to the dummy instead of crashing. Any produced match below ``tau`` is pushed back to the
    unmatched pools (post-filter)."""
    n, m = len(items_a), len(items_b)
    size = max(n, m)
    dummy_cost = 1.0 - tau
    cost = np.full((size, size), dummy_cost)
    sims = np.zeros((n, m))
    for i, a in enumerate(items_a):
        for j, b in enumerate(items_b):
            s = similarity(a, b, w, spatial_metric)
            sims[i, j] = s
            cost[i, j] = 1.0 - s
    row_ind, col_ind = linear_sum_assignment(cost)

    matched_a_idx: set[int] = set()
    matched_b_idx: set[int] = set()
    matched: list[tuple[Item, Item, float]] = []
    for r, c in zip(row_ind, col_ind):
        if r < n and c < m and sims[r, c] >= tau:
            matched.append((items_a[r], items_b[c], float(sims[r, c])))
            matched_a_idx.add(r)
            matched_b_idx.add(c)

    unmatched_a = [items_a[i] for i in range(n) if i not in matched_a_idx]
    unmatched_b = [items_b[j] for j in range(m) if j not in matched_b_idx]
    return matched, unmatched_a, unmatched_b


def _group_by_region(items: list[Item]) -> dict[str, list[Item]]:
    out: dict[str, list[Item]] = {}
    for it in items:
        out.setdefault(it.region, []).append(it)
    return out


def _runner_up_margins(
    unmatched: list[Item], opposite_all: list[Item], w: SimWeights, tau: float, spatial_metric: str
) -> dict[tuple[str, str], float]:
    margins: dict[tuple[str, str], float] = {}
    for it in unmatched:
        best = max((similarity(it, other, w, spatial_metric) for other in opposite_all), default=0.0)
        margins[(it.source_rev, it.id)] = best - tau
    return margins


def align(
    items_a: list[Item],
    items_b: list[Item],
    tau: float,
    *,
    weights: SimWeights | None = None,
    spatial_metric: str = "centroid",
) -> MatchResult:
    """Two-stage alignment (per-region Hungarian + global reconciliation, see module docstring).

    Returns a ``MatchResult`` whose matched pairs each carry their assignment similarity and whose
    unmatched items each carry a runner-up margin (best-rejected-candidate similarity minus ``tau``)
    for downstream add/remove confidence and missed-modify flagging.
    """
    w = weights or SimWeights()
    by_region_a = _group_by_region(items_a)
    by_region_b = _group_by_region(items_b)
    regions = sorted(set(by_region_a) | set(by_region_b))

    matched: list[tuple[Item, Item, float]] = []
    unmatched_a: list[Item] = []
    unmatched_b: list[Item] = []
    for region in regions:
        m, ua, ub = _solve(by_region_a.get(region, []), by_region_b.get(region, []), w, tau, spatial_metric)
        matched.extend(m)
        unmatched_a.extend(ua)
        unmatched_b.extend(ub)

    # Global cross-region reconciliation over the combined leftovers.
    recon_matched, final_unmatched_a, final_unmatched_b = _solve(
        unmatched_a, unmatched_b, w, tau, spatial_metric
    )
    matched.extend(recon_matched)

    margins = {
        **_runner_up_margins(final_unmatched_a, items_b, w, tau, spatial_metric),
        **_runner_up_margins(final_unmatched_b, items_a, w, tau, spatial_metric),
    }

    return MatchResult(matched=matched, unmatched_a=final_unmatched_a, unmatched_b=final_unmatched_b, margins=margins)
