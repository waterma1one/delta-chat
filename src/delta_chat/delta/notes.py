"""Ordered-list note diff (LCS/Myers) + fixed-cutoff embedding-cosine semantic gate.

Note-region items arrive from ingestion as per-word tokens (see ``canonical.model.Item``), so this
module first re-assembles them into logical note lines by row-clustering, then diffs the ordered line
sequence with ``difflib.SequenceMatcher`` (LCS/Myers-family algorithm) at line granularity.

This is the **only** place an LLM touches the delta pipeline, and even then it never authors old/new
values — those always come from the deterministic parse. A matched-but-reworded note is decided by an
embedding-cosine gate: clearly below ``semantic_low`` is meaningful, clearly at/above
``semantic_high`` is cosmetic (both no LLM call), and only the ambiguous band in between escalates to
exactly one LLM call. The cutoffs come from config, never hardcoded here. No calibration step exists
yet; ``config/default.yaml`` sets both to 0.90, so the LLM band is empty, and the default embedder is
a hashed character-trigram vector.
"""

from __future__ import annotations

import difflib
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Callable, Sequence

from ..canonical.model import BBox, Change, Item
from .align import normalize_text

# A list marker ("1.", "23.") is a leading integer + period followed by whitespace, and the period
# must NOT be immediately followed by another digit -- otherwise a leading decimal value like
# `1.5" MINIMUM CLEARANCE` would be mis-split into marker="1", content='5" MINIMUM CLEARANCE'.
_MARKER_RE = re.compile(r"^(\d+)\.(?!\d)\s+")
_EMBED_DIM = 64


@dataclass
class _NoteLine:
    marker: str | None
    content: str  # normalized, marker stripped -- what LCS/Myers and the cosine gate compare
    text: str  # raw joined line text -- what lands in Change.old/new
    item: Item  # representative item (first word) for id/page/sheet provenance
    bbox: BBox


def _cluster_rows(items: list[Item], row_tol: float) -> list[list[Item]]:
    """Group notes-region word items into visual lines by y0 proximity (within ``row_tol``, a fraction
    of page height), then sort each row by x0."""
    rows: list[list[Item]] = []
    for it in sorted(items, key=lambda i: (i.bbox.y0, i.bbox.x0)):
        for row in rows:
            if abs(row[0].bbox.y0 - it.bbox.y0) <= row_tol:
                row.append(it)
                break
        else:
            rows.append([it])
    for row in rows:
        row.sort(key=lambda i: i.bbox.x0)
    rows.sort(key=lambda row: row[0].bbox.y0)
    return rows


def _union_bbox(boxes: list[BBox]) -> BBox:
    return BBox(
        min(b.x0 for b in boxes), min(b.y0 for b in boxes), max(b.x1 for b in boxes), max(b.y1 for b in boxes)
    )


def _build_lines(items: list[Item], row_tol: float) -> list[_NoteLine]:
    lines: list[_NoteLine] = []
    for row in _cluster_rows(items, row_tol):
        text = " ".join(w.text for w in row)
        m = _MARKER_RE.match(text)
        marker = m.group(1) if m else None
        content = text[m.end():].strip() if m else text
        lines.append(
            _NoteLine(
                marker=marker,
                content=normalize_text(content),
                text=text,
                item=row[0],
                bbox=_union_bbox([w.bbox for w in row]),
            )
        )
    return lines


def _cosine(u: Sequence[float], v: Sequence[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v))
    nu = math.sqrt(sum(a * a for a in u))
    nv = math.sqrt(sum(b * b for b in v))
    if nu == 0 or nv == 0:
        return 0.0
    return dot / (nu * nv)


def _default_embedder(text: str) -> list[float]:
    """Deterministic, dependency-free fallback: a hashed character-trigram bag-of-words vector.

    Uses ``hashlib`` (not the builtin ``hash()``) so the bucket assignment is stable across processes
    and machines, not just within one interpreter's randomized hash seed — required for the pipeline's
    reproducibility guarantee. A real deployment should inject a hosted embedding model via the
    ``embedder`` parameter; this keeps the gate functional offline.
    """
    vec = [0.0] * _EMBED_DIM
    t = f"  {text.lower()}  "
    for i in range(len(t) - 2):
        trigram = t[i : i + 3]
        bucket = int(hashlib.md5(trigram.encode("utf-8")).hexdigest(), 16) % _EMBED_DIM
        vec[bucket] += 1.0
    return vec


def _make_renumbered(la: _NoteLine, lb: _NoteLine) -> Change:
    return Change(
        id="",
        type="renumbered",
        item_type="note",
        page=lb.item.page,
        sheet=lb.item.sheet,
        bbox=lb.bbox,
        description=f"Note {la.marker} renumbered to {lb.marker}",
        old=la.text,
        new=lb.text,
        rev_a_ref=la.item.id,
        rev_b_ref=lb.item.id,
        match_conf=1.0,
        class_conf=1.0,
        change_set_id="",
    )


def _make_removed(la: _NoteLine) -> Change:
    return Change(
        id="",
        type="removed",
        item_type="note",
        page=la.item.page,
        sheet=la.item.sheet,
        bbox=la.bbox,
        description=f"Note {la.marker or '?'} deleted",
        old=la.text,
        new=None,
        rev_a_ref=la.item.id,
        rev_b_ref=None,
        match_conf=1.0,
        class_conf=1.0,
        change_set_id="",
    )


def _make_added(lb: _NoteLine) -> Change:
    return Change(
        id="",
        type="added",
        item_type="note",
        page=lb.item.page,
        sheet=lb.item.sheet,
        bbox=lb.bbox,
        description=f"Note {lb.marker or '?'} added",
        old=None,
        new=lb.text,
        rev_a_ref=None,
        rev_b_ref=lb.item.id,
        match_conf=1.0,
        class_conf=1.0,
        change_set_id="",
    )


def _gate_reword(
    la: _NoteLine,
    lb: _NoteLine,
    low: float,
    high: float,
    llm: Callable[[str, str], dict] | None,
    embed: Callable[[str], Sequence[float]],
) -> Change | None:
    cos = _cosine(embed(la.content), embed(lb.content))

    if cos >= high:
        return None  # cosmetic reword -- not a reportable delta, no LLM call

    if cos < low:
        summary, severity, conf = None, None, 1.0
    else:
        if llm is None:
            raise ValueError("ambiguous note-gate band requires an 'llm' callable")
        result = llm(la.text, lb.text)
        if not result.get("is_meaningful", True):
            return None  # LLM judged the ambiguous-band reword cosmetic after all
        summary = result.get("summary")
        severity = result.get("severity")
        conf = float(result.get("confidence", 0.6))

    desc = summary or f"Note {lb.marker or '?'} reworded"
    if severity:
        desc = f"{desc} [{severity}]"
    return Change(
        id="",
        type="modified",
        item_type="note",
        page=lb.item.page,
        sheet=lb.item.sheet,
        bbox=lb.bbox,
        description=desc,
        old=la.text,
        new=lb.text,
        rev_a_ref=la.item.id,
        rev_b_ref=lb.item.id,
        match_conf=1.0,
        class_conf=conf,
        change_set_id="",
    )


def diff_notes(
    notes_a: list[Item],
    notes_b: list[Item],
    cfg,
    llm: Callable[[str, str], dict] | None = None,
    embedder: Callable[[str], Sequence[float]] | None = None,
) -> list[Change]:
    """LCS/Myers diff over ordered note lines -> NOTE_ADDED / NOTE_DELETED / RENUMBERED / reworded
    ``modified`` changes. ``llm``/``embedder`` are injectable so tests can mock the single ambiguous-
    band LLM call and supply deterministic cosine values."""
    embed = embedder or _default_embedder
    low = cfg.get("delta.notes.semantic_low")
    high = cfg.get("delta.notes.semantic_high")
    if low is None or high is None:
        raise ValueError(
            "delta.notes.semantic_low/high are not calibrated -- run a calibration step "
            "before computing note deltas"
        )
    low, high = float(low), float(high)
    row_tol = float(cfg.get("delta.notes.row_tol"))

    lines_a = _build_lines(notes_a, row_tol)
    lines_b = _build_lines(notes_b, row_tol)
    content_a = [ln.content for ln in lines_a]
    content_b = [ln.content for ln in lines_b]

    sm = difflib.SequenceMatcher(a=content_a, b=content_b, autojunk=False)
    changes: list[Change] = []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for ai, bj in zip(range(i1, i2), range(j1, j2)):
                la, lb = lines_a[ai], lines_b[bj]
                if la.marker != lb.marker:
                    changes.append(_make_renumbered(la, lb))
        elif tag == "delete":
            changes.extend(_make_removed(lines_a[ai]) for ai in range(i1, i2))
        elif tag == "insert":
            changes.extend(_make_added(lines_b[bj]) for bj in range(j1, j2))
        elif tag == "replace":
            pairs = min(i2 - i1, j2 - j1)
            for k in range(pairs):
                ch = _gate_reword(lines_a[i1 + k], lines_b[j1 + k], low, high, llm, embed)
                if ch is not None:
                    changes.append(ch)
            changes.extend(_make_removed(lines_a[ai]) for ai in range(i1 + pairs, i2))
            changes.extend(_make_added(lines_b[bj]) for bj in range(j1 + pairs, j2))

    return changes
