"""Orchestrates align -> classify -> note-diff into a stable, deterministic ``Delta``.

Determinism is a hard requirement: changes are sorted by a stable key (page, sheet, bbox, type, item
type, old/new) before ``CHG-000N`` ids are assigned, so two runs over the same input produce byte-
identical JSON. The only non-deterministic element permitted anywhere in the pipeline is the note-
gate's ambiguous-band LLM call (``notes.py``) — the structural delta itself is pure code.
"""

from __future__ import annotations

from typing import Callable, Sequence

from ..canonical.model import Change, Delta, Document
from ..observability import metrics as _metrics
from ..observability import tracing as _tracing
from .align import SimWeights, align
from .classify import assign_change_sets, classify
from .notes import diff_notes


def _sort_key(ch: Change):
    return (
        ch.page,
        ch.sheet or "",
        ch.bbox.y0,
        ch.bbox.x0,
        ch.type,
        ch.item_type,
        ch.old or "",
        ch.new or "",
    )


def compute_delta(
    doc_a: Document,
    doc_b: Document,
    cfg,
    llm: Callable[[str, str], dict] | None = None,
    embedder: Callable[[str], Sequence[float]] | None = None,
) -> Delta:
    """Orchestrate the full delta:
      1. split items by region: notes -> notes.diff_notes; everything else -> align.align.
      2. classify.classify(match_result) -> typed Changes with dual confidence.
      3. merge note changes + classified changes; assign stable CHG-000N ids (deterministic order);
         group into change-sets by proximity; tally counts by type.
    Returns ``Delta(pid_a, pid_b, changes, counts, dataset_version)``.
    """
    notes_a = [it for it in doc_a.items if it.region == "notes"]
    notes_b = [it for it in doc_b.items if it.region == "notes"]
    other_a = [it for it in doc_a.items if it.region != "notes"]
    other_b = [it for it in doc_b.items if it.region != "notes"]

    tau = float(cfg.get("delta.tau"))
    spatial_metric = cfg.get("delta.spatial_metric")
    weights = SimWeights.from_config(cfg)

    match_result = align(other_a, other_b, tau, weights=weights, spatial_metric=spatial_metric)
    changes = classify(match_result, cfg)
    changes.extend(diff_notes(notes_a, notes_b, cfg, llm=llm, embedder=embedder))

    changes.sort(key=_sort_key)
    for i, ch in enumerate(changes, start=1):
        ch.id = f"CHG-{i:04d}"

    radius = float(cfg.get("delta.change_set_radius"))
    assign_change_sets(changes, radius)

    counts: dict[str, int] = {}
    for ch in changes:
        counts[ch.type] = counts.get(ch.type, 0) + 1

    # Surface the delta size on the Prometheus counter and on the active trace's `delta` stage.
    # "delta counts" is a required metric; the counter existed but nothing ever incremented it, so
    # `delta_chat_delta_changes_total` sat at 0 no matter how many changes a run produced.
    _metrics.observe_delta_count(len(changes))
    _tracing.annotate_stage(change_count=len(changes), counts_by_type=counts)

    return Delta(
        pid_a=doc_a.pid,
        pid_b=doc_b.pid,
        changes=changes,
        counts=counts,
        dataset_version=cfg.get("eval.datasets.dataset_version", None),
    )
