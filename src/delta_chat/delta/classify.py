"""Deterministic typed classification: ``MatchResult`` -> ``list[Change]``.

Fully deterministic — no LLM ever authors a value here. Thresholds and weights come from
``cfg.delta.*``; nothing below is hardcoded.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from ..canonical.model import Change, Item
from .align import MatchResult, SimWeights, centroid_shift_frac, normalize_text

# Appended to the description when an add/remove's runner-up margin is close to tau — i.e. the best
# rejected candidate on the other side almost cleared the match bar, so this is plausibly a missed
# modify rather than a genuine add/remove. Surfaced via description + low match_conf since the shared
# Change schema has no dedicated flag field.
_MISSED_MODIFY_SUFFIX = " [possible missed-modify: low confidence]"


def _add_remove_confidence(margin: float, tau: float, flag_below: float) -> tuple[float, bool]:
    """margin = best-rejected-candidate similarity - tau. Confidence is 1.0 when nothing on the other
    side came close (margin == -tau) and falls to 0.0 as the runner-up approaches/clears tau."""
    if tau <= 0:
        return 1.0, False
    normalized = max(0.0, min(1.0, (margin + tau) / tau))
    conf = 1.0 - normalized
    return conf, conf < flag_below


def _typed_value(item: Item) -> str:
    """Reconstruct a canonical old/new value string for numeric types; fall back to raw text (tags
    and everything else carry their canonical form directly in ``Item.text``)."""
    if item.type == "dimension" and "value" in item.parsed_fields:
        v = item.parsed_fields["value"]
        unit = item.parsed_fields.get("unit", "")
        v_str = str(int(v)) if float(v).is_integer() else str(v)
        return f"{v_str}{unit}"
    return item.text


def _describe_added(item: Item) -> str:
    if item.region == "revtable":
        return f"Revision row added: {item.text}"
    return f"{item.type.capitalize()} added: {item.text}"


def _describe_removed(item: Item) -> str:
    if item.region == "revtable":
        return f"Revision row removed: {item.text}"
    return f"{item.type.capitalize()} removed: {item.text}"


def classify(match_result: MatchResult, cfg) -> list[Change]:
    """Deterministic typed classification:
      unmatched_B -> ADDED ; unmatched_A -> REMOVED ;
      matched, text ~= 1 & centroid shift large -> MOVED ;
      matched, text differs -> MODIFIED (old/new from the deterministic parse, never LLM-authored).
    ``change_set_id`` is left blank here; ``engine.py`` assigns it once note changes are merged in
    (see ``assign_change_sets``).
    """
    thresholds = cfg.get("delta.thresholds")
    unchanged_text_sim = float(thresholds["unchanged_text_sim"])
    unchanged_centroid_frac = float(thresholds["unchanged_centroid_frac"])
    moved_text_sim = float(thresholds["moved_text_sim"])
    tau = float(cfg.get("delta.tau"))
    missed_modify_conf = float(cfg.get("delta.thresholds.missed_modify_conf"))

    changes: list[Change] = []

    for a, b, sim in match_result.matched:
        text_sim = fuzz.WRatio(normalize_text(a.text), normalize_text(b.text)) / 100.0
        shift = centroid_shift_frac(a.bbox, b.bbox)

        if text_sim >= unchanged_text_sim and shift < unchanged_centroid_frac:
            continue  # unchanged: cosmetic normalization already folded into text_sim

        if text_sim >= moved_text_sim and shift >= unchanged_centroid_frac:
            changes.append(
                Change(
                    id="",
                    type="moved",
                    item_type=b.type,
                    page=b.page,
                    sheet=b.sheet,
                    bbox=b.bbox,
                    description=f"{b.type.capitalize()} moved: {b.text}",
                    old=a.text,
                    new=b.text,
                    rev_a_ref=a.id,
                    rev_b_ref=b.id,
                    match_conf=sim,
                    class_conf=1.0,
                    change_set_id="",
                )
            )
            continue

        old_v, new_v = _typed_value(a), _typed_value(b)
        item_type = b.type
        if item_type == "dimension":
            desc = f"Dimension {old_v} -> {new_v}"
        elif item_type == "tag":
            desc = f"Tag {old_v} -> {new_v}"
        else:
            desc = f"{item_type.capitalize()} changed: {old_v} -> {new_v}"
        changes.append(
            Change(
                id="",
                type="modified",
                item_type=item_type,
                page=b.page,
                sheet=b.sheet,
                bbox=b.bbox,
                description=desc,
                old=old_v,
                new=new_v,
                rev_a_ref=a.id,
                rev_b_ref=b.id,
                match_conf=sim,
                class_conf=min(a.conf, b.conf),
                change_set_id="",
            )
        )

    for item in match_result.unmatched_b:
        margin = match_result.margins.get((item.source_rev, item.id), -tau)
        conf, flagged = _add_remove_confidence(margin, tau, missed_modify_conf)
        desc = _describe_added(item)
        if flagged:
            desc += _MISSED_MODIFY_SUFFIX
        changes.append(
            Change(
                id="",
                type="added",
                item_type=item.type,
                page=item.page,
                sheet=item.sheet,
                bbox=item.bbox,
                description=desc,
                old=None,
                new=_typed_value(item),
                rev_a_ref=None,
                rev_b_ref=item.id,
                match_conf=conf,
                class_conf=item.conf,
                change_set_id="",
            )
        )

    for item in match_result.unmatched_a:
        margin = match_result.margins.get((item.source_rev, item.id), -tau)
        conf, flagged = _add_remove_confidence(margin, tau, missed_modify_conf)
        desc = _describe_removed(item)
        if flagged:
            desc += _MISSED_MODIFY_SUFFIX
        changes.append(
            Change(
                id="",
                type="removed",
                item_type=item.type,
                page=item.page,
                sheet=item.sheet,
                bbox=item.bbox,
                description=desc,
                old=_typed_value(item),
                new=None,
                rev_a_ref=item.id,
                rev_b_ref=None,
                match_conf=conf,
                class_conf=item.conf,
                change_set_id="",
            )
        )

    return changes


def assign_change_sets(changes: list[Change], radius: float) -> None:
    """Deterministic greedy single-linkage proximity clustering: same-page changes within ``radius``
    (fraction of page diagonal) of *any* existing member of a cluster join that cluster (not just its
    first member), so a chain of near-neighbors groups together even if the ends are farther apart
    than ``radius``. Mutates ``changes`` in place; called once by ``engine.py`` after classified + note
    changes are merged."""
    order = sorted(
        range(len(changes)),
        key=lambda i: (changes[i].page, changes[i].bbox.centroid[1], changes[i].bbox.centroid[0], i),
    )
    clusters: list[list[tuple[float, float]]] = []  # per-cluster list of member (cx, cy)
    cluster_pages: list[int] = []
    cluster_ids: list[str] = []
    counters: dict[int, int] = {}

    for idx in order:
        ch = changes[idx]
        cx, cy = ch.bbox.centroid
        found = None
        for ci, members in enumerate(clusters):
            if cluster_pages[ci] == ch.page and any(
                ((cx - mx) ** 2 + (cy - my) ** 2) ** 0.5 <= radius for mx, my in members
            ):
                found = ci
                break
        if found is None:
            n = counters.get(ch.page, 0) + 1
            counters[ch.page] = n
            clusters.append([(cx, cy)])
            cluster_pages.append(ch.page)
            cluster_ids.append(f"CS-{ch.page}-{n}")
            found = len(clusters) - 1
        else:
            clusters[found].append((cx, cy))
        changes[idx].change_set_id = cluster_ids[found]
