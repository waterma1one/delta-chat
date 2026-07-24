"""Tests for delta/classify.py — deterministic typed classification."""

from __future__ import annotations

from delta_chat.delta.align import MatchResult
from delta_chat.delta.classify import assign_change_sets, classify


def test_unmatched_b_added_unmatched_a_removed(cfg, item_factory):
    a = item_factory("a1", "26-PIT-9055")
    b = item_factory("b1", "26-TIT-9211", source_rev="B", bbox=(0.5, 0.5, 0.6, 0.52))
    mr = MatchResult(
        matched=[], unmatched_a=[a], unmatched_b=[b], margins={("A", "a1"): -0.5, ("B", "b1"): -0.5}
    )

    changes = classify(mr, cfg)
    by_type = {c.type: c for c in changes}
    assert by_type["added"].rev_b_ref == "b1" and by_type["added"].new == "26-TIT-9211"
    assert by_type["removed"].rev_a_ref == "a1" and by_type["removed"].old == "26-PIT-9055"


def test_matched_text_same_centroid_shift_is_moved(cfg, item_factory):
    a = item_factory("a1", "26-FV-9038", bbox=(0.1, 0.1, 0.2, 0.12))
    b = item_factory("b1", "26-FV-9038", source_rev="B", bbox=(0.8, 0.8, 0.9, 0.82))
    mr = MatchResult(matched=[(a, b, 0.9)], unmatched_a=[], unmatched_b=[], margins={})

    changes = classify(mr, cfg)
    assert len(changes) == 1 and changes[0].type == "moved"
    assert changes[0].old == "26-FV-9038" and changes[0].new == "26-FV-9038"


def test_matched_numeric_change_dimension(cfg, item_factory):
    a = item_factory("a1", "150mm", type="dimension", parsed_fields={"value": 150.0, "unit": "mm"})
    b = item_factory("b1", "200mm", source_rev="B", type="dimension", parsed_fields={"value": 200.0, "unit": "mm"})
    mr = MatchResult(matched=[(a, b, 0.6)], unmatched_a=[], unmatched_b=[], margins={})

    changes = classify(mr, cfg)
    assert len(changes) == 1
    ch = changes[0]
    assert ch.type == "modified" and ch.item_type == "dimension"
    assert ch.old == "150mm" and ch.new == "200mm"


def test_matched_tag_change(cfg, item_factory):
    a = item_factory("a1", "26-FV-9038", parsed_fields={"tag_class": "FV", "loop": "9038"})
    b = item_factory("b1", "26-FV-9040", source_rev="B", parsed_fields={"tag_class": "FV", "loop": "9040"})
    mr = MatchResult(matched=[(a, b, 0.85)], unmatched_a=[], unmatched_b=[], margins={})

    changes = classify(mr, cfg)
    assert len(changes) == 1
    ch = changes[0]
    assert ch.type == "modified" and ch.item_type == "tag"
    assert ch.old == "26-FV-9038" and ch.new == "26-FV-9040"


def test_cosmetic_normalization_no_change(cfg, item_factory):
    a = item_factory("a1", "150 mm", type="dimension")
    b = item_factory("b1", "150mm", source_rev="B", type="dimension")
    mr = MatchResult(matched=[(a, b, 1.0)], unmatched_a=[], unmatched_b=[], margins={})

    assert classify(mr, cfg) == []


def test_added_low_confidence_when_margin_near_tau(cfg, item_factory):
    tau = float(cfg.get("delta.tau"))
    b = item_factory("b1", "26-FV-9038", source_rev="B")
    mr_flagged = MatchResult(matched=[], unmatched_a=[], unmatched_b=[b], margins={("B", "b1"): -0.01})
    mr_confident = MatchResult(matched=[], unmatched_a=[], unmatched_b=[b], margins={("B", "b1"): -tau})

    flagged_change = classify(mr_flagged, cfg)[0]
    confident_change = classify(mr_confident, cfg)[0]

    assert flagged_change.match_conf < 0.5
    assert "missed-modify" in flagged_change.description.lower()
    assert confident_change.match_conf > flagged_change.match_conf


def test_revision_table_row_added(cfg, item_factory):
    b = item_factory("b1", "Rev C - updated tag", source_rev="B", type="table_cell", region="revtable")
    mr = MatchResult(matched=[], unmatched_a=[], unmatched_b=[b], margins={("B", "b1"): -0.5})

    ch = classify(mr, cfg)[0]
    assert ch.type == "added" and ch.item_type == "table_cell"
    assert "revision" in ch.description.lower()


def test_preexisting_redline_not_emitted_as_delta(cfg, item_factory):
    # Same literal authored text on both sides -> matched with sim ~= 1, no centroid shift -> unchanged.
    a = item_factory("a1", "6. DELETED.", type="text", region="notes")
    b = item_factory("b1", "6. DELETED.", source_rev="B", type="text", region="notes")
    mr = MatchResult(matched=[(a, b, 1.0)], unmatched_a=[], unmatched_b=[], margins={})

    assert classify(mr, cfg) == []


def test_change_set_grouping_clusters_spatially_near_changes(cfg, item_factory):
    near_a = item_factory("a1", "TAG-1", bbox=(0.10, 0.10, 0.20, 0.12))
    near_b = item_factory("b1", "TAG-1-NEW", source_rev="B", bbox=(0.10, 0.10, 0.20, 0.12))
    close_added_b = item_factory("b2", "TAG-2", source_rev="B", bbox=(0.12, 0.10, 0.22, 0.12))
    far_removed_a = item_factory("a3", "TAG-3", bbox=(0.9, 0.9, 0.95, 0.92))
    mr = MatchResult(
        matched=[(near_a, near_b, 0.5)],
        unmatched_a=[far_removed_a],
        unmatched_b=[close_added_b],
        margins={("A", "a3"): -0.5, ("B", "b2"): -0.5},
    )
    changes = classify(mr, cfg)
    radius = float(cfg.get("delta.change_set_radius"))
    assign_change_sets(changes, radius)

    by_ref = {(c.rev_a_ref, c.rev_b_ref): c for c in changes}
    modified = by_ref[("a1", "b1")]
    close_added = by_ref[(None, "b2")]
    far_removed = by_ref[("a3", None)]
    assert modified.change_set_id == close_added.change_set_id
    assert modified.change_set_id != far_removed.change_set_id
