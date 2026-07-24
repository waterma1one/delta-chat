"""Tests for delta/align.py — similarity blending + Hungarian matching + reconciliation."""

from __future__ import annotations

import inspect

from delta_chat.delta.align import SimWeights, align, similarity


def test_similarity_blends_components_text_dominant(item_factory):
    w = SimWeights()
    a = item_factory("a1", "26-FV-9038", bbox=(0.1, 0.1, 0.2, 0.12))
    b_same_pos = item_factory("b1", "26-FV-9038", source_rev="B", bbox=(0.1, 0.1, 0.2, 0.12))
    b_far = item_factory("b2", "26-FV-9038", source_rev="B", bbox=(0.85, 0.85, 0.95, 0.87))
    b_diff_text = item_factory("b3", "26-PIT-9999-ZZZZ", source_rev="B", bbox=(0.1, 0.1, 0.2, 0.12))

    sim_same = similarity(a, b_same_pos, w)
    sim_far = similarity(a, b_far, w)
    sim_diff_text = similarity(a, b_diff_text, w)

    assert sim_same > sim_far > sim_diff_text


def test_identical_item_same_position_sim_near_one(item_factory):
    a = item_factory("a1", "26-FV-9038")
    b = item_factory("b1", "26-FV-9038", source_rev="B")
    assert similarity(a, b, SimWeights()) > 0.99


def test_moved_identical_tag_still_matches_not_split(item_factory):
    a = item_factory("a1", "26-FV-9038", bbox=(0.1, 0.1, 0.2, 0.12))
    b = item_factory("b1", "26-FV-9038", source_rev="B", bbox=(0.85, 0.85, 0.95, 0.87))
    result = align([a], [b], tau=0.35)
    assert [(pa.id, pb.id) for pa, pb, _ in result.matched] == [("a1", "b1")]
    assert not result.unmatched_a and not result.unmatched_b


def test_duplicate_labels_one_moved_no_crossed_match(item_factory):
    a1 = item_factory("a1", "FV-101", bbox=(0.10, 0.10, 0.15, 0.12))
    a2 = item_factory("a2", "FV-101", bbox=(0.50, 0.50, 0.55, 0.52))
    b1 = item_factory("b1", "FV-101", source_rev="B", bbox=(0.10, 0.10, 0.15, 0.12))
    b2 = item_factory("b2", "FV-101", source_rev="B", bbox=(0.85, 0.85, 0.90, 0.87))  # a2 moved here

    result = align([a1, a2], [b1, b2], tau=0.35)
    pairs = {(pa.id, pb.id) for pa, pb, _ in result.matched}
    assert pairs == {("a1", "b1"), ("a2", "b2")}


def test_align_returns_unmatched_with_runner_up_margins(item_factory):
    a = item_factory("a1", "26-FV-9038", type="tag", region="drawing", bbox=(0.05, 0.05, 0.10, 0.07))
    b = item_factory(
        "b1", "42in", source_rev="B", type="dimension", region="revtable", bbox=(0.90, 0.90, 0.95, 0.92)
    )

    result = align([a], [b], tau=0.35)
    assert result.unmatched_a == [a]
    assert result.unmatched_b == [b]
    assert ("A", "a1") in result.margins and ("B", "b1") in result.margins
    assert result.margins[("A", "a1")] < 0


def test_margins_keyed_by_source_rev_survive_id_collisions(item_factory):
    # Item.id is only stable *within* a Document (ingestion assigns it positionally), so rev-A and
    # rev-B unmatched items routinely share an id string. Margins must not clobber across revisions.
    a = item_factory(
        "0:drawing:1", "26-FV-9038", type="tag", region="drawing", bbox=(0.05, 0.05, 0.10, 0.07)
    )
    b = item_factory(
        "0:drawing:1", "42in", source_rev="B", type="dimension", region="revtable", bbox=(0.90, 0.90, 0.95, 0.92)
    )

    result = align([a], [b], tau=0.35)

    # Both revisions' margins must survive -- a same-key dict merge would silently drop one.
    assert len(result.margins) == 2
    assert ("A", "0:drawing:1") in result.margins
    assert ("B", "0:drawing:1") in result.margins


def test_global_reconciliation_recovers_boundary_item(item_factory):
    a = item_factory("a1", "26-FV-9038", region="drawing", bbox=(0.5, 0.5, 0.6, 0.52))
    b = item_factory("b1", "26-FV-9038", source_rev="B", region="unknown", bbox=(0.5, 0.5, 0.6, 0.52))

    result = align([a], [b], tau=0.35)
    assert [(pa.id, pb.id) for pa, pb, _ in result.matched] == [("a1", "b1")]


def test_default_spatial_metric_is_centroid():
    sig = inspect.signature(align)
    assert sig.parameters["spatial_metric"].default == "centroid"


def test_rectangular_padding_handles_unequal_counts(item_factory):
    a_items = [item_factory(f"a{i}", f"TAG-{i}", bbox=(0.05 * i, 0.10, 0.05 * i + 0.03, 0.12)) for i in range(3)]
    b_items = [
        item_factory(f"b{i}", f"TAG-{i}", source_rev="B", bbox=(0.05 * i, 0.10, 0.05 * i + 0.03, 0.12))
        for i in range(5)
    ]

    result = align(a_items, b_items, tau=0.35)
    assert len(result.matched) == 3
    assert len(result.unmatched_a) == 0
    assert len(result.unmatched_b) == 2
