"""Tests for the canonical model + normalize_bbox (the load-bearing coordinate invariant)."""

from __future__ import annotations

import math

import pytest

from delta_chat.canonical.model import BBox, Change, Delta, Document, Item, normalize_bbox


def _approx(b: BBox, expected: tuple[float, float, float, float], tol: float = 1e-6) -> bool:
    return all(math.isclose(a, e, abs_tol=tol) for a, e in zip(b.as_tuple(), expected))


# --- BBox contract -------------------------------------------------------------------------------

def test_bbox_clamps_out_of_range_to_unit_interval():
    b = BBox(-0.2, 0.5, 1.4, 0.9)
    assert b.x0 == 0.0 and b.x1 == 1.0
    assert b.y0 == 0.5 and b.y1 == 0.9
    # in-range values pass through unchanged
    assert BBox(0.1, 0.2, 0.3, 0.4).as_tuple() == (0.1, 0.2, 0.3, 0.4)


def test_bbox_is_hashable_and_frozen():
    b = BBox(0.1, 0.2, 0.3, 0.4)
    assert hash(b) == hash(BBox(0.1, 0.2, 0.3, 0.4))
    with pytest.raises(Exception):
        b.x0 = 0.9  # type: ignore[misc]


# --- normalize_bbox per source space -------------------------------------------------------------

def test_normalize_pdf_points_flips_y_from_bottom_left():
    # PDF bottom-left origin: y grows upward. Box near the TOP of the page (high y) must land near
    # the TOP in top-left normalized space (small y).
    b = normalize_bbox((100, 700, 200, 800), "pdf_points", (1000, 1000))
    assert _approx(b, (0.1, 0.2, 0.2, 0.3))
    # the PDF-top edge (y=800) became the smaller normalized y -> y flipped.
    assert b.y0 == pytest.approx(0.2)


def test_normalize_pdf_topleft_no_flip():
    # PyMuPDF top-left space: straight divide, no y-flip.
    b = normalize_bbox((100, 100, 200, 150), "pdf_topleft", (1000, 1000))
    assert _approx(b, (0.1, 0.1, 0.2, 0.15))
    with pytest.raises(ValueError):
        normalize_bbox((0, 0, 1, 1), "pdf_topleft", (0, 100))


def test_normalize_image_px_roundtrips_to_native_normalized():
    page_pts = (612.0, 792.0)
    dpi = 200.0
    native = (0.10, 0.20, 0.30, 0.40)  # a known normalized span (top-left)
    # render it to raster pixels at dpi, then normalize back
    px = tuple(  # x0,y0,x1,y1 in pixels
        n * (page_pts[i % 2]) * dpi / 72.0 for i, n in enumerate(native)
    )
    b = normalize_bbox(px, "image_px", page_pts, dpi=dpi)
    assert _approx(b, native, tol=1e-6)


def test_normalize_dxf_wcs_maps_model_units_into_unit_interval():
    b = normalize_bbox((0, 0, 50, 100), "dxf_wcs", (100, 200), insunits=4)
    for v in b.as_tuple():
        assert 0.0 <= v <= 1.0
    assert _approx(b, (0.0, 0.5, 0.5, 1.0))


def test_normalize_bbox_rejects_degenerate_page():
    with pytest.raises(ValueError):
        normalize_bbox((0, 0, 1, 1), "pdf_points", (0, 100))
    with pytest.raises(ValueError):
        normalize_bbox((0, 0, 1, 1), "image_px", (100, 100), dpi=0)
    with pytest.raises(ValueError):
        normalize_bbox((0, 0, 1, 1), "dxf_wcs", (100, 0))


# --- Item / Change field contracts ---------------------------------------------------------------

def _mk_item(seq: int) -> Item:
    return Item(
        id=f"0:drawing:{seq}",
        source_rev="A",
        type="tag",
        text="26-KA-902",
        parsed_fields={"tag_class": "KA", "loop": "902"},
        bbox=BBox(0.1, 0.1, 0.2, 0.2),
        page=0,
        sheet=None,
        region="drawing",
    )


def test_item_id_stable_for_same_inputs_and_sheet_optional():
    a, b = _mk_item(3), _mk_item(3)
    assert a == b
    assert a.id == "0:drawing:3"
    assert a.sheet is None  # sheet may be None


def test_change_carries_provenance_and_dual_confidence():
    ch = Change(
        id="CHG-0001",
        type="modified",
        item_type="tag",
        page=0,
        sheet="3",
        bbox=BBox(0.4, 0.4, 0.5, 0.5),
        description="tag 9038 -> 9040",
        old="FV-9038",
        new="FV-9040",
        rev_a_ref="0:drawing:12",
        rev_b_ref="0:drawing:12",
        match_conf=0.91,
        class_conf=0.88,
        change_set_id="CS-1",
        hard_slice=True,
    )
    assert ch.rev_a_ref == "0:drawing:12" and ch.rev_b_ref == "0:drawing:12"
    assert ch.hard_slice is True
    assert (ch.match_conf, ch.class_conf) == (0.91, 0.88)


def test_document_and_delta_construct_with_defaults():
    doc = Document(
        pid="pidA",
        format="pdf_native",
        page_count=1,
        revision_label=None,
        sheet_size=(1191.0, 842.0),
        items=[_mk_item(1)],
    )
    assert doc.regions == [] and doc.raw_meta == {}
    d = Delta(pid_a="pidA", pid_b="pidB")
    assert d.changes == [] and d.counts == {}
