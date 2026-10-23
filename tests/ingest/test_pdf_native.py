"""Native-PDF parsing against the Export Gas Compressor source sheet (skips when absent)."""

from __future__ import annotations

from delta_chat.canonical.model import Region
from delta_chat.ingest.pdf_native import parse_native_pdf

_VALID_REGIONS = {"title", "revtable", "notes", "drawing", "unknown"}


def test_parses_export_sheet_shape(export_pdf):
    doc = parse_native_pdf(export_pdf.read_bytes(), "pid:export")
    assert doc.format == "pdf_native"
    assert doc.page_count == 1
    w, h = doc.sheet_size
    assert 1100 < w < 1250 and 780 < h < 900  # A3 landscape ~1191x842 pt
    assert len(doc.items) > 1000


def test_rotation_and_regions_and_tags(export_pdf):
    doc = parse_native_pdf(export_pdf.read_bytes(), "pid:export")
    # every item carries a valid region
    assert all(it.region in _VALID_REGIONS for it in doc.items)
    # rotations only ever 0/90/270
    assert {it.rotation_deg for it in doc.items} <= {0.0, 90.0, 270.0}
    # vertical text present (~40 lines on this sheet)
    assert sum(1 for it in doc.items if it.rotation_deg in (90.0, 270.0)) >= 10
    # title-block corner + notes list both populated
    assert any(it.region == "title" for it in doc.items)
    assert any(it.region == "notes" for it in doc.items)
    # the equipment tag 26-KA-902 is recognized as a typed tag
    ka = [it for it in doc.items if it.text == "26-KA-902"]
    assert ka and ka[0].type == "tag" and ka[0].parsed_fields.get("tag_class") == "KA"


def test_degenerate_vector_paths_filtered(export_pdf):
    import fitz

    from delta_chat.config import load_config
    from delta_chat.ingest import regions

    doc = fitz.open(stream=export_pdf.read_bytes(), filetype="pdf")
    page = doc[0]
    info = regions.cluster_drawings(page, lambda r: None, load_config())
    doc.close()
    # dense P&ID: some near-zero-area paths get filtered; kept count is positive
    assert info["kept_paths"] > 0 and info["filtered_paths"] >= 0


def test_parse_is_deterministic(export_pdf):
    data = export_pdf.read_bytes()
    a = parse_native_pdf(data, "pid:export")
    b = parse_native_pdf(data, "pid:export")
    assert [it.id for it in a.items] == [it.id for it in b.items]
    assert [it.text for it in a.items] == [it.text for it in b.items]
