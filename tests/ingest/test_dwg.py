"""DWG/DXF adapter tests — authored DXF fixtures + the synthesized DXF pair."""

from __future__ import annotations

import io

import ezdxf
import pytest

from delta_chat.ingest.dwg import parse_dwg
from delta_chat.ingest.pdf_native import parse_native_pdf


def _dxf_bytes(doc) -> bytes:
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


def _authored_doc():
    """A small in-memory DXF exercising TEXT, MTEXT (with formatting codes), an INSERT with an
    attribute, and a DIMENSION -- the synthesized fixture doesn't happen to contain a TEXT or
    DIMENSION entity, so this authors them directly per the section's test spec."""
    doc = ezdxf.new("R2018", setup=True)
    msp = doc.modelspace()

    msp.add_text("PLAIN LABEL", dxfattribs={"layer": "TXT", "insert": (0, 0), "height": 2.5})
    msp.add_mtext(
        r"\A1;26-KA-902\P{\fArial|b0|i0;NOTE}",
        dxfattribs={"layer": "MTXT", "insert": (10, 10), "char_height": 2.5},
    )

    block = doc.blocks.new(name="SYM")
    block.add_circle((0, 0), radius=1)
    insert = msp.add_blockref("SYM", (5, 5), dxfattribs={"layer": "SYMS"})
    insert.add_attrib("TAG", "26-FV-9038", insert=(5, 5))

    dim = msp.add_linear_dim(base=(0, 5), p1=(0, 0), p2=(10, 0), dimstyle="EZDXF")
    dim.render()

    return doc


def test_parse_dxf_yields_items_for_text_mtext_insert_attribs_dimension():
    data = _dxf_bytes(_authored_doc())
    doc = parse_dwg(data, "pid:authored")

    by_text = {it.text: it for it in doc.items}
    assert "PLAIN LABEL" in by_text  # TEXT
    assert any(it.layer == "MTXT" for it in doc.items)  # MTEXT
    assert any(it.layer == "SYMS" for it in doc.items)  # the INSERT itself
    assert "26-FV-9038" in by_text  # the INSERT's attrib
    dim_items = [it for it in doc.items if it.type == "dimension"]  # DIMENSION
    assert len(dim_items) == 1

    # section spec: every entity type carries a non-null layer, normalized bbox, source_engine ==
    # "ezdxf", conf == 1.0 (DXF is exact, not OCR'd) -- checked across all four types, not just TEXT.
    checked = [
        by_text["PLAIN LABEL"],
        next(it for it in doc.items if it.layer == "MTXT"),
        next(it for it in doc.items if it.layer == "SYMS"),
        by_text["26-FV-9038"],
        dim_items[0],
    ]
    for it in checked:
        assert it.source_engine == "ezdxf"
        assert it.conf == 1.0
        assert it.layer is not None
        assert 0.0 <= it.bbox.x0 <= it.bbox.x1 <= 1.0
        assert 0.0 <= it.bbox.y0 <= it.bbox.y1 <= 1.0


def test_mtext_plain_text_strips_formatting_codes():
    data = _dxf_bytes(_authored_doc())
    doc = parse_dwg(data, "pid:authored")

    mtext_items = [it for it in doc.items if it.layer == "MTXT"]
    assert len(mtext_items) == 1
    text = mtext_items[0].text
    assert "\\A1;" not in text and "\\P" not in text and "\\f" not in text
    assert "26-KA-902" in text and "NOTE" in text


def test_dimension_measurement_captured():
    data = _dxf_bytes(_authored_doc())
    doc = parse_dwg(data, "pid:authored")

    dim_items = [it for it in doc.items if it.type == "dimension"]
    assert len(dim_items) == 1
    assert dim_items[0].parsed_fields["value"] == pytest.approx(10.0)
    assert dim_items[0].parsed_fields["unit"] == "m"


def test_malformed_dxf_raises_clear_error_not_swallowed():
    from delta_chat.ingest.dwg import DwgParseError

    with pytest.raises(DwgParseError, match="malformed DXF"):
        parse_dwg(b"not a dxf file at all", "pid:garbage")


def test_converter_failure_raises_clear_error_not_swallowed(monkeypatch):
    import delta_chat.ingest.dwg as dwg_mod

    monkeypatch.setattr(dwg_mod, "_locate_converter", lambda: "/usr/bin/false")

    def _boom(*a, **k):
        import subprocess

        raise subprocess.CalledProcessError(1, ["dwg2dxf"], stderr=b"conversion failed")

    monkeypatch.setattr(dwg_mod.subprocess, "run", _boom)
    fake_dwg_bytes = b"AC1027" + b"\x00" * 64

    with pytest.raises(dwg_mod.DwgParseError, match="converter failed"):
        parse_dwg(fake_dwg_bytes, "pid:broken-converter")


def test_dwg_without_converter_returns_honest_stub(monkeypatch):
    import delta_chat.ingest.dwg as dwg_mod

    monkeypatch.setattr(dwg_mod, "_locate_converter", lambda: None)
    fake_dwg_bytes = b"AC1027" + b"\x00" * 64  # magic bytes only -- never reaches ezdxf

    doc = parse_dwg(fake_dwg_bytes, "pid:stub")

    assert doc.format == "dwg"
    assert doc.items == []
    assert doc.raw_meta["unsupported"] is True
    assert "would_run" in doc.raw_meta and "dwg2dxf" in doc.raw_meta["would_run"]


def test_dxf_input_always_parses_even_without_converter(monkeypatch):
    import delta_chat.ingest.dwg as dwg_mod

    monkeypatch.setattr(dwg_mod, "_locate_converter", lambda: None)
    data = _dxf_bytes(_authored_doc())

    doc = parse_dwg(data, "pid:authored")

    assert doc.raw_meta.get("unsupported") is not True
    assert len(doc.items) > 0


def test_adapter_emits_same_item_schema_as_pdf_native(export_pdf):
    dxf_doc = parse_dwg(_dxf_bytes(_authored_doc()), "pid:authored")
    native_doc = parse_native_pdf(export_pdf.read_bytes(), "pid:export")

    dxf_fields = set(vars(dxf_doc.items[0]).keys())
    native_fields = set(vars(native_doc.items[0]).keys())
    assert dxf_fields == native_fields

    doc_fields = set(vars(dxf_doc).keys())
    native_doc_fields = set(vars(native_doc).keys())
    assert doc_fields == native_doc_fields

    for it in dxf_doc.items:
        assert 0.0 <= it.bbox.x0 <= it.bbox.x1 <= 1.0
        assert 0.0 <= it.bbox.y0 <= it.bbox.y1 <= 1.0


def test_section04_synthesized_dxf_pair_reproduces_gold(repo_root):
    import json

    pair_dir = repo_root / "data" / "samples" / "pair_05_dxf"
    doc_a = parse_dwg((pair_dir / "rev_a.dxf").read_bytes(), "pid:pair_05_dxf:A", source_rev="A")
    doc_b = parse_dwg((pair_dir / "rev_b.dxf").read_bytes(), "pid:pair_05_dxf:B", source_rev="B")

    assert any(it.text == "26-KA-902" for it in doc_a.items)
    assert any(it.text == "26-KA-903" for it in doc_b.items)
    assert any(it.type == "symbol" and it.text == "LINE" for it in doc_b.items)
    # the line gold calls "added" exists in B but not A at the same location
    added_line_bbox = (0.0, 0.368421052631579, 1.0, 0.368421052631579)
    assert not any(
        it.type == "symbol" and it.text == "LINE" and it.bbox.as_tuple() == added_line_bbox
        for it in doc_a.items
    )
    assert any(
        it.type == "symbol" and it.text == "LINE" and it.bbox.as_tuple() == added_line_bbox
        for it in doc_b.items
    )

    gold_path = repo_root / "eval" / "datasets" / "delta_gold.jsonl"
    gold = [
        json.loads(line)
        for line in gold_path.read_text().splitlines()
        if line.strip() and json.loads(line)["pair_id"] == "pair_05_dxf"
    ]
    assert len(gold) == 3
