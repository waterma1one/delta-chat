"""Format detection + adapter routing."""

from __future__ import annotations

import pytest

from delta_chat.ingest import base


def test_detects_native_pdf(export_pdf):
    data = export_pdf.read_bytes()
    assert base.detect_format(data, "export.pdf") == "pdf_native"
    adapter = base.select_adapter(data, "export.pdf")
    assert adapter.format == "pdf_native"


def test_detects_scanned_when_text_layer_absent(export_pdf, monkeypatch):
    data = export_pdf.read_bytes()
    monkeypatch.setattr(base, "native_word_count", lambda d: 0)  # simulate image-only PDF
    assert base.detect_format(data, "scan.pdf") == "pdf_scanned"


def test_detects_dwg_by_extension_and_magic():
    assert base.detect_format(b"AC1027\x00\x00", "drawing.dwg") == "dwg"
    assert base.detect_format(b"0\nSECTION\n2\nHEADER", "drawing.dxf") == "dwg"


def test_unknown_format_raises_clear_error():
    with pytest.raises(base.UnknownFormatError):
        base.detect_format(b"not a known file at all", "mystery.bin")
