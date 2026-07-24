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


# --- native-vs-scanned probe (regression: a raw word-count threshold got all of these wrong) -----


def _pdf(*, words: str | None = None, full_page_image: bool = False, vectors: bool = False) -> bytes:
    """Build a one-page PDF with an optional text layer, page-covering raster, and vector geometry."""
    import fitz

    doc = fitz.open()
    page = doc.new_page(width=200, height=200)
    if full_page_image:
        pix = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 200, 200))
        pix.clear_with(255)
        page.insert_image(page.rect, pixmap=pix)
    if vectors:
        page.draw_line((10, 10), (190, 10))
        page.draw_rect(fitz.Rect(20, 20, 80, 80))
    if words:
        # lay the words out line by line so they all fit on the page and stay extractable
        # (a single long insert_text run overflows the page and silently clips most of them)
        for i, chunk in enumerate(_chunks(words.split(), 4)):
            page.insert_text((10, 15 + i * 10), " ".join(chunk), fontsize=6)
    return doc.tobytes()


def _chunks(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def test_sparse_native_pdf_is_native_not_scanned():
    """A born-digital drawing legitimately carries very few words (a P&ID is mostly geometry).
    The old `words > 20` rule classified every such page as a scan and sent it through OCR."""
    data = _pdf(words="26-FV-9040 150mm")  # 2 words, far below the old threshold of 20
    assert base.detect_format(data, "sparse.pdf") == "pdf_native"


def test_image_only_pdf_is_scanned():
    assert base.detect_format(_pdf(full_page_image=True), "scan.pdf") == "pdf_scanned"


def test_native_drawing_with_full_page_background_raster_is_still_native():
    """Regression: a born-digital drawing can carry a page-sized underlay/watermark/scanned
    title-block AND a perfect vector+text layer. Image-dominance alone flipped it to pdf_scanned
    and re-OCR'd exact text. Vector geometry keeps it native."""
    data = _pdf(words="26-FV-9040 150mm SUCTION", full_page_image=True, vectors=True)
    assert base.max_image_area_frac(data) >= 0.80  # raster does dominate the page
    assert base.vector_drawing_count(data) >= 1     # but there is real geometry
    assert base.detect_format(data, "native_with_underlay.pdf") == "pdf_native"


def test_scan_with_embedded_ocr_text_layer_is_still_scanned():
    """The case a word count can never get right: a scanned page whose OCR layer was baked back in
    has plenty of extractable words, but it is still a raster scan and must route to the OCR path."""
    data = _pdf(words=" ".join(f"OCRWORD{i}" for i in range(40)), full_page_image=True)
    # comfortably above the old `words > 20` rule, so this asserts the image-dominance signal and
    # not merely a low word count.
    assert base.native_word_count(data) > 20
    assert base.detect_format(data, "scan_with_ocr.pdf") == "pdf_scanned"


def test_detection_thresholds_are_config_driven(export_pdf):
    from delta_chat.config import Config, load_config

    data = export_pdf.read_bytes()
    assert base.detect_format(data, "export.pdf") == "pdf_native"
    raised = Config({**load_config().data})
    raised.data["ingest"] = {**raised.data["ingest"], "native_min_words": 10_000}
    assert base.detect_format(data, "export.pdf", raised) == "pdf_scanned"


def test_corrupt_pdf_raises_traceable_error():
    with pytest.raises(base.UnknownFormatError):
        base.detect_format(b"%PDF-1.4\nbut truncated garbage", "broken.pdf")
