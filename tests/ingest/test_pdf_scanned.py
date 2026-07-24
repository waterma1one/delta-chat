"""Scanned-PDF adapter tests. Tesseract-dependent tests skip cleanly (with a clear reason) when the
system binary isn't installed, rather than failing CI -- per the section spec."""

from __future__ import annotations

import copy

import fitz
import numpy as np
import pytest

from delta_chat.config import Config, load_config
from delta_chat.ingest.pdf_scanned import (
    _map_rotated_box_back,
    _ocr_words,
    parse_scanned_pdf,
    preprocess_for_ocr,
)


def _tesseract_available() -> bool:
    try:
        import pytesseract

        pytesseract.get_tesseract_version()
        return True
    except Exception:
        return False


needs_tesseract = pytest.mark.skipif(
    not _tesseract_available(), reason="tesseract binary not installed on this host"
)


@pytest.fixture(scope="module")
def export_pixmap(export_pdf):
    doc = fitz.open(stream=export_pdf.read_bytes(), filetype="pdf")
    pix = doc[0].get_pixmap(dpi=200)
    doc.close()
    return pix


def test_preprocess_for_ocr_output_shape_and_dtype(export_pixmap):
    cfg = load_config()
    img = preprocess_for_ocr(export_pixmap, cfg)

    assert img.ndim == 2  # grayscale, single channel
    assert img.dtype == np.uint8
    upscale = float(cfg.get("ingest.ocr_upscale"))
    assert img.shape[0] == pytest.approx(export_pixmap.height * upscale, rel=0.01)
    assert img.shape[1] == pytest.approx(export_pixmap.width * upscale, rel=0.01)
    # near-bitonal (Otsu binarize): overwhelmingly just 2 dominant intensities
    _, counts = np.unique(img, return_counts=True)
    top2_frac = counts[np.argsort(counts)[-2:]].sum() / counts.sum()
    assert top2_frac > 0.95


def test_map_rotated_box_back_round_trips_a_known_point():
    # A point at the exact center of a WxH image maps back to itself regardless of rotation
    # direction -- anchors the coordinate-mapping formula against a case verifiable by hand.
    orig_h, orig_w = 100, 200
    cx, cy = orig_w / 2 - 1, orig_h / 2 - 1
    for direction in ("cw", "ccw"):
        rotated_w, rotated_h = orig_h, orig_w
        rx, ry = rotated_w / 2 - 1, rotated_h / 2 - 1
        ox0, oy0, _, _ = _map_rotated_box_back(rx, ry, 0, 0, orig_h, orig_w, direction)
        assert ox0 == pytest.approx(cx, abs=1.5)
        assert oy0 == pytest.approx(cy, abs=1.5)


@needs_tesseract
@pytest.mark.slow
def test_ocr_words_carry_conf_and_drop_aggregate_rows(export_pixmap):
    cfg = load_config()
    img = preprocess_for_ocr(export_pixmap, cfg)
    words = _ocr_words(img, "tesseract", cfg)

    assert len(words) > 0
    for w in words:
        assert 0.0 <= w["conf"] <= 1.0
        assert w["text"]  # aggregate/blank rows are already filtered out


@needs_tesseract
@pytest.mark.slow
def test_ocr_pixel_bboxes_match_native_ground_truth(repo_root):
    from delta_chat.ingest.pdf_native import parse_native_pdf

    pair_dir = repo_root / "data" / "samples" / "pair_03_scanned"
    scanned_doc = parse_scanned_pdf((pair_dir / "rev_b_scanned.pdf").read_bytes(), "pid:scanned")
    native_doc = parse_native_pdf((pair_dir / "rev_b_native.pdf").read_bytes(), "pid:native")

    def iou(a, b):
        ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
        ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
        iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
        inter = iw * ih
        union = a.area + b.area - inter
        return inter / union if union > 0 else 0.0

    matched = 0
    for nit in native_doc.items:
        candidates = [sit.bbox for sit in scanned_doc.items if sit.text.upper() == nit.text.upper()]
        best = max((iou(nit.bbox, b) for b in candidates), default=0.0)
        if best > 0.3:
            matched += 1
    assert matched >= len(native_doc.items) * 0.5


@needs_tesseract
@pytest.mark.slow
def test_vertical_tag_rotated_repass_recovers_known_tag():
    # End-to-end through parse_scanned_pdf itself (not just the low-level OCR call), so this
    # actually exercises the rotated re-pass + _map_rotated_box_back coordinate mapping + IoU dedup
    # + merge wiring, not just "can Tesseract read horizontal text". A real vertical PDF tag, authored
    # independently of the sample fixtures via fitz's own rotated text insertion, so this doesn't
    # depend on the synthesized fixtures happening to contain a genuine vertical tag.
    doc = fitz.open()
    page = doc.new_page(width=400, height=400)
    page.insert_text((250, 350), "26-FV-9038", fontsize=28, rotate=90)
    pdf_bytes = doc.tobytes()
    doc.close()

    result = parse_scanned_pdf(pdf_bytes, "pid:vertical-tag")

    upright_texts = {it.text.upper() for it in result.items if it.rotation_deg == 0.0}
    rotated_texts = {it.text.upper() for it in result.items if it.rotation_deg in (90.0, 270.0)}

    assert not any("FV" in t and "9038" in t for t in upright_texts)
    assert any("FV" in t and "9038" in t for t in rotated_texts)
    recovered = next(it for it in result.items if "FV" in it.text.upper() and "9038" in it.text.upper())
    assert recovered.rotation_deg in (90.0, 270.0)
    assert 0.0 <= recovered.bbox.x0 <= recovered.bbox.x1 <= 1.0
    assert 0.0 <= recovered.bbox.y0 <= recovered.bbox.y1 <= 1.0


def test_low_mean_conf_sets_degraded_flag(monkeypatch):
    import delta_chat.ingest.pdf_scanned as scanned_mod

    monkeypatch.setattr(
        scanned_mod,
        "_ocr_words",
        lambda img, engine, cfg: [
            {"text": "BLURRY", "left": 1.0, "top": 1.0, "width": 5.0, "height": 5.0, "conf": 0.1}
        ],
    )
    monkeypatch.setattr(
        scanned_mod, "preprocess_for_ocr", lambda pix, cfg=None: np.zeros((50, 50), dtype=np.uint8)
    )

    data = copy.deepcopy(load_config().data)
    data["ingest"]["ocr_vertical_pass"] = False  # isolate: only the upright pass should run
    cfg = Config(data)

    doc = fitz.open()
    doc.new_page(width=200, height=200)
    pdf_bytes = doc.tobytes()
    doc.close()

    result = parse_scanned_pdf(pdf_bytes, "pid:degraded", cfg=cfg)

    assert result.raw_meta["degraded"] is True
    assert result.raw_meta["mean_word_conf"] == pytest.approx(0.1)
    assert len(result.items) == 1  # low-confidence items are surfaced, never silently dropped
