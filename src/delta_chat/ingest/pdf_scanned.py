"""Scanned-PDF adapter — raster + preprocess + Tesseract OCR -> canonical Document.

Plugs into the same ``FormatAdapter`` seam as the native-PDF adapter: the delta engine never
learns whether a page came from a text layer or an OCR pass. Coordinates go through the shared
``normalize_bbox(..., "image_px", page_size, dpi=...)`` helper, using the *effective* DPI of the
preprocessed (upscaled) image, not the raw render DPI.

OCR confidence is a measured, honest failure signal, never a silent drop: low-confidence pages are
flagged (``Document.raw_meta["degraded"]``, and an observability span mark if a request trace is
active) while every recovered word is still kept as an Item.
"""

from __future__ import annotations

import cv2
import fitz  # PyMuPDF
import numpy as np

from ..canonical.model import BBox, Document, Item, normalize_bbox
from ..config import Config, load_config
from ..observability import tracing
from . import regions as _regions
from . import tagging as _tagging
from .base import register_adapter

_ROTATE_CW = "cw"
_ROTATE_CCW = "ccw"


def preprocess_for_ocr(pix: "fitz.Pixmap", cfg: Config | None = None) -> "np.ndarray":
    """grayscale -> upscale (``ingest.ocr_upscale``, default 1.75x) -> Otsu binarize -> deskew.

    Returns a 2-D ``uint8`` ndarray ready for Tesseract.
    """
    cfg = cfg or load_config()
    upscale = float(cfg.get("ingest.ocr_upscale", 1.75))

    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
    gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_RGB2GRAY) if pix.n >= 3 else img[:, :, 0].copy()

    upscaled = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    _, binarized = cv2.threshold(upscaled, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return _deskew(binarized, cfg)


def _deskew(img: "np.ndarray", cfg: Config | None = None) -> "np.ndarray":
    """Estimate skew from the foreground (dark, post-Otsu) pixel cloud via ``minAreaRect`` and
    rotate it out. Uses nearest-neighbor interpolation so the result stays bitonal.

    ``minAreaRect`` runs over the whole foreground pixel cloud -- title-block rules and sheet-border
    lines, not just text -- so a misleading dominant shape could in principle yield a large spurious
    angle. Real-world scan skew is only ~1-3 degrees (see the section risk table), so an estimate past
    ``ingest.ocr_max_deskew_deg`` is treated as noise and skipped rather than applied.
    """
    cfg = cfg or load_config()
    max_deskew = float(cfg.get("ingest.ocr_max_deskew_deg", 5.0))

    coords = np.column_stack(np.where(img < 128))
    if coords.shape[0] < 10:
        return img
    angle = cv2.minAreaRect(coords.astype(np.float32))[-1]
    angle = -(90 + angle) if angle < -45 else -angle
    if abs(angle) < 0.1 or abs(angle) > max_deskew:
        return img
    h, w = img.shape
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
    return cv2.warpAffine(
        img, matrix, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE
    )


def _rotate_image(img: "np.ndarray", direction: str) -> "np.ndarray":
    code = cv2.ROTATE_90_CLOCKWISE if direction == _ROTATE_CW else cv2.ROTATE_90_COUNTERCLOCKWISE
    return cv2.rotate(img, code)


def _map_rotated_box_back(
    left: float, top: float, width: float, height: float, orig_h: int, orig_w: int, direction: str
) -> tuple[float, float, float, float]:
    """Map a pixel bbox detected in a 90deg-rotated copy back to the upright image's pixel frame.

    Derived (and verified empirically against ``cv2.rotate``) as the inverse of ``cv2.rotate``'s
    ``ROTATE_90_CLOCKWISE`` / ``ROTATE_90_COUNTERCLOCKWISE`` transforms, where ``orig_h``/``orig_w``
    are the dimensions of the image *before* this rotation was applied.
    """
    corners = [(left, top), (left + width, top + height)]
    mapped = []
    for x_rot, y_rot in corners:
        if direction == _ROTATE_CW:
            x_orig, y_orig = y_rot, orig_h - 1 - x_rot
        else:
            x_orig, y_orig = orig_w - 1 - y_rot, x_rot
        mapped.append((x_orig, y_orig))
    xs = [p[0] for p in mapped]
    ys = [p[1] for p in mapped]
    return min(xs), min(ys), max(xs), max(ys)


def _iou(a: BBox, b: BBox) -> float:
    ix0, iy0 = max(a.x0, b.x0), max(a.y0, b.y0)
    ix1, iy1 = min(a.x1, b.x1), min(a.y1, b.y1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def _norm_rect(rect, page_size: tuple[float, float]) -> BBox:
    return normalize_bbox((rect[0], rect[1], rect[2], rect[3]), "pdf_topleft", page_size)


def _ocr_words(img: "np.ndarray", engine: str, cfg: Config) -> list[dict]:
    """Run OCR and return surviving word rows as dicts with left/top/width/height/text/conf (0..1).
    Aggregate rows (Tesseract ``conf == -1``) and blank text are dropped here."""
    if engine == "tesseract":
        return _ocr_words_tesseract(img, cfg)
    if engine == "doctr":
        return _ocr_words_doctr(img, cfg)
    raise ValueError(
        f"unknown ingest.ocr_engine {engine!r} (expected 'tesseract' or 'doctr') -- "
        "a typo'd config value must not silently fall back to a different engine"
    )


def _ocr_words_tesseract(img: "np.ndarray", cfg: Config) -> list[dict]:
    import pytesseract

    psm = int(cfg.get("ingest.ocr_psm", 11))
    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT, config=f"--psm {psm}")
    words: list[dict] = []
    for i in range(len(data["text"])):
        conf_raw = float(data["conf"][i])
        text = data["text"][i].strip()
        if conf_raw < 0 or not text:
            continue  # aggregate/blank rows carry conf == -1 -- not real word detections
        words.append(
            {
                "text": text,
                "left": float(data["left"][i]),
                "top": float(data["top"][i]),
                "width": float(data["width"][i]),
                "height": float(data["height"][i]),
                "conf": min(1.0, max(0.0, conf_raw / 100.0)),
            }
        )
    return words


def _ocr_words_doctr(img: "np.ndarray", cfg: Config) -> list[dict]:  # pragma: no cover - optional dep
    """docTR path (``ingest.ocr_engine: doctr``, higher accuracy, opt-in — requires the optional
    ``python-doctr`` dependency, kept out of the default install so the critical path stays light)."""
    try:
        from doctr.models import ocr_predictor
    except ImportError as exc:
        raise ImportError(
            "ingest.ocr_engine is 'doctr' but the optional 'python-doctr' dependency is not "
            "installed. Install it with `uv pip install delta-chat[ocr]`, or set "
            "ingest.ocr_engine back to 'tesseract' (the zero-download default)."
        ) from exc

    predictor = ocr_predictor(pretrained=True)
    rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    result = predictor([rgb])
    h, w = img.shape
    words: list[dict] = []
    for page in result.pages:
        for block in page.blocks:
            for line in block.lines:
                for word in line.words:
                    (x0, y0), (x1, y1) = word.geometry
                    words.append(
                        {
                            "text": word.value,
                            "left": x0 * w,
                            "top": y0 * h,
                            "width": (x1 - x0) * w,
                            "height": (y1 - y0) * h,
                            "conf": float(word.confidence),
                        }
                    )
    return words


def parse_scanned_pdf(data: bytes, pid: str, source_rev: str = "A", cfg: Config | None = None) -> Document:
    cfg = cfg or load_config()
    dpi = float(cfg.get("ingest.dpi", 250))
    upscale = float(cfg.get("ingest.ocr_upscale", 1.75))
    degraded_threshold = float(cfg.get("ingest.ocr_degraded_conf_threshold", 0.5))
    vertical_pass = bool(cfg.get("ingest.ocr_vertical_pass", True))
    overlap_threshold = float(cfg.get("ingest.ocr_dedup_iou", 0.2))
    engine = cfg.get("ingest.ocr_engine", "tesseract")

    doc = fitz.open(stream=data, filetype="pdf")
    items: list[Item] = []
    all_regions: list[tuple[str, BBox]] = []
    all_confs: list[float] = []
    page_mean_confs: list[float] = []
    first_size = (doc[0].rect.width, doc[0].rect.height) if doc.page_count else (0.0, 0.0)

    for page_index, page in enumerate(doc):
        page_size = (page.rect.width, page.rect.height)
        pix = page.get_pixmap(dpi=int(dpi))
        pre = preprocess_for_ocr(pix, cfg)
        effective_dpi = dpi * upscale
        orig_h, orig_w = pre.shape

        title_rect = _regions._TITLE_RECT_DEFAULT
        revtable_rect = _regions._REVTABLE_RECT_DEFAULT
        # cluster_drawings already guards its own PyMuPDF calls internally and returns a
        # title_rect of None when a raster page carries no vector drawings -- no outer try/except
        # here, so a genuine bug surfaces instead of vanishing silently.
        cluster = _regions.cluster_drawings(page, lambda r: _norm_rect(r, page_size), cfg)
        if cluster.get("title_rect") is not None:
            title_rect = cluster["title_rect"]
        all_regions.append(("title", title_rect))
        all_regions.append(("revtable", revtable_rect))

        seq = 0
        placed: list[BBox] = []
        page_confs: list[float] = []

        def add_word(word: dict, rotation: float) -> None:
            nonlocal seq
            bbox = normalize_bbox(
                (word["left"], word["top"], word["left"] + word["width"], word["top"] + word["height"]),
                "image_px",
                page_size,
                dpi=effective_dpi,
            )
            region = _regions.assign_region(bbox, title_rect, revtable_rect)
            hits = _tagging.extract_tags(word["text"], cfg)
            itype, parsed = (hits[0].type, hits[0].parsed_fields) if hits else ("text", {})
            seq += 1
            items.append(
                Item(
                    id=f"{page_index}:{region}:{seq}",
                    source_rev=source_rev,  # type: ignore[arg-type]
                    type=itype,  # type: ignore[arg-type]
                    text=word["text"],
                    parsed_fields=parsed,
                    bbox=bbox,
                    page=page_index,
                    sheet=None,
                    region=region,  # type: ignore[arg-type]
                    rotation_deg=rotation,
                    layer=None,
                    conf=word["conf"],
                    source_engine=engine,
                )
            )
            placed.append(bbox)
            page_confs.append(word["conf"])

        for w in _ocr_words(pre, engine, cfg):
            add_word(w, 0.0)

        if vertical_pass:
            for direction, rotation_deg in ((_ROTATE_CW, 90.0), (_ROTATE_CCW, 270.0)):
                rotated = _rotate_image(pre, direction)
                for w in _ocr_words(rotated, engine, cfg):
                    ox0, oy0, ox1, oy1 = _map_rotated_box_back(
                        w["left"], w["top"], w["width"], w["height"], orig_h, orig_w, direction
                    )
                    bbox = normalize_bbox((ox0, oy0, ox1, oy1), "image_px", page_size, dpi=effective_dpi)
                    if any(_iou(bbox, existing) > overlap_threshold for existing in placed):
                        continue  # already recovered by the upright (or the other rotated) pass
                    add_word(
                        {"text": w["text"], "left": ox0, "top": oy0, "width": ox1 - ox0, "height": oy1 - oy0,
                         "conf": w["conf"]},
                        rotation_deg,
                    )

        # Per-page, not just overall: a single badly-degraded page must not get averaged out by
        # otherwise-clean pages and silently escape the failure signal.
        page_mean = sum(page_confs) / len(page_confs) if page_confs else 0.0
        page_mean_confs.append(page_mean)
        all_confs.extend(page_confs)
        if page_mean < degraded_threshold:
            tracing.mark_degraded("low_ocr_confidence", page=page_index, ocr_mean_word_conf=page_mean)

    page_count = doc.page_count
    doc.close()

    mean_conf = sum(all_confs) / len(all_confs) if all_confs else 0.0
    degraded_pages = [i for i, m in enumerate(page_mean_confs) if m < degraded_threshold]
    degraded = bool(degraded_pages)

    return Document(
        pid=pid,
        format="pdf_scanned",
        page_count=page_count,
        revision_label=None,
        sheet_size=first_size,
        items=items,
        regions=all_regions,  # type: ignore[arg-type]
        raw_meta={
            "degraded": degraded,
            "mean_word_conf": mean_conf,
            "ocr_engine": engine,
            "page_mean_word_conf": page_mean_confs,
            "degraded_pages": degraded_pages,
        },
    )


class ScannedPdfAdapter:
    format = "pdf_scanned"

    def detect(self, data: bytes, filename: str | None) -> bool:
        return data[:5] == b"%PDF-"

    def parse(self, data: bytes, pid: str, source_rev: str = "A") -> Document:
        return parse_scanned_pdf(data, pid, source_rev=source_rev)


register_adapter(ScannedPdfAdapter())
