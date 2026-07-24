"""Native-PDF adapter — PyMuPDF text + vector + tables -> canonical Document.

Deterministic (no LLM). ``conf=1.0``, ``source_engine="pymupdf"``. Coordinates go through the shared
``normalize_bbox(..., "pdf_topleft", page_size)`` helper (PyMuPDF page space is already top-left origin,
so no y-flip — keeping native boxes aligned with OCR image_px boxes).
"""

from __future__ import annotations

import fitz  # PyMuPDF

from ..canonical.model import BBox, Document, Item, normalize_bbox
from ..config import Config, load_config
from . import regions as _regions
from . import tagging as _tagging
from .base import register_adapter


def _norm(rect, page_size: tuple[float, float]) -> BBox:
    return normalize_bbox((rect[0], rect[1], rect[2], rect[3]), "pdf_topleft", page_size)


def _vertical_line_boxes(page) -> list[tuple["fitz.Rect", float]]:
    """Vertical text line boxes with their rotation (90 or 270 by the sign of the dir vector)."""
    out: list[tuple[fitz.Rect, float]] = []
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        for line in block.get("lines", []):
            dx, dy = line.get("dir", (1.0, 0.0))
            if abs(dy) > 0.5:
                out.append((fitz.Rect(line["bbox"]), 270.0 if dy > 0 else 90.0))
    return out


def parse_native_pdf(data: bytes, pid: str, source_rev: str = "A", cfg: Config | None = None) -> Document:
    cfg = cfg or load_config()
    doc = fitz.open(stream=data, filetype="pdf")
    items: list[Item] = []
    seq: dict[tuple[int, str], int] = {}
    all_regions: list[tuple[str, BBox]] = []
    table_meta: list[dict] = []
    first_size = (doc[0].rect.width, doc[0].rect.height) if doc.page_count else (0.0, 0.0)

    def add_item(page_index, region, text, itype, parsed, bbox, rotation) -> None:
        k = (page_index, region)
        seq[k] = seq.get(k, 0) + 1
        items.append(
            Item(
                id=f"{page_index}:{region}:{seq[k]}",
                source_rev=source_rev,  # type: ignore[arg-type]
                type=itype,  # type: ignore[arg-type]
                text=text,
                parsed_fields=parsed,
                bbox=bbox,
                page=page_index,
                sheet=None,
                region=region,  # type: ignore[arg-type]
                rotation_deg=rotation,
                layer=None,
                conf=1.0,
                source_engine="pymupdf",
            )
        )

    for page_index, page in enumerate(doc):
        page_size = (page.rect.width, page.rect.height)
        norm = lambda r: _norm(r, page_size)  # noqa: E731
        vboxes = _vertical_line_boxes(page)
        cluster = _regions.cluster_drawings(page, norm, cfg)
        title_rect = cluster["title_rect"] or _regions._TITLE_RECT_DEFAULT
        revtable_rect = _regions._REVTABLE_RECT_DEFAULT
        all_regions.append(("title", title_rect))
        all_regions.append(("revtable", revtable_rect))

        # tables -> regions + table_cell items + row metadata
        for region_lbl, tbbox, rows, tobj in _regions.find_tables(page, norm):
            all_regions.append((region_lbl, tbbox))
            table_meta.append({"page": page_index, "bbox": tbbox.as_tuple(), "rows": rows})
            try:
                for r in getattr(tobj, "rows", []):
                    for ci, cell in enumerate(getattr(r, "cells", []) or []):
                        if cell is None:
                            continue
                        crect = fitz.Rect(cell)
                        txt = page.get_textbox(crect).strip()
                        if txt:
                            add_item(page_index, region_lbl, txt, "table_cell", {}, norm(crect), 0.0)
            except Exception:
                pass

        note_marker_centroids: list[tuple[float, float]] = []
        pending: list[tuple[str, str, dict, BBox, float]] = []
        for w in page.get_text("words"):
            x0, y0, x1, y1, word = w[0], w[1], w[2], w[3], w[4]
            if not word.strip():
                continue
            bbox = norm((x0, y0, x1, y1))
            region = _regions.assign_region(bbox, title_rect, revtable_rect)
            cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            rotation = next((rot for vb, rot in vboxes if vb.contains(fitz.Point(cx, cy))), 0.0)
            itype, parsed = "text", {}
            hits = _tagging.extract_tags(word, cfg)
            if hits:
                itype, parsed = hits[0].type, hits[0].parsed_fields
            if _regions.is_note_marker(word):
                region = "notes"
                note_marker_centroids.append(bbox.centroid)
            pending.append((region, word, {"type": itype, "parsed": parsed}, bbox, rotation))

        # second pass: reclassify left-column items sharing a row with a note marker as notes
        for region, word, extra, bbox, rotation in pending:
            cx, cy = bbox.centroid
            if region == "drawing" and cx < 0.5 and any(
                abs(cy - my) < 0.008 for _, my in note_marker_centroids
            ):
                region = "notes"
            add_item(page_index, region, word, extra["type"], extra["parsed"], bbox, rotation)

    page_count = doc.page_count
    meta = {"pdf_metadata": dict(doc.metadata or {}), "tables": table_meta}
    doc.close()
    return Document(
        pid=pid,
        format="pdf_native",
        page_count=page_count,
        revision_label=None,
        sheet_size=first_size,
        items=items,
        regions=all_regions,  # type: ignore[arg-type]
        raw_meta=meta,
    )


class NativePdfAdapter:
    format = "pdf_native"

    def detect(self, data: bytes, filename: str | None) -> bool:
        return data[:5] == b"%PDF-"

    def parse(self, data: bytes, pid: str, source_rev: str = "A") -> Document:
        return parse_native_pdf(data, pid, source_rev=source_rev)


register_adapter(NativePdfAdapter())
