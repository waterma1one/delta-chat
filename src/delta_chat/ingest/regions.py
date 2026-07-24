"""Sheet region segmentation (title / revtable / notes / drawing).

Best-effort: engineering sheets are mostly ruled boxes. We derive the title-block region from PyMuPDF
vector clusters when possible (filtering degenerate paths via ``ingest.vector_min_area``), fall back to a
corner heuristic, detect the note-list from numbered markers, and read tables via ``find_tables``. Region
assignment for an item is a point-in-rect test on its normalized bbox centre; unassigned -> ``drawing``.
"""

from __future__ import annotations

import re

from ..canonical.model import BBox, Region

# Corner fallbacks (normalized, top-left). Used when cluster detection yields nothing.
_TITLE_RECT_DEFAULT = BBox(0.62, 0.72, 1.0, 1.0)
_REVTABLE_RECT_DEFAULT = BBox(0.62, 0.0, 1.0, 0.18)

_NOTE_MARKER = re.compile(r"^\d{1,2}\.$")  # "6.", "12."


def _contains(rect: BBox, cx: float, cy: float) -> bool:
    return rect.x0 <= cx <= rect.x1 and rect.y0 <= cy <= rect.y1


def cluster_drawings(page, normalize, cfg) -> dict:
    """Filter degenerate vector paths (``ingest.vector_min_area``) and derive a title-block rect from the
    largest bottom-right vector cluster. Returns {title_rect, filtered_paths, kept_paths}."""
    min_area = float(cfg.get("ingest.vector_min_area", 1.0))
    try:
        drawings = page.get_drawings()
    except Exception:
        drawings = []
    kept = [d for d in drawings if abs(d["rect"].width) * abs(d["rect"].height) >= min_area]

    title_rect = None
    try:
        clusters = page.cluster_drawings()
    except Exception:
        clusters = []
    w, h = page.rect.width, page.rect.height
    cands = [c for c in clusters if (c.x0 + c.x1) / 2 > w * 0.55 and (c.y0 + c.y1) / 2 > h * 0.6]
    if cands:
        best = max(cands, key=lambda c: c.width * c.height)
        title_rect = normalize(best)
    return {"title_rect": title_rect, "filtered_paths": len(drawings) - len(kept), "kept_paths": len(kept)}


def assign_region(bbox: BBox, title_rect: BBox, revtable_rect: BBox) -> Region:
    cx, cy = bbox.centroid
    if _contains(revtable_rect, cx, cy):
        return "revtable"
    if _contains(title_rect, cx, cy):
        return "title"
    return "drawing"


def is_note_marker(text: str) -> bool:
    return bool(_NOTE_MARKER.match(text.strip()))


def find_tables(page, normalize) -> list[tuple[Region, BBox, list, object]]:
    """Best-effort table detection. Returns (region, normalized bbox, rows, table_obj) per table so the
    classifier (§05) can read revision-table rows deterministically."""
    out: list[tuple[Region, BBox, list, object]] = []
    try:
        tf = page.find_tables(strategy="lines")
        for t in getattr(tf, "tables", []) or []:
            try:
                rows = t.extract()
            except Exception:
                rows = []
            out.append(("revtable", normalize(t.bbox), rows, t))
    except Exception:
        pass
    return out
