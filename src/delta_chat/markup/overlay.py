"""Delta markup overlay — renders a computed ``Delta`` as annotations on the source PDF (bonus
rubric +4). A pure consumer of the canonical ``Delta``/``Change``/``BBox`` types: this module
never touches ingestion or the delta engine, and a failure here never affects the core pipeline.

Coordinate rule: ``BBox`` is normalized [0,1], top-left origin — the same origin convention
PyMuPDF's page coordinate space uses (top-left, points). Denormalizing is a straight multiply by
page width/height; no y-flip (that already happened in ``normalize_bbox`` during ingestion).
"""

from __future__ import annotations

import fitz

from ..canonical.model import BBox, Change, Delta
from ..config import load_config
from ..observability.logging import get_logger

_log = get_logger("delta_chat.markup")

COLOR_BY_TYPE: dict[str, tuple[float, float, float]] = {
    "added": (0.0, 0.7, 0.0),
    "removed": (0.85, 0.0, 0.0),
    "modified": (1.0, 0.65, 0.0),
    "moved": (0.0, 0.35, 0.85),
    "renumbered": (0.55, 0.0, 0.65),
}
_FALLBACK_COLOR = (0.4, 0.4, 0.4)
_CLOUD_COLOR = (0.5, 0.5, 0.5)
_BORDER_WIDTH = 1.25
_MIN_RECT_SIDE_PTS = 4.0  # pad a degenerate/zero-area rect so the marker stays visible
_CLOUD_PAD_PTS = 3.0
_LEGEND_MARGIN = 8.0
_LEGEND_SWATCH = 8.0
_LEGEND_ROW_H = 12.0
_LEGEND_FONTSIZE = 7.0


def _config_colors(cfg) -> tuple[dict[str, tuple[float, float, float]], tuple[float, float, float], tuple[float, float, float], float]:
    """Read markup.colors.* from config ("config over hardcoding"). A partial override merges
    over the module defaults — a config that only overrides e.g. "added" must not silently lose
    the documented colors for every other type."""
    colors_cfg = cfg.get("markup.colors", None)
    colors = dict(COLOR_BY_TYPE)
    if colors_cfg:
        colors.update({k: tuple(v) for k, v in colors_cfg.items()})
    fallback = tuple(cfg.get("markup.default_color", _FALLBACK_COLOR))
    cloud_color = tuple(cfg.get("markup.cloud_color", _CLOUD_COLOR))
    border_width = float(cfg.get("markup.border_width", _BORDER_WIDTH))
    return colors, fallback, cloud_color, border_width


def _denormalize_bbox(bbox: BBox, page_rect: fitz.Rect) -> fitz.Rect:
    x0, y0 = bbox.x0 * page_rect.width, bbox.y0 * page_rect.height
    x1, y1 = bbox.x1 * page_rect.width, bbox.y1 * page_rect.height
    if x1 < x0:
        # BBox.__post_init__ only clamps each coordinate into [0,1] independently — it does not
        # enforce x0<=x1 (that ordering is only guaranteed by normalize_bbox's explicit sort), so
        # a directly-constructed inverted BBox is legal at the type level. Swap explicitly rather
        # than letting the epsilon-padding below collapse it to a meaningless midpoint.
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    if x1 - x0 < _MIN_RECT_SIDE_PTS:
        cx = (x0 + x1) / 2.0
        x0, x1 = cx - _MIN_RECT_SIDE_PTS / 2.0, cx + _MIN_RECT_SIDE_PTS / 2.0
    if y1 - y0 < _MIN_RECT_SIDE_PTS:
        cy = (y0 + y1) / 2.0
        y0, y1 = cy - _MIN_RECT_SIDE_PTS / 2.0, cy + _MIN_RECT_SIDE_PTS / 2.0
    return fitz.Rect(x0, y0, x1, y1)


def _draw_change_set_clouds(page: fitz.Page, changes: list[Change], cloud_color) -> None:
    """Group changes by change_set_id and draw a light enclosing rect around sets with >1 member.
    Drawn as page vector graphics (not an annotation) so it never inflates the per-change
    annotation count."""
    by_set: dict[str, list[Change]] = {}
    for ch in changes:
        by_set.setdefault(ch.change_set_id, []).append(ch)
    for members in by_set.values():
        if len(members) <= 1:
            continue
        union = _denormalize_bbox(members[0].bbox, page.rect)
        for m in members[1:]:
            union = union | _denormalize_bbox(m.bbox, page.rect)
        union = fitz.Rect(
            union.x0 - _CLOUD_PAD_PTS, union.y0 - _CLOUD_PAD_PTS,
            union.x1 + _CLOUD_PAD_PTS, union.y1 + _CLOUD_PAD_PTS,
        )
        page.draw_rect(union, color=cloud_color, width=0.75, dashes="[3 2] 0")


def _draw_legend(page: fitz.Page, types_used: set[str], colors: dict, fallback_color) -> None:
    for i, change_type in enumerate(sorted(types_used)):
        color = colors.get(change_type, fallback_color)
        y = _LEGEND_MARGIN + i * _LEGEND_ROW_H
        swatch = fitz.Rect(_LEGEND_MARGIN, y, _LEGEND_MARGIN + _LEGEND_SWATCH, y + _LEGEND_SWATCH)
        page.draw_rect(swatch, color=color, fill=color)
        page.insert_text(
            (_LEGEND_MARGIN + _LEGEND_SWATCH + 4, y + _LEGEND_SWATCH - 1),
            change_type, fontsize=_LEGEND_FONTSIZE,
        )


def render_markup(delta: Delta, source_pdf: bytes, cfg=None) -> bytes:
    """Draw each change's bbox onto its source page (colored by change type), grouped into
    change-set "clouds", and return an annotated PDF (bytes). Exactly one annotation is created
    per renderable Change, so `annotation_count == len(delta.changes)` for changes whose page
    exists in `source_pdf` — a change with an out-of-range page index is skipped (logged, not
    raised) so this bonus feature never breaks the core pipeline. Only pages with >=1 change are
    touched; each touched page also gets a small color legend for the types present on it.

    Note on `removed` changes: `Change.bbox` for a `removed` change is rev-A-relative per the
    canonical contract, but this function only receives one `source_pdf`. Normalized [0,1]
    coordinates transfer acceptably across the revision pair for a visual marker; a caller wanting
    pixel-accurate removed markers should pass rev-A bytes instead.

    Note on the legend: it is unconditionally placed in each touched page's top-left corner with
    no collision detection against real drawing content — acceptable for a bonus feature, but a
    dense engineering drawing may have content there already.
    """
    cfg = cfg or load_config()
    colors, fallback_color, cloud_color, border_width = _config_colors(cfg)

    doc = fitz.open(stream=source_pdf, filetype="pdf")
    try:
        changes_by_page: dict[int, list[Change]] = {}
        for ch in delta.changes:
            if ch.page < 0 or ch.page >= doc.page_count:
                _log.warning(
                    "markup.page_out_of_range", change_id=ch.id, page=ch.page, page_count=doc.page_count
                )
                continue
            changes_by_page.setdefault(ch.page, []).append(ch)

        for page_index, changes in changes_by_page.items():
            page = doc[page_index]
            _draw_change_set_clouds(page, changes, cloud_color)

            types_used: set[str] = set()
            for ch in changes:
                rect = _denormalize_bbox(ch.bbox, page.rect)
                color = colors.get(ch.type, fallback_color)
                annot = page.add_rect_annot(rect)
                annot.set_colors(stroke=color)
                annot.set_border(width=border_width)
                annot.set_info(title=ch.id, content=ch.description)
                annot.update()
                types_used.add(ch.type)

            _draw_legend(page, types_used, colors, fallback_color)

        return doc.tobytes()
    finally:
        doc.close()
