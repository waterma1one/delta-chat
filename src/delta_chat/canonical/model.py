"""Canonical representation — the format-agnostic seam.

Every ingestion adapter (native PDF, scanned PDF, DWG/DXF) emits this ``Document``/``Item`` model,
and every downstream component (delta engine, retrieval, report, markup) consumes only it. No
downstream code knows the source format, so adding a 4th format touches nothing here.

BBox contract: coordinates are page-relative, **top-left origin**, in **normalized [0,1] fractions**
of page width/height — scale-invariant across DPI, A3/A4, and DXF model units. Out-of-range inputs
are **clamped** to [0,1] (chosen over raising so a stray sub-pixel overflow never crashes a pipeline
run); callers that need strictness can assert before constructing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import structlog

_log = structlog.get_logger("delta_chat.canonical")

Region = Literal["title", "revtable", "notes", "drawing", "unknown"]
ItemType = Literal["text", "tag", "line_number", "note", "dimension", "table_cell", "symbol"]
ChangeType = Literal["added", "removed", "modified", "moved", "renumbered"]
SourceSpace = Literal["pdf_points", "pdf_topleft", "image_px", "dxf_wcs"]


def _clamp01(v: float) -> float:
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return float(v)


@dataclass(frozen=True)
class BBox:
    """Page-relative, top-left origin, normalized [0,1] fractions of page w/h. Immutable + hashable."""

    x0: float
    y0: float
    x1: float
    y1: float

    def __post_init__(self) -> None:
        # frozen dataclass -> must use object.__setattr__ to normalize on construction.
        raw = (self.x0, self.y0, self.x1, self.y1)
        clamped = tuple(_clamp01(v) for v in raw)
        if clamped != raw:
            # An out-of-bounds box is a real upstream problem (bad OCR/DXF coords). Surface it
            # rather than silently collapsing/shifting the box under IoU/centroid later.
            _log.warning("bbox.clamped", raw=raw, clamped=clamped)
        object.__setattr__(self, "x0", clamped[0])
        object.__setattr__(self, "y0", clamped[1])
        object.__setattr__(self, "x1", clamped[2])
        object.__setattr__(self, "y1", clamped[3])

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    @property
    def area(self) -> float:
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)

    def as_tuple(self) -> tuple[float, float, float, float]:
        return (self.x0, self.y0, self.x1, self.y1)


@dataclass
class Item:
    id: str                       # stable within a Document (e.g. f"{page}:{region}:{seq}")
    source_rev: Literal["A", "B"]
    type: ItemType
    text: str
    parsed_fields: dict
    bbox: BBox
    page: int                     # physical page index (0-based)
    sheet: str | None             # drawing sheet label (e.g. "3"); may be None
    region: Region
    rotation_deg: float = 0.0      # 0 / 90 / 270 from line dir vector
    layer: str | None = None       # DWG layer / OCG
    conf: float = 1.0              # ingestion confidence (1.0 native/DXF; OCR word conf otherwise)
    source_engine: str = "pymupdf"  # pymupdf | tesseract | doctr | ezdxf


@dataclass
class Document:
    pid: str
    format: Literal["pdf_native", "pdf_scanned", "dwg"]
    page_count: int
    revision_label: str | None
    sheet_size: tuple[float, float]
    items: list[Item] = field(default_factory=list)
    regions: list[tuple[Region, BBox]] = field(default_factory=list)
    raw_meta: dict = field(default_factory=dict)


@dataclass
class Change:
    id: str                       # "CHG-0007"
    type: ChangeType
    item_type: ItemType
    page: int
    sheet: str | None
    bbox: BBox                    # location on rev B (or A for removed)
    description: str
    old: str | None
    new: str | None
    rev_a_ref: str | None         # source Item.id in rev A (provenance for debug/markup/citation)
    rev_b_ref: str | None         # source Item.id in rev B
    match_conf: float             # matched: 1 - assignment cost; added/removed: runner-up margin
    class_conf: float             # parse certainty / embedding margin / LLM self-report
    change_set_id: str            # proximity grouping
    hard_slice: bool = False       # part of the deliberately-authored hard eval slice


@dataclass
class Delta:
    pid_a: str
    pid_b: str
    changes: list[Change] = field(default_factory=list)
    counts: dict = field(default_factory=dict)
    dataset_version: str | None = None


def normalize_bbox(
    raw: tuple[float, float, float, float],
    source_space: SourceSpace,
    page_size: tuple[float, float],
    *,
    dpi: float | None = None,
    insunits: float | None = None,
) -> BBox:
    """Convert native coordinates into the canonical top-left, normalized-[0,1] BBox.

    - ``pdf_points``: PDF user-space points, bottom-left origin -> flip y, divide by page_size (pts).
    - ``image_px``: raster pixels at ``dpi``, top-left origin -> px->pts (x72/dpi) then divide by
      page_size (pts). Round-trips to the same [0,1] as the equivalent native span.
    - ``dxf_wcs``: DXF model-space units (y-up) -> flip y, divide by model extents ``page_size``.

    ``insunits`` is accepted for API symmetry (DXF $INSUNITS) but is not applied here: when
    ``page_size`` is the model-unit extent, dividing by it normalizes unit-agnostically, so no extra
    scaling is required. Raises ValueError on a degenerate (zero) page dimension.
    """
    pw, ph = page_size
    if pw <= 0 or ph <= 0:
        raise ValueError(f"degenerate page_size {page_size!r}: width and height must be > 0")
    x0, y0, x1, y1 = raw

    if source_space == "pdf_points":
        nx0, nx1 = x0 / pw, x1 / pw
        # bottom-left -> top-left: top edge (larger PDF y) maps to smaller normalized y.
        ny0, ny1 = 1.0 - (max(y0, y1) / ph), 1.0 - (min(y0, y1) / ph)
    elif source_space == "pdf_topleft":
        # PyMuPDF page space is already top-left origin in points -> straight divide (no flip). This is
        # the variant native ingestion uses so its boxes align with image_px (also top-left).
        nx0, ny0, nx1, ny1 = x0 / pw, y0 / ph, x1 / pw, y1 / ph
    elif source_space == "image_px":
        if dpi is None or dpi <= 0:
            raise ValueError("image_px requires a positive dpi")
        scale = 72.0 / dpi
        nx0, ny0 = (x0 * scale) / pw, (y0 * scale) / ph
        nx1, ny1 = (x1 * scale) / pw, (y1 * scale) / ph
    elif source_space == "dxf_wcs":
        nx0, nx1 = x0 / pw, x1 / pw
        # DXF is y-up -> flip like pdf_points.
        ny0, ny1 = 1.0 - (max(y0, y1) / ph), 1.0 - (min(y0, y1) / ph)
    else:  # pragma: no cover - guarded by typing
        raise ValueError(f"unknown source_space {source_space!r}")

    lo_x, hi_x = (nx0, nx1) if nx0 <= nx1 else (nx1, nx0)
    lo_y, hi_y = (ny0, ny1) if ny0 <= ny1 else (ny1, ny0)
    return BBox(lo_x, lo_y, hi_x, hi_y)
