"""Ingestion seam: the FormatAdapter interface, format detection, and a routing registry.

Every adapter normalizes raw bytes into the canonical ``Document`` (typed ``Item`` list, normalized
bboxes). Detection decides native-PDF vs scanned-PDF by probing for an extractable text layer, and DWG
by magic/extension. Unknown inputs raise a clear, traceable error (never a silent default).
"""

from __future__ import annotations

from typing import Protocol

from ..canonical.model import Document
from ..config import Config, load_config
from ..observability.logging import get_logger

_log = get_logger("delta_chat.ingest")


class UnknownFormatError(ValueError):
    """Raised when no adapter can handle the input; message names the probe reason."""


class FormatAdapter(Protocol):
    format: str

    def detect(self, data: bytes, filename: str | None) -> bool: ...

    def parse(self, data: bytes, pid: str, source_rev: str = "A") -> Document: ...


# --- format probes -------------------------------------------------------------------------------

def _looks_like_pdf(data: bytes) -> bool:
    return data[:5] == b"%PDF-"


def _looks_like_dwg(data: bytes, filename: str | None) -> bool:
    if filename and filename.lower().endswith((".dwg", ".dxf")):
        return True
    # DWG version signature "AC10xx"; DXF ascii starts with a group code + SECTION.
    return data[:4] == b"AC10" or data[:22].lstrip().startswith(b"0\nSECTION") or data[:10].find(b"SECTION") != -1


def native_word_count(data: bytes) -> int:
    """Count extractable words across pages (0 for a scan/image-only PDF)."""
    import fitz  # local import: keeps the seam importable without PyMuPDF at type-check time

    total = 0
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            total += len(page.get_text("words"))
    return total


def max_image_area_frac(data: bytes) -> float:
    """Largest fraction of any page's area covered by a single raster image.

    ~1.0 means a page is a scanned photo/scan; ~0.0 means a born-digital vector page. This is the
    signal that separates "a drawing with little text" from "a scan", which a word count cannot:
    a P&ID is mostly geometry and carries very few words, while a scan re-imported with its OCR
    layer carries many. Only the raster tells you which pipeline the page actually needs.
    """
    import fitz

    frac = 0.0
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            page_area = abs(page.rect.get_area())
            if page_area <= 0:
                continue
            for img in page.get_images(full=True):
                for rect in page.get_image_rects(img[0]):
                    frac = max(frac, abs(rect.get_area()) / page_area)
    return frac


def vector_drawing_count(data: bytes) -> int:
    """Total vector drawing paths across pages (lines, rects, curves from ``page.get_drawings()``).

    A born-digital drawing carries real vector geometry; a scan is a flat raster with none. This is
    what lets a native page keep its text layer even when it *also* has a full-page background
    raster (an underlay, watermark, or scanned title-block): image-dominance alone would misroute
    such a page to OCR and throw away its exact text.
    """
    import fitz

    total = 0
    with fitz.open(stream=data, filetype="pdf") as doc:
        for page in doc:
            total += len(page.get_drawings())
    return total


# --- registry ------------------------------------------------------------------------------------

# adapter registry keyed by format name; populated by the adapters at import time via register().
_ADAPTERS: dict[str, FormatAdapter] = {}


def register_adapter(adapter: FormatAdapter) -> None:
    _ADAPTERS[adapter.format] = adapter


def get_adapter(fmt: str) -> FormatAdapter:
    if fmt not in _ADAPTERS:
        raise UnknownFormatError(f"no adapter registered for format {fmt!r}")
    return _ADAPTERS[fmt]


def detect_format(data: bytes, filename: str | None, cfg: Config | None = None) -> str:
    """Return the format name ('pdf_native' | 'pdf_scanned' | 'dwg'); raise on unknown.

    DWG routing is delegated to the registered ``dwg`` adapter's ``detect()`` (so a real 4th format
    plugs in behind the seam). PDF native-vs-scanned needs the text-layer probe, which lives here
    because both PDF adapters answer ``detect()`` True on a PDF. If the filename extension and the
    magic bytes disagree, we warn and trust the bytes.
    """
    cfg = cfg or load_config()
    is_pdf = _looks_like_pdf(data)
    dwg_adapter = _ADAPTERS.get("dwg")
    dwg_by_ext = bool(filename and filename.lower().endswith((".dwg", ".dxf")))
    if dwg_by_ext and is_pdf:
        _log.warning("ingest.ext_magic_mismatch", filename=filename, note="extension says DWG, bytes say PDF; trusting bytes")
    elif dwg_adapter is not None and dwg_adapter.detect(data, filename):
        return "dwg"
    if is_pdf:
        # Native needs a real text layer (>= min_words). Beyond that, image-dominance alone is not
        # enough to call something a scan: a born-digital drawing can legitimately carry a full-page
        # background raster (underlay/watermark/scanned title-block) while still having a perfect
        # vector+text layer. So a raster-dominated page is only treated as scanned when it ALSO has
        # no vector geometry — the combination that actually characterises a flat scan. Vector paths
        # present => keep it native and use its real text instead of re-OCR'ing it.
        min_words = int(cfg.get("ingest.native_min_words", 1))
        img_frac_max = float(cfg.get("ingest.scanned_image_area_frac", 0.80))
        min_vectors = int(cfg.get("ingest.native_min_vectors", 1))
        try:
            words = native_word_count(data)
            img_frac = max_image_area_frac(data)
            vectors = vector_drawing_count(data)
        except Exception as exc:  # corrupt/unreadable PDF -> traceable error
            raise UnknownFormatError(f"unreadable PDF ({exc})") from exc
        if words < min_words:
            return "pdf_scanned"
        raster_dominates = img_frac >= img_frac_max
        if raster_dominates and vectors < min_vectors:
            return "pdf_scanned"  # flat raster, no geometry, text is a baked-in OCR layer
        return "pdf_native"
    raise UnknownFormatError(
        f"unrecognized format (filename={filename!r}, magic={data[:8]!r})"
    )


def select_adapter(data: bytes, filename: str | None = None, cfg: Config | None = None) -> FormatAdapter:
    return get_adapter(detect_format(data, filename, cfg))


def parse(data: bytes, pid: str, filename: str | None = None, *, source_rev: str = "A",
          cfg: Config | None = None) -> Document:
    """Detect + route + parse in one call."""
    adapter = select_adapter(data, filename, cfg)
    return adapter.parse(data, pid, source_rev=source_rev)


def _register_builtin_adapters() -> None:
    """Import + register the concrete adapters. Scanned/DWG parse() are stubs until §06."""
    from . import pdf_native  # noqa: F401  (registers on import)

    try:
        from . import pdf_scanned  # noqa: F401
    except Exception:
        pass
    try:
        from . import dwg  # noqa: F401
    except Exception:
        pass


_register_builtin_adapters()
