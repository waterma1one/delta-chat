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
        threshold = int(cfg.get("ingest.native_text_word_threshold", 20))
        try:
            words = native_word_count(data)
        except Exception as exc:  # corrupt/unreadable PDF -> traceable error
            raise UnknownFormatError(f"unreadable PDF ({exc})") from exc
        return "pdf_native" if words > threshold else "pdf_scanned"
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
