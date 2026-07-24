"""DWG/DXF adapter — detection registered here; parse() implemented in section-06."""

from __future__ import annotations

from ..canonical.model import Document
from .base import register_adapter


class DwgAdapter:
    format = "dwg"

    def detect(self, data: bytes, filename: str | None) -> bool:
        if filename and filename.lower().endswith((".dwg", ".dxf")):
            return True
        return data[:4] == b"AC10" or b"SECTION" in data[:64]

    def parse(self, data: bytes, pid: str, source_rev: str = "A") -> Document:  # pragma: no cover
        raise NotImplementedError("DWG/DXF ingestion is implemented in section-06")


register_adapter(DwgAdapter())
