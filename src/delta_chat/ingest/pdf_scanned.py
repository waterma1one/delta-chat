"""Scanned-PDF adapter — detection registered here; parse() implemented in section-06."""

from __future__ import annotations

from ..canonical.model import Document
from .base import register_adapter


class ScannedPdfAdapter:
    format = "pdf_scanned"

    def detect(self, data: bytes, filename: str | None) -> bool:
        return data[:5] == b"%PDF-"

    def parse(self, data: bytes, pid: str, source_rev: str = "A") -> Document:  # pragma: no cover
        raise NotImplementedError("scanned-PDF ingestion is implemented in section-06")


register_adapter(ScannedPdfAdapter())
