"""Global pytest fixtures shared across the suite."""

from __future__ import annotations

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_PDF_DIR = REPO_ROOT / "data" / "samples" / "source"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def source_pdfs() -> list[Path]:
    """The two provided native P&ID PDFs (fixtures for native-PDF ingestion tests)."""
    return sorted(SOURCE_PDF_DIR.glob("*.pdf"))


@pytest.fixture(scope="session")
def export_pdf() -> Path:
    p = SOURCE_PDF_DIR / "export-gas-compressor.pdf"
    if not p.exists():  # pragma: no cover - environment guard
        pytest.skip(f"sample PDF missing: {p}")
    return p
