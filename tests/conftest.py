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


@pytest.fixture
def trace_cfg(tmp_path, monkeypatch):
    """Config pointed at a temp trace dir with OTLP off — reusable by §07/§08/§10 tracing tests."""
    from delta_chat.config import load_config

    monkeypatch.setenv("DELTA_CHAT__OBS__TRACE_DIR", str(tmp_path / "traces"))
    monkeypatch.setenv("DELTA_CHAT__OBS__OTLP_ENABLED", "false")
    return load_config()

