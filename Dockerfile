FROM python:3.11-slim

# System deps: Tesseract for scanned-PDF OCR. No poppler — PyMuPDF covers raster/extract.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv (matches the host toolchain / pyproject)
COPY --from=ghcr.io/astral-sh/uv:0.11.7 /uv /usr/local/bin/uv

WORKDIR /app
# Full source is copied before `uv sync` (rather than pyproject.toml first, for layer-cache reuse)
# because the local `delta-chat` package builds/installs as part of this sync and its source must
# already be present — trades away dependency-layer caching on every source change for correctness.
COPY . .
RUN uv sync --frozen --no-dev || uv sync

# Belt-and-suspenders: uv's editable install of the local package has proven order-dependent flaky
# in some environments (see Makefile's RUN comment) — PYTHONPATH makes the CLI/eval entrypoints
# resolve `delta_chat`/`eval` deterministically regardless of that install's state.
ENV PYTHONPATH=/app/src

# data/samples/**/*.pdf are gitignored (regenerable, not committed — see .gitignore) — bake them
# into the image so the app is immediately usable without a first-run synthesis step.
RUN uv run python scripts/synth_samples.py

# The image ships without a DWG converter (ODA File Converter is not bundled). The DWG
# adapter returns its "conversion required" stub for .dwg input; .dxf inputs parse normally.

EXPOSE 8000
CMD ["uv", "run", "uvicorn", "delta_chat.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
