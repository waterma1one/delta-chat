"""PID resolver — the seam from a persistent document identifier to raw bytes + metadata.

Minimal by design: supports ``file://`` URIs, bare filesystem paths, and a small in-memory registry
for synthesized sample PIDs. Richer resolution (object stores, a DB) is out of scope for the
critical path.

Security: bare/``file://`` paths must resolve **inside an allowed root** (defaults to the repo root)
so a client-supplied PID at the API seam cannot read arbitrary host files. Registered sample
PIDs bypass the check (they are set by trusted synthesis code). A deployment can narrow/widen the
allowlist via :func:`configure`.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import unquote, urlparse

# In-memory registry: pid -> filesystem path. Populated by sample synthesis / tests.
_REGISTRY: dict[str, str] = {}

# Default allowed root = repo root (src/delta_chat/pids/resolver.py -> parents[3]).
_DEFAULT_ROOTS: list[Path] = [Path(__file__).resolve().parents[3]]
_allowed_roots: list[Path] = list(_DEFAULT_ROOTS)


class PidResolutionError(ValueError):
    """Raised when a PID cannot be resolved to bytes (unknown, unreadable, or outside allowed roots)."""


def configure(allowed_roots: list[str] | None = None) -> None:
    """Set the allowlist of directories that filesystem PIDs may resolve inside.

    Pass ``None`` to reset to the default (repo root). Paths are resolved to absolutes.
    """
    global _allowed_roots
    if allowed_roots is None:
        _allowed_roots = list(_DEFAULT_ROOTS)
    else:
        _allowed_roots = [Path(r).resolve() for r in allowed_roots]


def register(pid: str, path: str) -> None:
    """Register a sample PID to a local file path (trusted; bypasses the allowlist)."""
    _REGISTRY[pid] = path


def _to_path(pid: str) -> tuple[Path, bool]:
    """Return (path, is_registered). Registered PIDs are trusted and skip the allowlist check."""
    if pid in _REGISTRY:
        return Path(_REGISTRY[pid]), True
    if pid.startswith("file://"):
        return Path(unquote(urlparse(pid).path)), False
    return Path(pid), False


def _within_allowed(path: Path) -> bool:
    rp = path.resolve()
    return any(rp == root or root in rp.parents for root in _allowed_roots)


def resolve(pid: str) -> tuple[bytes, dict]:
    """Resolve a PID to ``(raw_bytes, metadata)``.

    Supports ``file://`` URIs, bare paths, and registered sample PIDs. Raises
    :class:`PidResolutionError` on an unknown/unreadable PID, or one that resolves outside the
    allowed roots (unless it was registered).
    """
    path, registered = _to_path(pid)
    if not registered and not _within_allowed(path):
        raise PidResolutionError(
            f"refusing to resolve PID {pid!r}: {path} is outside the allowed roots "
            f"{[str(r) for r in _allowed_roots]}"
        )
    if not path.exists() or not path.is_file():
        raise PidResolutionError(f"cannot resolve PID {pid!r}: no file at {path}")
    data = path.read_bytes()
    meta = {
        "pid": pid,
        "path": str(path),
        "filename": path.name,
        "size_bytes": len(data),
        "suffix": path.suffix.lower(),
    }
    return data, meta
