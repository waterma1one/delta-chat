"""PID resolver: happy path, registry bypass, and the allowed-root sandbox."""

from __future__ import annotations

import pytest

from delta_chat.pids import resolver


def test_resolves_registered_and_file_uri(tmp_path, monkeypatch):
    f = tmp_path / "doc.pdf"
    f.write_bytes(b"%PDF-1.7 hi")
    monkeypatch.setattr(resolver, "_allowed_roots", [tmp_path])
    # file:// inside allowed root
    data, meta = resolver.resolve(f"file://{f}")
    assert data == b"%PDF-1.7 hi" and meta["filename"] == "doc.pdf"
    # registered PID bypasses the allowlist even if outside roots
    resolver.configure(allowed_roots=["/nonexistent-root"])
    resolver.register("pid:sample", str(f))
    data2, _ = resolver.resolve("pid:sample")
    assert data2 == b"%PDF-1.7 hi"
    resolver.configure(None)


def test_rejects_path_outside_allowed_roots(tmp_path, monkeypatch):
    outside = tmp_path / "secret.txt"
    outside.write_text("classified")
    monkeypatch.setattr(resolver, "_allowed_roots", [tmp_path / "allowed"])
    (tmp_path / "allowed").mkdir()
    with pytest.raises(resolver.PidResolutionError):
        resolver.resolve(str(outside))


def test_unknown_pid_raises(tmp_path, monkeypatch):
    monkeypatch.setattr(resolver, "_allowed_roots", [tmp_path])
    with pytest.raises(resolver.PidResolutionError):
        resolver.resolve(str(tmp_path / "missing.pdf"))
