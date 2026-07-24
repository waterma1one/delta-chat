"""Shared fixtures for the FastAPI serving-layer test suite."""

from __future__ import annotations

import copy
import uuid
from pathlib import Path

import pytest

from delta_chat.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[2]
PAIR_01_DIR = REPO_ROOT / "data" / "samples" / "pair_01_native"


@pytest.fixture
def app_cfg(tmp_path):
    """Default config pointed at a temp trace dir, OTLP off, entailment check off (deterministic
    without needing to mock a second LLM call for every /chat test)."""
    data = copy.deepcopy(load_config().data)
    data["obs"]["trace_dir"] = str(tmp_path / "traces")
    data["obs"]["otlp_enabled"] = False
    data["chat"]["sufficiency"]["use_entailment_check"] = False
    # delta.notes.semantic_low/high are calibrated in section-10 and null by default; the delta
    # engine's own test suite (tests/delta/conftest.py) fills concrete values for the same reason.
    data["delta"]["notes"]["semantic_low"] = 0.35
    data["delta"]["notes"]["semantic_high"] = 0.85
    return Config(data)


@pytest.fixture
def registered_pair():
    """Register the golden native sample pair (section-04) under fresh, test-unique pids so
    parallel/repeated test runs never collide on the resolver's shared registry."""
    from delta_chat.pids import resolver

    if not PAIR_01_DIR.exists():  # pragma: no cover - environment guard
        pytest.skip(f"sample pair missing: {PAIR_01_DIR}")
    suffix = uuid.uuid4().hex[:8]
    pid_a, pid_b = f"test-pair01-a-{suffix}", f"test-pair01-b-{suffix}"
    resolver.register(pid_a, str(PAIR_01_DIR / "rev_a.pdf"))
    resolver.register(pid_b, str(PAIR_01_DIR / "rev_b.pdf"))
    return pid_a, pid_b


@pytest.fixture
def client(app_cfg):
    from fastapi.testclient import TestClient

    from delta_chat.api.app import create_app

    return TestClient(create_app(cfg=app_cfg))


class _MockAnswer:
    """Queue canned `ChatAnswer`s for `delta_chat.api.app.answer`; records each call's args."""

    def __init__(self):
        self.calls: list[dict] = []
        self._queue: list[object] = []

    def queue(self, result) -> None:
        self._queue.append(result)

    def __call__(self, query, index, cfg):
        self.calls.append({"query": query, "index": index, "cfg": cfg})
        if not self._queue:
            raise AssertionError("mock_answer: no queued ChatAnswer for this call")
        return self._queue.pop(0)


@pytest.fixture
def mock_answer(monkeypatch):
    mock = _MockAnswer()
    monkeypatch.setattr("delta_chat.api.app.answer", mock)
    return mock
