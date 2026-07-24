"""Shared fixtures for the grounded-chat test suite."""

from __future__ import annotations

import copy

import pytest

from delta_chat.canonical.model import BBox, Change, Delta, Document, Item
from delta_chat.chat.index import build_index
from delta_chat.config import Config, load_config


@pytest.fixture
def cfg():
    """Default config as a fresh mutable copy — tests may override a key without affecting others."""
    return Config(copy.deepcopy(load_config().data))


def _item(id_, text, *, type="tag", region="notes", page=0, sheet="1", bbox=(0.1, 0.2, 0.5, 0.22)):
    x0, y0, x1, y1 = bbox
    return Item(
        id=id_, source_rev="A", type=type, text=text, parsed_fields={},
        bbox=BBox(x0, y0, x1, y1), page=page, sheet=sheet, region=region,
    )


@pytest.fixture
def doc_a() -> Document:
    return Document(
        pid="A", format="pdf_native", page_count=1, revision_label="A", sheet_size=(1.0, 1.0),
        items=[
            _item("0:notes:1", "FV-9040 control valve on cooling water line", type="tag"),
            _item("0:notes:2", "Setpoint for FV-9040 is 150 psi", type="note"),
            _item("0:title:1", "Cooling Water P&ID Sheet 1", type="text", region="title"),
        ],
    )


@pytest.fixture
def doc_b() -> Document:
    return Document(
        pid="B", format="pdf_native", page_count=1, revision_label="B", sheet_size=(1.0, 1.0),
        items=[
            _item("0:notes:1", "FV-9040 control valve on cooling water line", type="tag"),
            _item("0:notes:2", "Setpoint for FV-9040 is 175 psi", type="note"),
            _item("0:title:1", "Cooling Water P&ID Sheet 1", type="text", region="title"),
        ],
    )


@pytest.fixture
def sample_delta() -> Delta:
    return Delta(
        pid_a="A", pid_b="B",
        changes=[
            Change(
                id="CHG-0001", type="modified", item_type="note", page=0, sheet="1",
                bbox=BBox(0.1, 0.2, 0.5, 0.22),
                description="Setpoint for FV-9040 changed",
                old="150 psi", new="175 psi",
                rev_a_ref="0:notes:2", rev_b_ref="0:notes:2",
                match_conf=0.95, class_conf=0.9, change_set_id="CS-0-1",
            ),
        ],
        counts={"modified": 1},
        dataset_version="v1",
    )


@pytest.fixture
def chat_index(doc_a, doc_b, sample_delta, cfg):
    return build_index(doc_a, doc_b, sample_delta, cfg)


class _MockLlm:
    """Queue canned `LlmResult`s (or exceptions) for `delta_chat.chat.llm.complete`; records every
    call's messages/documents/response_format for assertions."""

    def __init__(self):
        self.calls: list[dict] = []
        self._queue: list[object] = []

    def queue(self, result) -> None:
        self._queue.append(result)

    def __call__(self, messages, *, model=None, response_format=None, documents=None, cfg=None):
        self.calls.append(
            {"messages": messages, "model": model, "response_format": response_format, "documents": documents}
        )
        if not self._queue:
            raise AssertionError("mock_llm: no queued response for this call")
        item = self._queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


@pytest.fixture
def mock_llm(monkeypatch):
    mock = _MockLlm()
    monkeypatch.setattr("delta_chat.chat.llm.complete", mock)
    return mock
