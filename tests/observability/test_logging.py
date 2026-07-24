"""Structured JSON logging carries request_id."""

from __future__ import annotations

import json

from delta_chat.observability import logging as olog


def test_logs_are_json_and_carry_request_id(capsys):
    olog.configure_logging()
    olog.bind_request_id("rid-123")
    olog.get_logger("t").info("hello", foo=1)
    out = capsys.readouterr().out.strip().splitlines()[-1]
    line = json.loads(out)
    assert line["request_id"] == "rid-123"
    assert line["event"] == "hello" and line["foo"] == 1
    olog.clear_request_context()


def test_configure_logging_idempotent():
    olog.configure_logging()
    olog.configure_logging()  # no raise / no double-config
    assert olog._configured is True
