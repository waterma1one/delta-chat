"""Tests for cli.py — `demo`/`run` are the reproducible-run path. The LLM is
mocked at `cli.answer` (canned `ChatAnswer`); ingest/delta run for real against the golden native
sample pair."""

from __future__ import annotations

import copy
import io
import json
import uuid
from pathlib import Path

import pytest

from delta_chat import cli
from delta_chat.chat.types import ChatAnswer
from delta_chat.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
PAIR_01_DIR = REPO_ROOT / "data" / "samples" / "pair_01_native"


@pytest.fixture
def cli_cfg(tmp_path):
    data = copy.deepcopy(load_config().data)
    data["obs"]["trace_dir"] = str(tmp_path / "traces")
    data["obs"]["otlp_enabled"] = False
    data["delta"]["notes"]["semantic_low"] = 0.35
    data["delta"]["notes"]["semantic_high"] = 0.85
    return Config(data)


@pytest.fixture
def registered_pair():
    from delta_chat.pids import resolver

    if not PAIR_01_DIR.exists():  # pragma: no cover - environment guard
        pytest.skip(f"sample pair missing: {PAIR_01_DIR}")
    suffix = uuid.uuid4().hex[:8]
    pid_a, pid_b = f"test-cli-pair01-a-{suffix}", f"test-cli-pair01-b-{suffix}"
    resolver.register(pid_a, str(PAIR_01_DIR / "rev_a.pdf"))
    resolver.register(pid_b, str(PAIR_01_DIR / "rev_b.pdf"))
    return pid_a, pid_b


@pytest.fixture
def mock_answer(monkeypatch):
    calls = []
    queue = []

    def _answer(query, index, cfg):
        calls.append(query)
        if not queue:
            raise AssertionError("mock_answer: no queued ChatAnswer")
        return queue.pop(0)

    monkeypatch.setattr(cli, "answer", _answer)

    class Recorder:
        def __init__(self):
            self.calls = calls

        def queue(self, result):
            queue.append(result)

    return Recorder()


def test_demo_runs_ingest_delta_and_chat_in_one_invocation(registered_pair, cli_cfg, mock_answer, tmp_path, capsys):
    pid_a, pid_b = registered_pair
    mock_answer.queue(ChatAnswer(
        state="Grounded", text="the setpoint changed to 175 psi",
        claims=[], used_chunks=[], refusal_reason=None,
    ))

    exit_code = cli.demo(pid_a, pid_b, query="what changed?", cfg=cli_cfg, out_dir=tmp_path / "out")

    assert exit_code == 0
    assert len(mock_answer.calls) == 1
    assert (tmp_path / "out" / "report.json").exists()
    assert (tmp_path / "out" / "report.md").exists()
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert "changes" in report and "counts" in report

    out = capsys.readouterr().out
    assert "Grounded" in out
    assert "the setpoint changed to 175 psi" in out


def test_run_writes_trace_file_with_zero_obs_services(registered_pair, cli_cfg, tmp_path):
    pid_a, pid_b = registered_pair

    exit_code = cli.run(pid_a, pid_b, cfg=cli_cfg, out_dir=tmp_path / "out")

    assert exit_code == 0
    trace_dir = Path(cli_cfg.get("obs.trace_dir"))
    trace_files = list(trace_dir.glob("*.json"))
    assert trace_files
    trace = json.loads(trace_files[0].read_text())
    assert set(trace) >= {"request_id", "trace_id", "duration_ms", "status", "stages"}
    assert trace["status"] in ("ok", "degraded")
    stage_names = {s["name"] for s in trace["stages"]}
    assert {"ingest", "delta"} <= stage_names


def test_run_writes_markup_pdf(registered_pair, cli_cfg, tmp_path):
    # markup.overlay now exists — `run` writes a real annotated PDF rather than
    # skipping, which is what this test asserted before it was built.
    pid_a, pid_b = registered_pair
    exit_code = cli.run(pid_a, pid_b, cfg=cli_cfg, out_dir=tmp_path / "out")
    assert exit_code == 0
    markup_pdf = tmp_path / "out" / "markup.pdf"
    assert markup_pdf.exists()
    assert markup_pdf.read_bytes().startswith(b"%PDF-")


def test_run_skips_markup_honestly_when_overlay_unavailable(registered_pair, cli_cfg, tmp_path, capsys, monkeypatch):
    import sys

    import delta_chat.markup as markup_pkg

    monkeypatch.delattr(markup_pkg, "overlay", raising=False)
    monkeypatch.setitem(sys.modules, "delta_chat.markup.overlay", None)

    pid_a, pid_b = registered_pair
    exit_code = cli.run(pid_a, pid_b, cfg=cli_cfg, out_dir=tmp_path / "out")
    assert exit_code == 0
    assert not (tmp_path / "out" / "markup.pdf").exists()
    assert "markup overlay not available" in capsys.readouterr().err


def test_demo_interactive_loop_reads_from_injected_stream(registered_pair, cli_cfg, mock_answer, tmp_path, capsys):
    pid_a, pid_b = registered_pair
    mock_answer.queue(ChatAnswer(state="Grounded", text="answer one", claims=[], used_chunks=[], refusal_reason=None))
    stream = io.StringIO("what changed?\nexit\n")

    exit_code = cli.demo(pid_a, pid_b, query=None, cfg=cli_cfg, out_dir=tmp_path / "out", in_stream=stream)

    assert exit_code == 0
    assert mock_answer.calls == ["what changed?"]
    assert "answer one" in capsys.readouterr().out


def test_main_dispatches_demo_subcommand(registered_pair, mock_answer, tmp_path, monkeypatch):
    pid_a, pid_b = registered_pair
    mock_answer.queue(ChatAnswer(state="Rejected", text="", claims=[], used_chunks=[], refusal_reason="INSUFFICIENT_EVIDENCE"))
    monkeypatch.setattr(cli, "load_config", lambda: _cfg_with_tmp_trace(tmp_path))

    exit_code = cli.main([
        "demo", "--pid-a", pid_a, "--pid-b", pid_b, "--query", "anything?",
        "--out-dir", str(tmp_path / "out"),
    ])

    assert exit_code == 0
    assert (tmp_path / "out" / "report.json").exists()
    assert mock_answer.calls == ["anything?"]


def test_delta_id_does_not_collide_on_unescaped_separator():
    assert cli._delta_id("a|b", "c") != cli._delta_id("a", "b|c")


def _cfg_with_tmp_trace(tmp_path):
    data = copy.deepcopy(load_config().data)
    data["obs"]["trace_dir"] = str(tmp_path / "traces")
    data["obs"]["otlp_enabled"] = False
    data["delta"]["notes"]["semantic_low"] = 0.35
    data["delta"]["notes"]["semantic_high"] = 0.85
    return Config(data)
