"""CLI entrypoints: ``demo | run | chat | markup``.

``demo`` is the single acceptance-criterion command: ingest both PIDs, compute + write the delta
report, then run grounded chat (one query non-interactively, or an interactive terminal loop) — all
in one invocation. Every subcommand opens a `pipeline.request` trace and writes a JSON trace file
unconditionally (works with zero external obs services).
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

from .chat.answer import answer
from .chat.index import build_index
from .config import default_config_path, load_config
from .delta.engine import compute_delta
from .delta.report import render_report
from .ingest import base as ingest_base
from .observability import tracing
from .pids.resolver import resolve as resolve_pid

_REPO_ROOT = default_config_path().parent.parent


def _ingest_and_delta(pid_a: str, pid_b: str, cfg):
    with tracing.stage("ingest"):
        data_a, meta_a = resolve_pid(pid_a)
        doc_a = ingest_base.parse(data_a, pid_a, meta_a.get("filename"), source_rev="A", cfg=cfg)
        data_b, meta_b = resolve_pid(pid_b)
        doc_b = ingest_base.parse(data_b, pid_b, meta_b.get("filename"), source_rev="B", cfg=cfg)
    with tracing.stage("delta"):
        delta = compute_delta(doc_a, doc_b, cfg)
        report_md, report_json = render_report(delta)
    return doc_a, doc_b, delta, report_json, report_md


def _delta_id(pid_a: str, pid_b: str) -> str:
    import hashlib

    # json.dumps unambiguously delimits/escapes each string, so a literal separator character
    # inside a pid can't make two distinct (pid_a, pid_b) pairs collide on the same hash.
    return hashlib.sha256(json.dumps([pid_a, pid_b]).encode()).hexdigest()[:16]


def _write_report(report_json: dict, report_md: str, out_dir: str | Path | None, pid_a: str, pid_b: str) -> Path:
    out = Path(out_dir) if out_dir else _REPO_ROOT / "data" / "output" / _delta_id(pid_a, pid_b)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    (out / "report.md").write_text(report_md, encoding="utf-8")
    return out


def _print_answer(result) -> None:
    print(f"[{result.state}] {result.text}")
    if result.refusal_reason:
        print(f"  refusal_reason: {result.refusal_reason}")
    for claim in result.claims:
        for cit in claim.citations:
            print(f"  cite: {cit.citation_id} ({cit.source_type}) page={cit.page} sheet={cit.sheet}")


def _interactive_chat_loop(index, cfg, *, in_stream=None) -> None:
    stream = in_stream or sys.stdin
    print("Grounded chat ready. Type a question (blank line or 'exit' to quit).")
    while True:
        print("> ", end="", flush=True)
        line = stream.readline()
        if not line:
            break
        query = line.strip()
        if not query or query.lower() in ("exit", "quit"):
            break
        _print_answer(answer(query, index, cfg))


def demo(pid_a: str, pid_b: str, query: str | None = None, cfg=None, *, out_dir=None, in_stream=None) -> int:
    """SINGLE acceptance command: ingest both PIDs -> compute delta -> render report (JSON+MD to
    disk) -> then run grounded chat. Non-interactive if `query` is given, else an interactive loop."""
    cfg = cfg or load_config()
    tr = tracing.start_request({"pid_a": pid_a, "pid_b": pid_b, "command": "demo"}, cfg=cfg)
    try:
        doc_a, doc_b, delta, report_json, report_md = _ingest_and_delta(pid_a, pid_b, cfg)
        out = _write_report(report_json, report_md, out_dir, pid_a, pid_b)
        total = sum(report_json["counts"].values())
        print(f"Delta computed: {total} change(s). Report written to {out}")
        index = build_index(doc_a, doc_b, delta, cfg)
        if query is not None:
            _print_answer(answer(query, index, cfg))
        else:
            _interactive_chat_loop(index, cfg, in_stream=in_stream)
    except Exception as exc:
        traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        tr.finish()
    return 0


def run(pid_a: str, pid_b: str, cfg=None, *, out_dir=None) -> int:
    """Ingest both -> delta report (JSON+MD) + markup (best-effort) -> write a trace file."""
    cfg = cfg or load_config()
    tr = tracing.start_request({"pid_a": pid_a, "pid_b": pid_b, "command": "run"}, cfg=cfg)
    try:
        doc_a, doc_b, delta, report_json, report_md = _ingest_and_delta(pid_a, pid_b, cfg)
        out = _write_report(report_json, report_md, out_dir, pid_a, pid_b)
        total = sum(report_json["counts"].values())
        print(f"Delta computed: {total} change(s). Report written to {out}")
        try:
            from .markup import overlay
        except ImportError:
            print("markup overlay not available (section-09 not yet built) — skipping", file=sys.stderr)
        else:
            # Markup is a best-effort bonus: a bug/malformed-PDF failure here must not turn an
            # otherwise-successful `run` (delta report already written above) into exit code 1.
            try:
                data_b, _meta = resolve_pid(pid_b)
                pdf_bytes = overlay.render_markup(delta, data_b, cfg=cfg)
                (out / "markup.pdf").write_bytes(pdf_bytes)
                print(f"Markup written to {out / 'markup.pdf'}")
            except Exception as markup_exc:
                traceback.print_exc()
                print(f"markup failed, skipping: {type(markup_exc).__name__}: {markup_exc}", file=sys.stderr)
    except Exception as exc:
        traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        tr.finish()
    return 0


def chat(pid_a: str, pid_b: str, cfg=None, *, in_stream=None) -> int:
    """Interactive grounded chat in the terminal over the computed delta+docs."""
    cfg = cfg or load_config()
    tr = tracing.start_request({"pid_a": pid_a, "pid_b": pid_b, "command": "chat"}, cfg=cfg)
    try:
        doc_a, doc_b, delta, _report_json, _report_md = _ingest_and_delta(pid_a, pid_b, cfg)
        index = build_index(doc_a, doc_b, delta, cfg)
        _interactive_chat_loop(index, cfg, in_stream=in_stream)
    except Exception as exc:
        traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        tr.finish()
    return 0


def markup(pid_a: str, pid_b: str, out: str, cfg=None) -> int:
    """Ingest -> delta -> render_markup -> write annotated PDF to `out`."""
    cfg = cfg or load_config()
    tr = tracing.start_request({"pid_a": pid_a, "pid_b": pid_b, "command": "markup"}, cfg=cfg)
    try:
        _doc_a, _doc_b, delta, _report_json, _report_md = _ingest_and_delta(pid_a, pid_b, cfg)
        try:
            from .markup import overlay
        except ImportError:
            print("markup overlay not available (section-09 not yet built)", file=sys.stderr)
            return 1
        data_b, _meta = resolve_pid(pid_b)
        pdf_bytes = overlay.render_markup(delta, data_b)
        Path(out).write_bytes(pdf_bytes)
        print(f"Markup written to {out}")
    except Exception as exc:
        traceback.print_exc()
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        tr.finish()
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="delta-chat")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_pids(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--pid-a", required=True)
        sp.add_argument("--pid-b", required=True)

    p_demo = sub.add_parser("demo", help="ingest -> delta report -> grounded chat, one invocation")
    add_pids(p_demo)
    p_demo.add_argument("--query", default=None)
    p_demo.add_argument("--out-dir", default=None, help="defaults to data/output/<delta_id>")

    p_run = sub.add_parser("run", help="ingest -> delta report + markup (best-effort)")
    add_pids(p_run)
    p_run.add_argument("--out-dir", default=None, help="defaults to data/output/<delta_id>")

    p_chat = sub.add_parser("chat", help="interactive grounded chat over an ingested pair")
    add_pids(p_chat)

    p_markup = sub.add_parser("markup", help="ingest -> delta -> annotated PDF")
    add_pids(p_markup)
    p_markup.add_argument("--out", required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    args = _build_parser().parse_args(argv)
    if args.command == "demo":
        return demo(args.pid_a, args.pid_b, query=args.query, out_dir=args.out_dir)
    if args.command == "run":
        return run(args.pid_a, args.pid_b, out_dir=args.out_dir)
    if args.command == "chat":
        return chat(args.pid_a, args.pid_b)
    if args.command == "markup":
        return markup(args.pid_a, args.pid_b, args.out)
    return 1  # pragma: no cover - argparse `required=True` makes this unreachable


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
