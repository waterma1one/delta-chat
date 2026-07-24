"""Retrieval-quality metrics + cost/latency budget table (§10.4). Both are deterministic, pure
code over a labeled query -> relevant-chunk-ids set (retrieval metrics) or trace JSON files
(budget table) — no LLM involved.
"""

from __future__ import annotations

import json
from pathlib import Path

from .gold import RetrievalGold


def score_retrieval(retrieved: dict[str, list[str]], gold: list[RetrievalGold], ks: list[int]) -> dict[int, dict]:
    """`retrieved` maps query -> ranked chunk_ids. Return {k: {hit_rate, recall, mrr}} per k.
    Queries in `gold` with no entry in `retrieved` are skipped (not counted as a miss) — they
    represent a query the harness never ran, not a retrieval failure."""
    result: dict[int, dict] = {}
    for k in ks:
        hits = 0
        recalls: list[float] = []
        reciprocal_ranks: list[float] = []
        n = 0
        for g in gold:
            if g.query not in retrieved:
                continue
            n += 1
            ranked = retrieved[g.query][:k]
            relevant = set(g.relevant_chunk_ids)
            found = set(ranked) & relevant
            if found:
                hits += 1
            recalls.append(len(found) / len(relevant) if relevant else 1.0)
            rr = 0.0
            for rank, chunk_id in enumerate(ranked, start=1):
                if chunk_id in relevant:
                    rr = 1.0 / rank
                    break
            reciprocal_ranks.append(rr)
        result[k] = {
            "hit_rate": hits / n if n else 0.0,
            "recall": sum(recalls) / len(recalls) if recalls else 0.0,
            "mrr": sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
            "n_queries": n,
        }
    return result


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    # round-half-up, not Python's round() (banker's rounding) — on an even sample count landing
    # exactly on a boundary (e.g. n=2, p50), round-half-to-even silently picks the lower value.
    idx = int(pct / 100.0 * (len(ordered) - 1) + 0.5)
    idx = min(len(ordered) - 1, max(0, idx))
    return ordered[idx]


def compute_budget_stats(trace_files: list[str | Path]) -> dict:
    """End-to-end (not per-stage) p95 request latency and mean USD per request, aggregated across
    `trace_files` — the two deterministic budget numbers the CI gate checks (`eval.
    regression_thresholds.latency_p95_ms`/`cost_usd_per_request`)."""
    durations: list[float] = []
    total_usd = 0.0
    n_runs = 0
    for path in trace_files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        n_runs += 1
        durations.append(float(data.get("duration_ms", 0.0)))
        for stage in data.get("stages", []):
            attrs = stage.get("attrs") or {}
            total_usd += float(attrs.get("usd", 0.0) or 0.0)
    return {
        "latency_p95_ms": _percentile(durations, 95),
        "cost_usd_per_request": total_usd / n_runs if n_runs else 0.0,
    }


def build_budget_table(trace_files: list[str | Path], *, sheets_extrapolation: int = 500) -> str:
    """Read trace JSON files (§9 schema); for each stage compute p50/p95 latency and sum
    input/output tokens + USD (from `stages[].attrs`); return a Markdown table for `budget.md`,
    including a 'what for a `sheets_extrapolation`-sheet set?' extrapolation note."""
    latencies: dict[str, list[float]] = {}
    tokens_in: dict[str, int] = {}
    tokens_out: dict[str, int] = {}
    usd_by_stage: dict[str, float] = {}
    n_runs = 0

    for path in trace_files:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        n_runs += 1
        for stage in data.get("stages", []):
            name = stage["name"]
            latencies.setdefault(name, []).append(float(stage.get("duration_ms", 0.0)))
            attrs = stage.get("attrs") or {}
            tokens_in[name] = tokens_in.get(name, 0) + int(attrs.get("gen_ai.usage.input_tokens", 0) or 0)
            tokens_out[name] = tokens_out.get(name, 0) + int(attrs.get("gen_ai.usage.output_tokens", 0) or 0)
            usd_by_stage[name] = usd_by_stage.get(name, 0.0) + float(attrs.get("usd", 0.0) or 0.0)

    lines = [
        "# Budget",
        "",
        f"Runs aggregated: {n_runs}",
        "",
        "| stage | p50 ms | p95 ms | input tokens | output tokens | USD |",
        "|---|---|---|---|---|---|",
    ]
    total_usd = 0.0
    for name in sorted(latencies):
        vals = latencies[name]
        p50, p95 = _percentile(vals, 50), _percentile(vals, 95)
        total_usd += usd_by_stage.get(name, 0.0)
        lines.append(
            f"| {name} | {p50:.1f} | {p95:.1f} | {tokens_in.get(name, 0)} | "
            f"{tokens_out.get(name, 0)} | {usd_by_stage.get(name, 0.0):.4f} |"
        )
    lines.append("")
    lines.append(f"**Total cost across {n_runs} run(s): ${total_usd:.4f}**")
    if n_runs:
        per_run_usd = total_usd / n_runs
        lines.append(
            f"Extrapolated to a {sheets_extrapolation}-sheet set (assuming ~1 request per sheet "
            f"pair): ${per_run_usd * sheets_extrapolation:.2f}"
        )
    return "\n".join(lines)
