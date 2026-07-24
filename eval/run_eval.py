"""Evaluation orchestrator (§10.6): scorecard.json + budget.md + baseline diff + CI exit code.

**CI gates ONLY the `deterministic` block** (delta P/R/F1 on held-out test gold, citation
validity — both pure code, byte-stable) vs. a stored baseline, using `eval.regression_thresholds`.
Model-graded metrics (faithfulness, relevancy, context P/R, citation support) are **never gating**:
they're reported as mean +/- stdev over `eval.judge.n_samples` samples so judge/temp-0
nondeterminism can never fire a false regression. The scorecard's gated sections are byte-stable
across runs — floats rounded to 6dp, dict keys sorted at write time, and the run timestamp lives in
a separate non-gated `meta` field, never inside `deterministic`.

Predictions (`delta_pred_by_pair`, `chat_preds`, `retrieved`) are dependency-injected: pass them
explicitly for a deterministic, network-free run (what the test suite does), or omit them to run
the real pipeline (ingest -> compute_delta -> chat.answer.answer against `data/samples/`), which
does hit the LLM — this "live" path is not exercised by the default test suite, matching the same
convention used for other live-network paths elsewhere in this codebase.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from delta_chat.canonical.model import Change
from delta_chat.chat.types import ChatAnswer
from delta_chat.config import default_config_path, load_config

from .chat_metrics import Judge, score_chat
from .delta_metrics import DeltaScore, score_delta
from .gold import GoldChange, QAGold, dataset_hash, load_delta_gold, load_qa_gold, qa_gold_as_retrieval_gold
from .retrieval_metrics import build_budget_table, compute_budget_stats, score_retrieval

_REPO_ROOT = default_config_path().parent.parent
# metrics where a larger number is a regression, not an improvement — check_regression must flip
# the comparison direction for these instead of assuming "higher is always better".
_LOWER_IS_BETTER = {"latency_p95_ms", "cost_usd_per_request"}


def _round_floats(obj):
    """Round every float to 6dp so the gated scorecard sections are byte-stable across runs
    (avoids float repr jitter across platforms/runs); combined with json.dumps(sort_keys=True)."""
    if isinstance(obj, float):
        return round(obj, 6)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    return obj


def _delta_score_dict(score: DeltaScore | None) -> dict | None:
    if score is None:
        return None
    return {
        "per_tau": {str(tau): v for tau, v in score.per_tau.items()},
        "soft_f1": score.soft_f1,
        "type_confusion": score.type_confusion,
        "by_segment": score.by_segment,
        "counts": score.counts,
    }


def _multi_sample_chat_score(preds: list[ChatAnswer], gold: list[QAGold], judge: Judge, n_samples: int) -> dict:
    """Only the genuinely judge-touched fields (faithfulness, answer_relevancy, citation_support)
    vary run to run with judge noise, so only those are reported as mean +/- stdev over
    `n_samples` calls (`statistics.pstdev`: we have the complete set of N samples actually taken,
    not a sample estimating a larger population, so population variance is the right estimator).
    citation_validity/context_precision/context_recall/abstention/per_class are pure code (no
    judge call) and byte-stable given fixed preds, so they're taken once rather than wastefully
    recomputed N times with an always-zero stdev."""
    samples = [score_chat(preds, gold, judge) for _ in range(max(1, n_samples))]

    def _agg(field: str) -> dict:
        values = [getattr(s, field) for s in samples]
        return {"mean": statistics.mean(values), "stdev": statistics.pstdev(values) if len(values) > 1 else 0.0}

    return {
        "faithfulness": _agg("faithfulness"),
        "answer_relevancy": _agg("answer_relevancy"),
        "citation_support": _agg("citation_support"),
        "context_precision": samples[0].context_precision,
        "context_recall": samples[0].context_recall,
        "citation_validity": samples[0].citation_validity,
        "abstention": samples[0].abstention,
        "per_class": samples[0].per_class,
    }


def _compute_delta_pred_live(cfg, pair_ids: set[str]) -> dict[str, list[Change]]:
    """Live path: ingest each pair's rev_a/rev_b PDFs from data/samples/ and compute the real
    Delta. Skips a pair gracefully (rather than crashing the whole eval run) if its source files
    aren't the plain rev_a.pdf/rev_b.pdf naming (e.g. the scanned/DXF pairs use different names)."""
    from delta_chat.delta.engine import compute_delta
    from delta_chat.ingest import base as ingest_base

    samples_dir = _REPO_ROOT / "data" / "samples"
    out: dict[str, list[Change]] = {}
    for pair_id in pair_ids:
        rev_a = samples_dir / pair_id / "rev_a.pdf"
        rev_b = samples_dir / pair_id / "rev_b.pdf"
        if not rev_a.exists() or not rev_b.exists():
            continue
        doc_a = ingest_base.parse(rev_a.read_bytes(), f"{pair_id}:A", source_rev="A", cfg=cfg)
        doc_b = ingest_base.parse(rev_b.read_bytes(), f"{pair_id}:B", source_rev="B", cfg=cfg)
        out[pair_id] = compute_delta(doc_a, doc_b, cfg).changes
    return out


def _compute_chat_preds_live(
    cfg, delta_pred_by_pair: dict[str, list[Change]], chat_gold: list[QAGold]
) -> tuple[list[ChatAnswer], dict[str, list[str]]]:
    """Live path: for each QA gold question, ingest its pair's docs, build a ChatIndex, and run
    the real grounded-chat pipeline (hits the LLM). Returns (preds, retrieved) where `retrieved`
    maps question -> the citation_ids chat.index.retrieve returned, for the retrieval metrics."""
    from delta_chat.chat.answer import answer
    from delta_chat.chat.index import build_index, retrieve
    from delta_chat.delta.engine import compute_delta
    from delta_chat.ingest import base as ingest_base

    samples_dir = _REPO_ROOT / "data" / "samples"
    index_cache: dict[str, object] = {}
    preds: list[ChatAnswer] = []
    retrieved: dict[str, list[str]] = {}
    top_k = int(cfg.get("chat.top_k", 8))

    for g in chat_gold:
        if g.pair_id not in index_cache:
            rev_a = samples_dir / g.pair_id / "rev_a.pdf"
            rev_b = samples_dir / g.pair_id / "rev_b.pdf"
            if not rev_a.exists() or not rev_b.exists():
                index_cache[g.pair_id] = None
            else:
                doc_a = ingest_base.parse(rev_a.read_bytes(), f"{g.pair_id}:A", source_rev="A", cfg=cfg)
                doc_b = ingest_base.parse(rev_b.read_bytes(), f"{g.pair_id}:B", source_rev="B", cfg=cfg)
                delta = delta_pred_by_pair.get(g.pair_id) or compute_delta(doc_a, doc_b, cfg).changes
                from delta_chat.canonical.model import Delta as _Delta
                index_cache[g.pair_id] = build_index(doc_a, doc_b, _Delta(pid_a="A", pid_b="B", changes=delta), cfg)
        index = index_cache[g.pair_id]
        if index is None:
            continue
        retrieved[g.question] = [c.citation_id for c in retrieve(index, g.question, top_k, cfg)]
        preds.append(answer(g.question, index, cfg))
    return preds, retrieved


def build_scorecard(
    cfg,
    *,
    delta_pred_by_pair: dict[str, list[Change]],
    delta_gold: list[GoldChange],
    chat_preds: list[ChatAnswer],
    chat_gold: list[QAGold],
    judge: Judge,
    retrieved: dict[str, list[str]],
    dataset_paths: list[Path],
    trace_files: list[str | Path] | None = None,
) -> dict:
    """Pure scorecard builder — no I/O beyond hashing the dataset files, reading `trace_files`,
    and calling the metric functions. Segmented by pair/format/hard_slice inside `delta`; never a
    single averaged headline number."""
    taus = list(cfg.get("eval.taus"))
    ks = list(cfg.get("eval.retrieval_ks"))
    n_samples = int(cfg.get("eval.judge.n_samples", 1))
    split = cfg.get("eval.datasets.split")
    train_pairs, test_pairs = set(split.get("train", [])), set(split.get("test", []))

    train_gold = [g for g in delta_gold if g.pair_id in train_pairs]
    test_gold = [g for g in delta_gold if g.pair_id in test_pairs]
    train_pred = [c for pid in train_pairs for c in delta_pred_by_pair.get(pid, [])]
    test_pred = [c for pid in test_pairs for c in delta_pred_by_pair.get(pid, [])]

    delta_test = score_delta(test_pred, test_gold, taus)
    delta_train = score_delta(train_pred, train_gold, taus) if train_gold else None

    chat_block = _multi_sample_chat_score(chat_preds, chat_gold, judge, n_samples)
    retrieval_gold = qa_gold_as_retrieval_gold(chat_gold)
    retrieval_block = score_retrieval(retrieved, retrieval_gold, ks)
    budget_stats = compute_budget_stats(trace_files or [])

    max_tau = max(taus) if taus else 0.5
    # The exact numbers the CI gate checks: delta P/R/F1 on held-out test gold, citation validity,
    # latency, cost — all pure code / byte-stable. Gating only actually fires for whichever of
    # these also has a threshold configured in eval.regression_thresholds (see check_regression).
    deterministic = {
        "delta_f1_test": delta_test.per_tau[max_tau]["f1"],
        "citation_validity": chat_block["citation_validity"],
        "latency_p95_ms": budget_stats["latency_p95_ms"],
        "cost_usd_per_request": budget_stats["cost_usd_per_request"],
    }

    scorecard = {
        "dataset_hash": dataset_hash(dataset_paths),
        "split": {"train": sorted(train_pairs), "test": sorted(test_pairs)},
        "delta": {"test": _delta_score_dict(delta_test), "train": _delta_score_dict(delta_train)},
        "chat": chat_block,
        "retrieval": retrieval_block,
        "deterministic": deterministic,
    }
    return _round_floats(scorecard)


def check_regression(scorecard: dict, baseline: dict | None, thresholds: dict) -> list[str]:
    """Compare ONLY the `deterministic` block against `baseline`; return regression messages
    (empty == no regression == exit 0). Never inspects model-graded fields. Direction-aware: most
    deterministic metrics are higher-is-better (a DROP is a regression), but `_LOWER_IS_BETTER`
    metrics (latency, cost) regress on an INCREASE instead."""
    if baseline is None:
        return []
    problems: list[str] = []
    det = scorecard.get("deterministic", {})
    base_det = baseline.get("deterministic", {})
    for key in det:
        if key not in base_det or key not in thresholds:
            continue
        threshold = float(thresholds[key])
        if key in _LOWER_IS_BETTER:
            change = det[key] - base_det[key]
            direction = "increase"
        else:
            change = base_det[key] - det[key]
            direction = "drop"
        if change > threshold:
            problems.append(
                f"{key} regressed: {base_det[key]:.4f} -> {det[key]:.4f} "
                f"({direction} {change:.4f} > threshold {threshold})"
            )
    return problems


def run_eval(
    cfg=None,
    *,
    baseline_path: str | Path | None = None,
    out_dir: str | Path | None = None,
    judge: Judge | None = None,
    delta_pred_by_pair: dict[str, list[Change]] | None = None,
    delta_gold: list[GoldChange] | None = None,
    chat_preds: list[ChatAnswer] | None = None,
    chat_gold: list[QAGold] | None = None,
    retrieved: dict[str, list[str]] | None = None,
    trace_files: list[str | Path] | None = None,
) -> int:
    """Load held-out test gold (+ dataset hash + split provenance). Run delta, chat, retrieval,
    judge metrics. Write scorecard.json (segmented by pair/format/hard_slice) and budget.md. Diff
    vs baseline. Return the process exit code: non-zero iff a DETERMINISTIC metric regressed past
    its config threshold.

    `delta_gold`/`chat_gold` default to loading from `eval.datasets.dir` on disk but can be
    injected (alongside matching predictions) for a fully isolated, deterministic test run."""
    cfg = cfg or load_config()
    out_dir = Path(out_dir) if out_dir else _REPO_ROOT / "eval" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_dir = _REPO_ROOT / cfg.get("eval.datasets.dir")
    delta_gold_path, qa_gold_path = dataset_dir / "delta_gold.jsonl", dataset_dir / "qa_gold.jsonl"
    delta_gold = delta_gold if delta_gold is not None else load_delta_gold(delta_gold_path)
    chat_gold = chat_gold if chat_gold is not None else load_qa_gold(qa_gold_path)

    split = cfg.get("eval.datasets.split")
    all_pairs = set(split.get("train", [])) | set(split.get("test", []))

    if delta_pred_by_pair is None:
        delta_pred_by_pair = _compute_delta_pred_live(cfg, all_pairs)
    if chat_preds is None or retrieved is None:
        live_preds, live_retrieved = _compute_chat_preds_live(cfg, delta_pred_by_pair, chat_gold)
        chat_preds = chat_preds if chat_preds is not None else live_preds
        retrieved = retrieved if retrieved is not None else live_retrieved
    if judge is None:
        raise ValueError("run_eval requires a `judge` (inject a mock for tests, a real one live)")
    trace_files = trace_files if trace_files is not None else sorted(Path(cfg.get("obs.trace_dir", "traces")).glob("*.json"))

    scorecard = build_scorecard(
        cfg,
        delta_pred_by_pair=delta_pred_by_pair, delta_gold=delta_gold,
        chat_preds=chat_preds, chat_gold=chat_gold, judge=judge,
        retrieved=retrieved, dataset_paths=[delta_gold_path, qa_gold_path],
        trace_files=trace_files,
    )
    scorecard["meta"] = {"generated_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()}

    (out_dir / "scorecard.json").write_text(
        json.dumps(scorecard, indent=2, sort_keys=True), encoding="utf-8"
    )
    (out_dir / "budget.md").write_text(
        build_budget_table(trace_files, sheets_extrapolation=int(cfg.get("eval.budget.sheets_extrapolation", 500))),
        encoding="utf-8",
    )

    baseline = None
    if baseline_path and Path(baseline_path).exists():
        baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    problems = check_regression(scorecard, baseline, dict(cfg.get("eval.regression_thresholds", {})))
    for p in problems:
        print(f"REGRESSION: {p}")
    return 1 if problems else 0


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin CLI wrapper
    import argparse

    parser = argparse.ArgumentParser(prog="eval.run_eval")
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args(argv)

    from .judge import assert_cross_family

    cfg = load_config()
    assert_cross_family(cfg)

    return run_eval(cfg, baseline_path=args.baseline, out_dir=args.out_dir, judge=_LiveJudge(cfg))


class _LiveJudge:
    """Best-effort live judge for `python -m eval.run_eval`. `chat_metrics.Judge.entails` only
    receives a `citation_id`, not the actual cited chunk text (`score_chat` has no chunk-text
    access), so this asks the judge to assess the claim's internal specificity/coherence rather
    than a real span-entailment check — a materially weaker signal, documented here rather than
    silently claimed as equivalent. This is still a genuine model call, not a hardcoded verdict:
    the prior version of this class always returned True/1.0 regardless of input, fabricating a
    perfect score for every claim — exactly the "eval you can't trust" failure this section exists
    to catch. Not exercised by the default test suite (network path), same convention as other
    live-LLM paths elsewhere in this codebase."""

    def __init__(self, cfg):
        self._cfg = cfg
        self._model = str(cfg.get("eval.judge.model"))

    def entails(self, claim_text: str, citation_id: str) -> bool:
        from .judge import judge_pointwise

        verdict = judge_pointwise(
            f"Claim: {claim_text}\nCited source id: {citation_id}",
            model=self._model, cfg=self._cfg,
            rubric=(
                "Score 4 if the claim reads as specific and plausibly supportable by a real "
                "source; score 1 if it reads vague, generic, or fabricated-sounding."
            ),
        )
        return verdict.score >= 3

    def relevancy(self, answer_text: str, question: str) -> float:
        from .judge import judge_pointwise

        verdict = judge_pointwise(
            f"Question: {question}\nAnswer: {answer_text}",
            model=self._model, cfg=self._cfg,
            rubric="Score how directly the answer addresses the question: 1 (off-topic) to 4 (fully addresses it).",
        )
        return (verdict.score - 1) / 3.0  # map the 1..4 scale onto 0..1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
