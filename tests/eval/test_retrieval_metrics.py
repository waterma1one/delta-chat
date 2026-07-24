"""Tests for eval/retrieval_metrics.py — Hit-Rate@k/Recall@k/MRR and the trace-derived budget
table. Pure code, no LLM."""

from __future__ import annotations

from eval.gold import RetrievalGold
from eval.retrieval_metrics import build_budget_table, compute_budget_stats, score_retrieval


def test_hit_rate_recall_mrr_on_tiny_labeled_set():
    gold = [
        RetrievalGold(query="q1", relevant_chunk_ids=["a", "b"], pair_id="p1"),
        RetrievalGold(query="q2", relevant_chunk_ids=["x"], pair_id="p1"),
    ]
    retrieved = {
        "q1": ["z", "a", "b"],  # first relevant hit at rank 2
        "q2": ["y", "z"],  # no relevant hit at all
    }

    score = score_retrieval(retrieved, gold, ks=[3])

    assert score[3]["n_queries"] == 2
    assert score[3]["hit_rate"] == 0.5  # only q1 has >=1 relevant chunk in top-3
    assert score[3]["recall"] == 0.5  # q1: 2/2 relevant found=1.0, q2: 0/1=0.0 -> avg 0.5
    assert score[3]["mrr"] == 0.25  # q1: 1/2, q2: 0 -> avg 0.25


def test_hit_rate_is_zero_when_k_too_small_to_reach_the_relevant_chunk():
    gold = [RetrievalGold(query="q1", relevant_chunk_ids=["a"], pair_id="p1")]
    retrieved = {"q1": ["z", "y", "a"]}  # relevant chunk is rank 3

    score = score_retrieval(retrieved, gold, ks=[1, 3])

    assert score[1]["hit_rate"] == 0.0
    assert score[3]["hit_rate"] == 1.0


def test_query_missing_from_retrieved_is_skipped_not_counted_as_a_miss():
    gold = [RetrievalGold(query="never-run", relevant_chunk_ids=["a"], pair_id="p1")]
    score = score_retrieval({}, gold, ks=[5])
    assert score[5]["n_queries"] == 0
    assert score[5]["hit_rate"] == 0.0


def test_budget_table_aggregates_p50_p95_tokens_usd_from_trace_files(trace_files):
    table = build_budget_table(trace_files, sheets_extrapolation=10)

    assert "Runs aggregated: 2" in table
    assert "llm.call" in table
    assert "ingest" in table
    # llm.call tokens: run1 in=100/out=50, run2 in=300/out=60 -> totals 400/110
    assert "400" in table
    assert "110" in table
    # total usd 0.01 + 0.02 = 0.03
    assert "0.0300" in table
    assert "sheet set" in table.lower()


def test_compute_budget_stats_p95_and_cost_per_request(trace_files):
    # end-to-end request durations from the fixture: run1=100.0ms, run2=300.0ms.
    stats = compute_budget_stats(trace_files)

    assert stats["latency_p95_ms"] in (100.0, 300.0)  # p95 of a 2-sample set is one of the two values
    assert stats["cost_usd_per_request"] == (0.01 + 0.02) / 2


def test_compute_budget_stats_empty_input_is_zero_not_crash():
    stats = compute_budget_stats([])
    assert stats == {"latency_p95_ms": 0.0, "cost_usd_per_request": 0.0}
