"""Tests for chat/index.py — retrieval over doc + delta pools. No LLM involved."""

from __future__ import annotations

from delta_chat.chat.index import build_index, retrieve, route_intent


def test_bm25_retrieves_exact_tag_query_into_top_k(chat_index, cfg):
    hits = retrieve(chat_index, "FV-9040", k=3, cfg=cfg)
    assert any("FV-9040" in c.text for c in hits)


def test_chunks_tagged_by_source_and_delta_one_chunk_per_change(chat_index, sample_delta):
    delta_chunks = [c for c in chat_index.chunks if c.source_type == "delta"]
    assert len(delta_chunks) == len(sample_delta.changes)
    assert {c.citation_id for c in delta_chunks} == {f"delta:{ch.id}" for ch in sample_delta.changes}

    doc_a_chunks = [c for c in chat_index.chunks if c.source_type == "doc_a"]
    doc_b_chunks = [c for c in chat_index.chunks if c.source_type == "doc_b"]
    assert doc_a_chunks and doc_b_chunks
    assert all(c.citation_id.startswith("doc_a:") for c in doc_a_chunks)
    assert all(c.citation_id.startswith("doc_b:") for c in doc_b_chunks)


def test_intent_router_sends_what_changed_to_delta_pool():
    assert route_intent("what changed between rev A and rev B?") == "delta"


def test_intent_router_sends_content_question_to_doc_pool():
    assert route_intent("what is the setpoint of FV-9040?") == "doc"


def test_delta_intent_query_retrieves_only_delta_chunks(chat_index, cfg):
    hits = retrieve(chat_index, "what changed for FV-9040?", k=5, cfg=cfg)
    assert hits
    assert all(c.source_type == "delta" for c in hits)


def test_doc_intent_query_retrieves_only_doc_chunks(chat_index, cfg):
    hits = retrieve(chat_index, "what is the setpoint of FV-9040?", k=5, cfg=cfg)
    assert hits
    assert all(c.source_type in ("doc_a", "doc_b") for c in hits)


def test_retrieved_chunks_carry_citation_id_page_sheet_bbox(chat_index, cfg):
    hits = retrieve(chat_index, "FV-9040", k=3, cfg=cfg)
    assert hits
    for c in hits:
        assert c.citation_id
        assert c.page is not None
        assert c.sheet is not None
        assert c.bbox is not None


def test_build_index_is_deterministic_given_same_inputs(doc_a, doc_b, sample_delta, cfg):
    idx1 = build_index(doc_a, doc_b, sample_delta, cfg)
    idx2 = build_index(doc_a, doc_b, sample_delta, cfg)
    assert [c.citation_id for c in idx1.chunks] == [c.citation_id for c in idx2.chunks]


def test_hybrid_retrieval_rrf_fuses_bm25_and_dense(chat_index, cfg):
    cfg.data["chat"]["retrieval"] = "hybrid"
    hits = retrieve(chat_index, "FV-9040 setpoint", k=3, cfg=cfg)
    assert hits
    assert any("FV-9040" in c.text for c in hits)
