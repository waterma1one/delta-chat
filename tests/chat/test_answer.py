"""Tests for chat/grounding.py + chat/answer.py — grounding paths, sufficiency gate, citation
verification, and the 3-state ChatAnswer. The LLM is always mocked via `mock_llm`; no network."""

from __future__ import annotations

import json

import pytest

from delta_chat.chat import answer as answer_mod
from delta_chat.chat import grounding
from delta_chat.chat.types import Chunk, LlmResult


def _json_llm_result(payload: dict, **usage) -> LlmResult:
    return LlmResult(
        text=json.dumps(payload), blocks=None,
        usage=usage or {"input_tokens": 100, "output_tokens": 20},
        model="claude-sonnet-5", finish_reason="stop",
    )


# --- JSON-schema grounding path -------------------------------------------------------------------

def test_json_schema_path_resolves_citation_ids(chat_index, cfg, mock_llm):
    chunks = list(chat_index.chunks)
    target = chunks[0]
    mock_llm.queue(_json_llm_result({
        "answer_supported": True,
        "claims": [{"text": target.text, "citations": [target.citation_id]}],
        "insufficient_evidence": False, "refusal_reason": None,
    }))

    result = grounding.ground_json_schema("some question", chunks, cfg)

    assert len(mock_llm.calls) == 1
    assert result.claims
    assert result.claims[0].citations[0].citation_id == target.citation_id
    assert not result.unresolved_ids


def test_json_schema_path_reasks_once_on_unresolved_id_then_succeeds(chat_index, cfg, mock_llm):
    chunks = list(chat_index.chunks)
    target = chunks[0]
    mock_llm.queue(_json_llm_result({
        "answer_supported": True,
        "claims": [{"text": "bad", "citations": ["doc_a:DOES-NOT-EXIST"]}],
        "insufficient_evidence": False, "refusal_reason": None,
    }))
    mock_llm.queue(_json_llm_result({
        "answer_supported": True,
        "claims": [{"text": target.text, "citations": [target.citation_id]}],
        "insufficient_evidence": False, "refusal_reason": None,
    }))

    result = grounding.ground_json_schema("some question", chunks, cfg)

    assert len(mock_llm.calls) == 2  # exactly one re-ask
    assert result.claims
    assert result.claims[0].citations[0].citation_id == target.citation_id


def test_json_schema_path_flags_when_still_unresolved_after_reask(chat_index, cfg, mock_llm):
    chunks = list(chat_index.chunks)
    bad_payload = {
        "answer_supported": True,
        "claims": [{"text": "bad", "citations": ["doc_a:DOES-NOT-EXIST"]}],
        "insufficient_evidence": False, "refusal_reason": None,
    }
    mock_llm.queue(_json_llm_result(bad_payload))
    mock_llm.queue(_json_llm_result(bad_payload))

    result = grounding.ground_json_schema("some question", chunks, cfg)

    assert len(mock_llm.calls) == 2  # never loops beyond the single re-ask
    assert result.unresolved_ids
    assert not result.claims  # a claim with zero resolvable citations is dropped, not fabricated


def test_delta_and_doc_citations_are_both_representable_and_distinguished(chat_index, cfg, mock_llm):
    doc_chunk = next(c for c in chat_index.chunks if c.source_type == "doc_a")
    delta_chunk = next(c for c in chat_index.chunks if c.source_type == "delta")
    mock_llm.queue(_json_llm_result({
        "answer_supported": True,
        "claims": [{
            "text": "combined claim",
            "citations": [doc_chunk.citation_id, delta_chunk.citation_id],
        }],
        "insufficient_evidence": False, "refusal_reason": None,
    }))

    result = grounding.ground_json_schema("q", [doc_chunk, delta_chunk], cfg)

    source_types = {c.source_type for c in result.claims[0].citations}
    assert source_types == {"doc_a", "delta"}


# --- Anthropic Citations grounding path -------------------------------------------------------------------

def test_anthropic_citations_path_parses_interleaved_blocks(chat_index, cfg, mock_llm):
    chunks = list(chat_index.chunks)
    target_idx = 1
    blocks = [{
        "type": "text",
        "text": "The setpoint changed to 175 psi.",
        "citations": [{
            "type": "char_location", "document_index": target_idx,
            "start_char_index": 0, "end_char_index": 20,
        }],
    }]
    mock_llm.queue(LlmResult(
        text="", blocks=blocks, usage={"input_tokens": 50, "output_tokens": 10},
        model="claude-sonnet-5", finish_reason="stop",
    ))

    result = grounding.ground_anthropic_citations("what changed?", chunks, cfg)

    assert mock_llm.calls[0]["documents"] is not None
    assert result.claims
    assert result.claims[0].citations[0].citation_id == chunks[target_idx].citation_id
    assert result.claims[0].citations[0].source_type == chunks[target_idx].source_type


# --- sufficiency gate -------------------------------------------------------------------

def test_sufficiency_gate_abstains_on_flat_normalized_margin(cfg):
    identical_text = "valve pressure rating specification document text"
    chunks = [
        Chunk(citation_id="doc_a:1", source_type="doc_a", text=identical_text, pid="A",
              page=0, sheet=None, bbox=None, char_range=(0, len(identical_text))),
        Chunk(citation_id="doc_a:2", source_type="doc_a", text=identical_text, pid="A",
              page=0, sheet=None, bbox=None, char_range=(0, len(identical_text))),
    ]

    sufficient, reason = grounding.check_sufficiency("valve pressure rating", chunks, cfg)

    assert sufficient is False
    assert reason


def test_sufficiency_gate_no_chunks_is_insufficient(cfg):
    sufficient, reason = grounding.check_sufficiency("anything", [], cfg)
    assert sufficient is False
    assert reason


# --- full answer() orchestration -------------------------------------------------------------------

def test_answer_unanswerable_question_rejected_with_refusal_reason(chat_index, cfg):
    result = answer_mod.answer("completely unrelated xyz zzy qqq query", chat_index, cfg)

    assert result.state == "Rejected"
    assert result.refusal_reason


def test_answer_grounded_when_evidence_strong_and_claim_supported(chat_index, cfg, mock_llm):
    cfg.data["chat"]["sufficiency"]["use_entailment_check"] = False
    delta_chunk = next(c for c in chat_index.chunks if c.source_type == "delta")
    mock_llm.queue(_json_llm_result({
        "answer_supported": True,
        "claims": [{"text": delta_chunk.text, "citations": [delta_chunk.citation_id]}],
        "insufficient_evidence": False, "refusal_reason": None,
    }))

    result = answer_mod.answer("what changed for FV-9040?", chat_index, cfg)

    assert result.state == "Grounded"
    assert result.claims
    for claim in result.claims:
        assert len(claim.citations) >= 1
        assert claim.verified is True


def test_answer_citation_verification_catches_non_entailing_span(chat_index, cfg, mock_llm):
    cfg.data["chat"]["sufficiency"]["use_entailment_check"] = False
    delta_chunk = next(c for c in chat_index.chunks if c.source_type == "delta")
    mock_llm.queue(_json_llm_result({
        "answer_supported": True,
        "claims": [{
            "text": "The cafeteria menu now offers vegan tacos on Fridays",
            "citations": [delta_chunk.citation_id],
        }],
        "insufficient_evidence": False, "refusal_reason": None,
    }))

    result = answer_mod.answer("what changed for FV-9040?", chat_index, cfg)

    assert result.state != "Grounded"
    assert result.refusal_reason
    # the unverified claim is still surfaced (not dropped) so eval/section-10 can see what was
    # emitted, just marked unverified rather than silently discarded.
    assert result.claims
    assert result.claims[0].verified is False


def test_json_schema_path_reasks_once_on_malformed_json_then_succeeds(chat_index, cfg, mock_llm):
    chunks = list(chat_index.chunks)
    target = chunks[0]
    mock_llm.queue(LlmResult(
        text="not valid json {{{", blocks=None,
        usage={"input_tokens": 10, "output_tokens": 5}, model="m", finish_reason="stop",
    ))
    mock_llm.queue(_json_llm_result({
        "answer_supported": True,
        "claims": [{"text": target.text, "citations": [target.citation_id]}],
        "insufficient_evidence": False, "refusal_reason": None,
    }))

    result = grounding.ground_json_schema("some question", chunks, cfg)

    assert len(mock_llm.calls) == 2
    assert result.claims
    assert result.claims[0].citations[0].citation_id == target.citation_id


def test_json_schema_path_flags_when_response_shape_is_wrong(chat_index, cfg, mock_llm):
    chunks = list(chat_index.chunks)
    # a syntactically valid JSON array (not an object) — a realistic malformed-shape response.
    mock_llm.queue(LlmResult(
        text="[1, 2, 3]", blocks=None,
        usage={"input_tokens": 10, "output_tokens": 5}, model="m", finish_reason="stop",
    ))
    mock_llm.queue(LlmResult(
        text="[1, 2, 3]", blocks=None,
        usage={"input_tokens": 10, "output_tokens": 5}, model="m", finish_reason="stop",
    ))

    result = grounding.ground_json_schema("some question", chunks, cfg)

    assert len(mock_llm.calls) == 2  # re-asked once, then flagged rather than crashing
    assert not result.claims
    assert result.unresolved_ids


def test_response_format_and_documents_are_mutually_exclusive(cfg, monkeypatch):
    from delta_chat.chat import llm as llm_mod

    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    with pytest.raises(ValueError):
        llm_mod._call_provider(
            "anthropic", "claude-sonnet-5", [{"role": "user", "content": "hi"}],
            documents=[{"citation_id": "x", "text": "y"}],
            response_format={"type": "json_object"},
            cfg=cfg, timeout_s=30,
        )
