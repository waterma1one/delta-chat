"""Tests for eval/chat_metrics.py — faithfulness, citation validity/support, abstention P/R.
Judge is always mocked (`mock_judge` from conftest); no network."""

from __future__ import annotations

from delta_chat.chat.types import ChatAnswer, Citation, Claim

from eval.chat_metrics import score_chat
from eval.gold import QAGold


def test_faithfulness_equals_supported_over_total_claims_on_hand_built_set(mock_judge):
    preds = [
        ChatAnswer(
            state="Grounded", text="a",
            claims=[
                Claim(text="claim one", citations=[Citation("known:1", "doc_a", 0, None, None)]),
                Claim(text="claim two", citations=[Citation("doc_a:0:notes:2", "doc_a", 0, None, None)]),
            ],
            used_chunks=["known:1", "doc_a:0:notes:2"], refusal_reason=None,
        ),
    ]
    gold = [QAGold(question="q", reference_answer="a", expected_citation_ids=["known:1"],
                    answerable=True, citation_class="doc_span", pair_id="p1")]

    score = score_chat(preds, gold, mock_judge)

    # claim one: resolves + entails (mock_judge only refuses doc_a:0:notes:2) -> supported.
    # claim two: resolves but mock_judge.entails() returns False for exactly this id -> unsupported.
    assert score.faithfulness == 0.5


def test_citation_validity_is_pure_code_and_catches_hallucinated_id(mock_judge):
    preds = [
        ChatAnswer(
            state="Grounded", text="a",
            claims=[Claim(text="a", citations=[Citation("doc_a:HALLUCINATED", "doc_a", 0, None, None)])],
            used_chunks=["doc_a:real"], refusal_reason=None,
        ),
    ]
    gold = [QAGold(question="q", reference_answer="a", expected_citation_ids=["doc_a:real"],
                    answerable=True, citation_class="doc_span", pair_id="p1")]

    score = score_chat(preds, gold, mock_judge)

    assert score.citation_validity == 0.0  # the only citation doesn't resolve against used_chunks


def test_citation_support_flags_claim_whose_cited_span_does_not_entail_it(mock_judge):
    preds = [
        ChatAnswer(
            state="Grounded", text="a",
            claims=[Claim(text="a", citations=[Citation("doc_a:0:notes:2", "doc_a", 0, None, None)])],
            used_chunks=["doc_a:0:notes:2"], refusal_reason=None,  # resolves...
        ),
    ]
    gold = [QAGold(question="q", reference_answer="a", expected_citation_ids=["doc_a:0:notes:2"],
                    answerable=True, citation_class="doc_span", pair_id="p1")]

    score = score_chat(preds, gold, mock_judge)

    assert score.citation_validity == 1.0  # id resolves...
    assert score.citation_support == 0.0  # ...but mock_judge says this exact id doesn't entail


def test_abstention_correctness_precision_recall(mock_judge):
    preds = [
        ChatAnswer(state="Rejected", text="", claims=[], used_chunks=[], refusal_reason="INSUFFICIENT_EVIDENCE"),
        ChatAnswer(state="Grounded", text="a", claims=[], used_chunks=[], refusal_reason=None),
        ChatAnswer(state="Rejected", text="", claims=[], used_chunks=[], refusal_reason="INSUFFICIENT_EVIDENCE"),
    ]
    gold = [
        QAGold(question="q1", reference_answer="", expected_citation_ids=[], answerable=False,
               citation_class="doc_span", pair_id="p1"),  # correctly abstained -> TP
        QAGold(question="q2", reference_answer="a", expected_citation_ids=[], answerable=True,
               citation_class="doc_span", pair_id="p1"),  # correctly answered -> TN (not counted)
        QAGold(question="q3", reference_answer="a", expected_citation_ids=[], answerable=True,
               citation_class="doc_span", pair_id="p1"),  # WRONGLY abstained on an answerable Q -> FP
    ]

    score = score_chat(preds, gold, mock_judge)

    assert score.abstention["precision"] == 0.5  # 1 TP / (1 TP + 1 FP)
    assert score.abstention["recall"] == 1.0  # 1 TP / (1 TP + 0 FN)


def test_full_fixture_end_to_end(chat_preds, qa_gold, mock_judge):
    score = score_chat(chat_preds, qa_gold, mock_judge)

    assert 0.0 < score.faithfulness < 1.0
    assert 0.0 < score.citation_validity < 1.0
    assert 0.0 < score.citation_support < 1.0
    assert score.abstention["recall"] == 1.0  # the one unanswerable question was correctly abstained
    assert set(score.per_class) >= {"delta", "doc_span"}
