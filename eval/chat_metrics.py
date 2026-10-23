"""Grounded-chat quality metrics. Two families:

- **Model-graded, non-gating**: faithfulness and answer-relevancy — backed by an injected `judge`
  (never called directly here; tests pass a `mock_judge` so this module never hits the network). A
  real deployment can swap RAGAS/DeepEval in behind the same judge interface without changing this
  module.
- **Bespoke, deterministic**: citation VALIDITY (pure code: does the id resolve against that
  answer's own `used_chunks`? — catches a hallucinated id), context PRECISION/RECALL (pure code:
  set-overlap between `used_chunks` and `expected_citation_ids` — no chunk text is available to
  `score_chat` to do a real judge-graded relevance check, so this is a coarse citation-id-overlap
  proxy, not a RAGAS-style semantic score), and ABSTENTION correctness (precision/recall of
  `state == "Rejected"` over the answerable/unanswerable gold split). Citation SUPPORT is the one
  judge-touched bespoke metric (does the cited span entail the claim? — catches a wrong-span cite
  even when the id resolves).

`preds` and `gold` are aligned 1:1 by position — each `ChatAnswer` is the result of running the
corresponding `QAGold.question` through `chat.answer.answer`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from delta_chat.chat.types import ChatAnswer

from .gold import QAGold


class Judge(Protocol):
    def entails(self, claim_text: str, citation_id: str) -> bool: ...
    def relevancy(self, answer_text: str, question: str) -> float: ...


def _prf1(tp: int, fp: int, fn: int) -> dict:
    precision = tp / (tp + fp) if (tp + fp) else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) else (1.0 if fp == 0 else 0.0)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


@dataclass
class ChatScore:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float
    citation_validity: float
    citation_support: float
    abstention: dict
    per_class: dict = field(default_factory=dict)


def score_chat(preds: list[ChatAnswer], gold: list[QAGold], judge: Judge) -> ChatScore:
    """Compute faithfulness/relevancy/context P/R (via `judge`), deterministic citation VALIDITY
    (id resolves against that answer's own used_chunks), judge-based citation SUPPORT (cited span
    entails the claim), and abstention P/R of INSUFFICIENT_EVIDENCE over answerable/unanswerable
    gold. A claim counts toward faithfulness's numerator iff it has >=1 citation and every one of
    its citations both resolves and entails (mirrors `chat.answer._verify_claim`'s definition)."""
    if len(preds) != len(gold):
        raise ValueError(f"preds ({len(preds)}) and gold ({len(gold)}) must be aligned 1:1 by question")

    total_claims = supported_claims = 0
    total_citations = valid_citations = 0
    support_checked = support_ok = 0
    relevancy_scores: list[float] = []
    context_precisions: list[float] = []
    context_recalls: list[float] = []
    per_class: dict[str, dict[str, int]] = {"delta": {"total": 0, "valid": 0}, "doc_span": {"total": 0, "valid": 0}}
    abstain_tp = abstain_fp = abstain_fn = abstain_tn = 0

    for pred, g in zip(preds, gold):
        abstained = pred.state == "Rejected"
        should_abstain = not g.answerable
        if should_abstain and abstained:
            abstain_tp += 1
        elif should_abstain and not abstained:
            abstain_fn += 1  # over-answered an unanswerable question
        elif not should_abstain and abstained:
            abstain_fp += 1
        else:
            abstain_tn += 1

        if g.answerable:
            used = set(pred.used_chunks)
            expected = set(g.expected_citation_ids)
            overlap = len(used & expected)
            context_precisions.append(overlap / len(used) if used else 0.0)
            context_recalls.append(overlap / len(expected) if expected else 1.0)
            relevancy_scores.append(judge.relevancy(pred.text, g.question))

        for claim in pred.claims:
            total_claims += 1
            claim_ok = bool(claim.citations)
            for cit in claim.citations:
                total_citations += 1
                resolves = cit.citation_id in pred.used_chunks
                entails = judge.entails(claim.text, cit.citation_id) if resolves else False
                support_checked += 1
                if resolves:
                    valid_citations += 1
                if entails:
                    support_ok += 1
                claim_ok = claim_ok and resolves and entails

                cls = "delta" if cit.source_type == "delta" else "doc_span"
                per_class[cls]["total"] += 1
                if resolves:
                    per_class[cls]["valid"] += 1
            if claim_ok:
                supported_claims += 1

    return ChatScore(
        faithfulness=supported_claims / total_claims if total_claims else 1.0,
        answer_relevancy=sum(relevancy_scores) / len(relevancy_scores) if relevancy_scores else 0.0,
        context_precision=sum(context_precisions) / len(context_precisions) if context_precisions else 0.0,
        context_recall=sum(context_recalls) / len(context_recalls) if context_recalls else 0.0,
        citation_validity=valid_citations / total_citations if total_citations else 1.0,
        citation_support=support_ok / support_checked if support_checked else 1.0,
        abstention=_prf1(abstain_tp, abstain_fp, abstain_fn),
        per_class=per_class,
    )
