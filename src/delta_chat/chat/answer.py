"""Orchestrator: retrieve -> sufficiency gate -> ground (config-selected path) -> verify citations
-> `ChatAnswer`.

State machine:
  - sufficiency gate fails, or the model itself reports insufficient evidence -> Rejected
    (claims=[] — grounding never ran, so there is nothing to show)
  - every claim resolves + its cited span entails it                          -> Grounded
  - at least one but not all claims verify                                    -> Partial
  - zero claims verify (e.g. every citation points at a non-entailing span)   -> Rejected
    (this is what catches a hallucinated/wrong-span citation — it never counts as Grounded)

Once grounding has run, `ChatAnswer.claims` always carries every claim the model produced — verified
and unverified alike, each tagged via `Claim.verified` — so section-10's eval harness (faithfulness,
citation validity, abstention P/R) can see what was emitted, not just what survived.
"""

from __future__ import annotations

from rapidfuzz import fuzz

from ..observability import tracing
from . import grounding as _grounding
from . import index as _index
from .types import ChatAnswer, ChatIndex, Claim

_ENTAILMENT_THRESHOLD = 45.0  # rapidfuzz partial_ratio (0-100); cheap lexical-overlap stand-in for NLI


def _entails(claim_text: str, source_text: str) -> bool:
    return fuzz.partial_ratio(claim_text.lower(), source_text.lower()) >= _ENTAILMENT_THRESHOLD


def _verify_claim(claim: Claim, index: ChatIndex) -> bool:
    if not claim.citations:
        return False
    for citation in claim.citations:
        chunk = index.resolve(citation.citation_id)
        if chunk is None or not _entails(claim.text, chunk.text):
            return False
    return True


def answer(query: str, index: ChatIndex, cfg) -> ChatAnswer:
    with tracing.stage("answer") as rec:
        top_k = int(cfg.get("chat.top_k", 8))
        chunks = _index.retrieve(index, query, top_k, cfg)
        used_chunks = [c.citation_id for c in chunks]

        sufficient, reason = _grounding.check_sufficiency(query, chunks, cfg)
        if not sufficient:
            rec.attrs["chat.state"] = "Rejected"
            return ChatAnswer(
                state="Rejected", text="", claims=[], used_chunks=used_chunks, refusal_reason=reason
            )

        ground_fn = _grounding.GROUNDING_PATHS.get(cfg.get("chat.grounding", "json"))
        if ground_fn is None:
            raise ValueError(
                f"unknown chat.grounding {cfg.get('chat.grounding', 'json')!r}; "
                f"expected one of {sorted(_grounding.GROUNDING_PATHS)}"
            )
        result = ground_fn(query, chunks, cfg)

        if result.insufficient_evidence or not result.claims:
            rec.attrs["chat.state"] = "Rejected"
            return ChatAnswer(
                state="Rejected", text="", claims=[], used_chunks=used_chunks,
                refusal_reason=result.refusal_reason or "model reported insufficient evidence",
            )

        for claim in result.claims:
            claim.verified = _verify_claim(claim, index)
        verified_count = sum(c.verified for c in result.claims)

        if verified_count == len(result.claims):
            state, refusal_reason = "Grounded", None
        elif verified_count:
            state, refusal_reason = "Partial", None
        else:
            state = "Rejected"
            refusal_reason = "INSUFFICIENT_EVIDENCE: cited spans did not entail the claim text"

        rec.attrs["chat.state"] = state
        text = " ".join(c.text for c in result.claims if c.verified)
        return ChatAnswer(
            state=state, text=text, claims=result.claims, used_chunks=used_chunks,
            refusal_reason=refusal_reason,
        )
