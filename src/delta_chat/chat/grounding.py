"""Two config-selected grounding paths plus the sufficiency gate that decides whether to run
generation at all.

`config.chat.grounding`:
  - "json"                -> `ground_json_schema` (default, provider-agnostic)
  - "anthropic_citations"  -> `ground_anthropic_citations` (Anthropic Citations API only)

Both produce a `GroundingResult{claims, ...}`: claims whose citations resolve to a source id.
Neither path fabricates a citation — an unresolved id is either re-asked once (JSON path) or
dropped and recorded in `unresolved_ids`, never silently accepted.
"""

from __future__ import annotations

import json

from .index import tokenize
from .types import Chunk, Citation, Claim, GroundingResult

_JSON_SYSTEM_PROMPT = (
    "You are a grounded QA assistant for engineering-drawing revisions. Answer only from the "
    "provided sources; every claim must cite at least one source id from the bracketed list. "
    "If the sources do not contain the answer, set insufficient_evidence=true and refusal_reason "
    "to what is missing. Respond with JSON only, matching exactly this schema:\n"
    '{"answer_supported": bool, "claims": [{"text": str, "citations": [chunk_id, ...]}], '
    '"insufficient_evidence": bool, "refusal_reason": str | null}'
)

_CITATIONS_SYSTEM_PROMPT = (
    "You are a grounded QA assistant for engineering-drawing revisions. Answer only from the "
    "provided documents and cite every claim. If the documents do not contain the answer, say so "
    "plainly and cite nothing."
)


def _source_listing(chunks: list[Chunk]) -> str:
    return "\n".join(f"[{c.citation_id}] {c.text}" for c in chunks)


def _score_chunks(query: str, chunks: list[Chunk]) -> list[float]:
    """Query-token coverage ratio per chunk, in [0, 1] — how much of the query each chunk covers.

    BM25 over a tiny (top-k-sized) mini-corpus is not used here: its IDF term is calibrated for a
    real corpus and misbehaves (even goes negative) for the handful of documents a single query's
    top-k produces — exactly the kind of raw, corpus-dependent score this gate must not threshold.
    Coverage ratio is bounded and well-defined for any pool size, including a single chunk.
    """
    q_tokens = set(tokenize(query))
    if not q_tokens:
        return [0.0] * len(chunks)
    return [len(q_tokens & set(tokenize(c.text))) / len(q_tokens) for c in chunks]


def check_sufficiency(query: str, chunks: list[Chunk], cfg) -> tuple[bool, str | None]:
    """Normalized top-k score-margin gate (+ optional entailment check). Never thresholds a raw
    score directly: a flat top-k (no chunk stands out) abstains even when absolute coverage looks
    strong, because a flat top-k means nothing distinguishes the best answer from the rest."""
    if not chunks:
        return False, "no relevant chunks were retrieved for this query"

    scores = _score_chunks(query, chunks)
    if len(scores) == 1:
        # nothing to compare against: a single retrieved chunk is "sufficient" iff it has any
        # lexical overlap with the query at all (min-max normalization is undefined for n=1).
        margin = 1.0 if scores[0] > 0 else 0.0
    else:
        lo, hi = min(scores), max(scores)
        normalized = [0.0] * len(scores) if hi <= lo else [(s - lo) / (hi - lo) for s in scores]
        ranked = sorted(normalized, reverse=True)
        margin = ranked[0] - ranked[1]

    margin_min = float(cfg.get("chat.sufficiency.margin_min", 0.15))
    if margin < margin_min:
        return False, (
            f"retrieved evidence has no clear best match (normalized score margin "
            f"{margin:.2f} < {margin_min:.2f}) — top candidates are too similar in relevance to trust"
        )

    if cfg.get("chat.sufficiency.use_entailment_check", True):
        return _entailment_check(query, chunks, cfg)
    return True, None


def _entailment_check(query: str, chunks: list[Chunk], cfg) -> tuple[bool, str | None]:
    from . import llm as _llm  # module import so tests can monkeypatch `llm.complete`

    messages = [
        {
            "role": "system",
            "content": "Answer only 'yes' or 'no': does the given context contain enough "
            "information to answer the question?",
        },
        {"role": "user", "content": f"Question: {query}\n\nContext:\n{_source_listing(chunks)}"},
    ]
    result = _llm.complete(messages, model=cfg.get("llm.model"), cfg=cfg)
    if result.text.strip().lower().startswith("yes"):
        return True, None
    return False, "sufficiency check: retrieved context judged insufficient to answer the question"


def _resolve_citations(citation_ids: list[str], by_id: dict[str, Chunk]) -> tuple[list[Citation], list[str]]:
    resolved, unresolved = [], []
    for cid in citation_ids:
        chunk = by_id.get(cid)
        if chunk is None:
            unresolved.append(cid)
            continue
        resolved.append(
            Citation(
                citation_id=chunk.citation_id, source_type=chunk.source_type,
                page=chunk.page, sheet=chunk.sheet, bbox=chunk.bbox,
            )
        )
    return resolved, unresolved


def _parse_json_payload(text: str, by_id: dict[str, Chunk]) -> tuple[dict | None, list[str]]:
    """Parse the model's JSON response. Guards against syntactically-valid-but-wrong-shaped JSON
    (a bare list/string/number, or a `claims` entry that isn't an object) so a malformed model
    response triggers the re-ask path instead of an uncaught AttributeError crashing the request."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return None, ["<malformed-json: not valid JSON>"]
    if not isinstance(data, dict):
        return None, ["<malformed-json: response is not a JSON object>"]
    claims = data.get("claims", [])
    if not isinstance(claims, list):
        return None, ["<malformed-json: 'claims' is not a list>"]
    unresolved: list[str] = []
    for claim in claims:
        if not isinstance(claim, dict):
            unresolved.append("<malformed-claim: not an object>")
            continue
        for cid in claim.get("citations", []) or []:
            if cid not in by_id:
                unresolved.append(cid)
    return data, unresolved


def _grounding_result_from_json(
    data: dict, by_id: dict[str, Chunk], usage: dict, *, extra_unresolved: list[str] | None = None
) -> GroundingResult:
    claims: list[Claim] = []
    unresolved = list(extra_unresolved or [])
    for c in data.get("claims", []) or []:
        if not isinstance(c, dict):
            continue
        citations, bad = _resolve_citations(c.get("citations", []) or [], by_id)
        unresolved.extend(bad)
        if citations:  # a claim surviving with zero resolvable citations is dropped, not fabricated
            claims.append(Claim(text=c.get("text", ""), citations=citations))
    return GroundingResult(
        claims=claims,
        raw_text=json.dumps(data),
        usage=usage,
        unresolved_ids=unresolved,
        insufficient_evidence=bool(data.get("insufficient_evidence", False)),
        refusal_reason=data.get("refusal_reason"),
    )


def ground_json_schema(query: str, chunks: list[Chunk], cfg) -> GroundingResult:
    """Model returns structured JSON citing chunk ids. On parse failure or an unresolvable id,
    re-ask exactly once; if it still fails, flag/downgrade rather than silently accept."""
    from . import llm as _llm

    by_id = {c.citation_id: c for c in chunks}
    messages = [
        {"role": "system", "content": _JSON_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}\n\nSources:\n{_source_listing(chunks)}"},
    ]
    response_format = {"type": "json_object"}

    data, unresolved, result = None, ["<no-response>"], None
    for attempt in range(2):  # initial attempt + exactly one re-ask
        result = _llm.complete(messages, model=cfg.get("llm.model"), response_format=response_format, cfg=cfg)
        data, unresolved = _parse_json_payload(result.text, by_id)
        if data is not None and not unresolved:
            return _grounding_result_from_json(data, by_id, result.usage)
        if attempt == 0:
            messages = messages + [
                {"role": "assistant", "content": result.text},
                {
                    "role": "user",
                    "content": (
                        f"Your previous response was malformed or cited an unknown source id "
                        f"{unresolved!r}. Re-answer using ONLY the exact bracketed ids from Sources, "
                        "as valid JSON matching the schema."
                    ),
                },
            ]

    # Still failing after the re-ask -> flag/downgrade rather than silently accept.
    if data is not None:
        return _grounding_result_from_json(data, by_id, result.usage, extra_unresolved=unresolved)
    return GroundingResult(claims=[], raw_text=result.text, usage=result.usage, unresolved_ids=unresolved)


def ground_anthropic_citations(query: str, chunks: list[Chunk], cfg) -> GroundingResult:
    """Pass each chunk as an Anthropic custom-content block; parse the interleaved response blocks
    into claims + citation pointers. Cannot combine with strict JSON output — never pass
    `response_format` alongside `documents` (enforced in `llm._call_provider`)."""
    from . import llm as _llm

    documents = [{"citation_id": c.citation_id, "text": c.text} for c in chunks]
    messages = [
        {"role": "system", "content": _CITATIONS_SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {query}"},
    ]
    result = _llm.complete(messages, model=cfg.get("llm.model"), documents=documents, cfg=cfg)
    return _parse_citation_blocks(result.text, result.usage, result.blocks or [], chunks)


def _parse_citation_blocks(
    text: str, usage: dict, blocks: list[dict], chunks_by_index: list[Chunk]
) -> GroundingResult:
    claims: list[Claim] = []
    unresolved: list[str] = []
    for block in blocks:
        block_text = block.get("text", "")
        citations: list[Citation] = []
        for cit in block.get("citations", []) or []:
            idx = cit.get("document_index")
            chunk = chunks_by_index[idx] if isinstance(idx, int) and 0 <= idx < len(chunks_by_index) else None
            if chunk is None:
                unresolved.append(str(cit.get("document_title") or f"doc#{idx}"))
                continue
            citations.append(
                Citation(
                    citation_id=chunk.citation_id, source_type=chunk.source_type,
                    page=chunk.page, sheet=chunk.sheet, bbox=chunk.bbox,
                )
            )
        if block_text.strip() and citations:
            claims.append(Claim(text=block_text, citations=citations))
    raw_text = " ".join(b.get("text", "") for b in blocks) or text
    return GroundingResult(claims=claims, raw_text=raw_text, usage=usage, unresolved_ids=unresolved)


GROUNDING_PATHS = {
    "json": ground_json_schema,
    "anthropic_citations": ground_anthropic_citations,
}
