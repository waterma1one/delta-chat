"""Retrieval over document + delta pools — BM25 (+ optional hybrid RRF) with a deterministic intent
router. No vector DB: two documents plus a delta report is tiny, and BM25 is strong on exact
instrument tags / line numbers (the tokenizer keeps hyphens so "FV-9040" stays one token).
"""

from __future__ import annotations

import math
import re
from typing import Literal

from rank_bm25 import BM25Okapi

from ..canonical.model import Delta, Document
from ..observability import metrics as _metrics
from ..observability import tracing
from .types import ChatIndex, Chunk, SourceType

_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-\./]*")

_DELTA_KEYWORDS = (
    "what changed", "what's changed", "whats changed", "difference", "differences",
    "between rev", "changed", "changes", "change", "delta", "added", "removed", "modified",
    # surface forms a user actually types for the four change types. Without these, e.g. "Which
    # note was deleted?" scored as a pure content question (on "which") and never reached the
    # delta pool that holds its answer.
    "deleted", "delete", "add", "remove", "moved", "move", "renumbered", "revised", "new in rev",
)
_DOC_KEYWORDS = ("what is", "which", "setpoint", "value of", "spec", "rating", "how many", "where is")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall((text or "").lower())


def _phrase_present(phrase: str, query_lc: str, query_tokens: set[str]) -> bool:
    """Match a keyword against the query. Single-word keys must match a WHOLE token (so 'move'
    doesn't fire on 'remove'/'movement' and 'add' doesn't fire on 'address'); multi-word phrases
    ('what changed') are matched as substrings since they can't be a single token."""
    return phrase in query_tokens if " " not in phrase else phrase in query_lc


def route_intent(query: str) -> Literal["delta", "doc", "both"]:
    """Deterministic keyword router (no LLM). 'what changed'-style queries -> delta pool; content
    questions ('what is the setpoint of...') -> doc pools; ambiguous (or neither) -> both."""
    q = query.lower()
    qtok = set(tokenize(query))
    is_delta = any(_phrase_present(k, q, qtok) for k in _DELTA_KEYWORDS)
    is_doc = any(_phrase_present(k, q, qtok) for k in _DOC_KEYWORDS)
    if is_delta and not is_doc:
        return "delta"
    if is_doc and not is_delta:
        return "doc"
    return "both"


def build_index(doc_a: Document, doc_b: Document, delta: Delta, cfg=None) -> ChatIndex:
    """Chunk both documents (one chunk per Item) and the delta report (one chunk per Change) into
    a single tagged, citable pool."""
    chunks: list[Chunk] = []
    chunks.extend(_chunk_document(doc_a, "doc_a"))
    chunks.extend(_chunk_document(doc_b, "doc_b"))
    chunks.extend(_chunk_delta(delta))
    return ChatIndex(chunks=chunks)


def _chunk_document(doc: Document, source_type: SourceType) -> list[Chunk]:
    out = []
    for item in doc.items:
        text = item.text or ""
        out.append(
            Chunk(
                citation_id=f"{source_type}:{item.id}",
                source_type=source_type,
                text=text,
                pid=doc.pid,
                page=item.page,
                sheet=item.sheet,
                bbox=item.bbox,
                char_range=(0, len(text)),
                meta={"item_type": item.type, "region": item.region},
            )
        )
    return out


def _chunk_delta(delta: Delta) -> list[Chunk]:
    out = []
    for ch in delta.changes:
        parts = [ch.description]
        if ch.old is not None:
            parts.append(f"old: {ch.old}")
        if ch.new is not None:
            parts.append(f"new: {ch.new}")
        text = " | ".join(parts)
        out.append(
            Chunk(
                citation_id=f"delta:{ch.id}",
                source_type="delta",
                text=text,
                pid=f"{delta.pid_a}->{delta.pid_b}",
                page=ch.page,
                sheet=ch.sheet,
                bbox=ch.bbox,
                char_range=(0, len(text)),
                meta={"change_type": ch.type},
            )
        )
    return out


def _select_pool(index: ChatIndex, route: str) -> list[Chunk]:
    if route == "delta":
        pool = [c for c in index.chunks if c.source_type == "delta"]
    elif route == "doc":
        pool = [c for c in index.chunks if c.source_type in ("doc_a", "doc_b")]
    else:
        pool = list(index.chunks)
    return pool or list(index.chunks)  # empty pool (e.g. no delta chunks) -> fall back to everything


def _bm25_rank(pool: list[Chunk], query: str) -> list[tuple[Chunk, float]]:
    if not pool:
        return []
    corpus = [tokenize(c.text) for c in pool]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(tokenize(query))
    return sorted(zip(pool, scores), key=lambda t: t[1], reverse=True)


def _dense_rank(pool: list[Chunk], query: str) -> list[tuple[Chunk, float]]:
    """Dependency-free 'dense' signal: cosine similarity over term-frequency vectors. Stands in for
    embeddings so hybrid mode works without a model download; fused with BM25 via RRF."""
    if not pool:
        return []

    def _vec(tokens: list[str]) -> dict[str, int]:
        v: dict[str, int] = {}
        for t in tokens:
            v[t] = v.get(t, 0) + 1
        return v

    def _cosine(a: dict[str, int], b: dict[str, int]) -> float:
        common = set(a) & set(b)
        dot = sum(a[t] * b[t] for t in common)
        na = math.sqrt(sum(v * v for v in a.values()))
        nb = math.sqrt(sum(v * v for v in b.values()))
        return dot / (na * nb) if na and nb else 0.0

    q_vec = _vec(tokenize(query))
    return sorted(
        ((c, _cosine(q_vec, _vec(tokenize(c.text)))) for c in pool),
        key=lambda t: t[1], reverse=True,
    )


def _rrf_fuse(rankings: list[list[tuple[Chunk, float]]], rrf_k: int) -> list[tuple[Chunk, float]]:
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, (chunk, _score) in enumerate(ranking, start=1):
            scores[chunk.citation_id] = scores.get(chunk.citation_id, 0.0) + 1.0 / (rrf_k + rank)
            by_id[chunk.citation_id] = chunk
    fused = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return [(by_id[cid], score) for cid, score in fused]


def retrieve(index: ChatIndex, query: str, k: int, cfg) -> list[Chunk]:
    """Intent-route, then score (BM25, or RRF-fused BM25+dense when hybrid). Emits a `retrieval`
    span with attrs retrieval.hits, retrieval.top_k."""
    pool = _select_pool(index, route_intent(query))
    mode = cfg.get("chat.retrieval", "bm25") if cfg is not None else "bm25"
    with tracing.stage("retrieval") as rec:
        if mode == "hybrid":
            rrf_k = int(cfg.get("chat.rrf_k", 60))
            ranked = _rrf_fuse([_bm25_rank(pool, query), _dense_rank(pool, query)], rrf_k)
        else:
            ranked = _bm25_rank(pool, query)
        hits = [chunk for chunk, _score in ranked[:k]]
        rec.attrs["retrieval.hits"] = len(hits)
        rec.attrs["retrieval.top_k"] = k
    _metrics.observe_retrieval_hits(len(hits))
    return hits
