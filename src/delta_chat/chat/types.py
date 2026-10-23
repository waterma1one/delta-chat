"""Shared dataclasses for the grounded-chat subsystem — the public contract the API and eval layers consume.

Keep this module free of retrieval/LLM/grounding logic: it exists so ``answer.py`` can define
``ChatAnswer`` without importing ``index.py``'s BM25 machinery, and so they can import just the
shapes they need.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from ..canonical.model import BBox

SourceType = Literal["doc_a", "doc_b", "delta"]


@dataclass
class Chunk:
    citation_id: str  # stable, resolvable id, e.g. "doc_a:0:notes:12" or "delta:CHG-0007"
    source_type: SourceType
    text: str
    pid: str | None  # source document pid (None for delta chunks, or the "A->B" pair id)
    page: int | None
    sheet: str | None
    bbox: BBox | None  # location for UI highlight; None where not applicable
    char_range: tuple[int, int] | None  # char offsets within source, for Citations-API char_location
    meta: dict = field(default_factory=dict)


@dataclass
class ChatIndex:
    chunks: list[Chunk]
    by_id: dict[str, Chunk] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.by_id:
            self.by_id = {c.citation_id: c for c in self.chunks}

    def resolve(self, citation_id: str) -> Chunk | None:
        return self.by_id.get(citation_id)


@dataclass
class LlmResult:
    text: str
    blocks: list | None  # interleaved content blocks (Anthropic Citations); None otherwise
    usage: dict  # {"input_tokens": int, "output_tokens": int}
    model: str
    finish_reason: str | None


@dataclass
class Citation:
    citation_id: str
    source_type: SourceType
    page: int | None
    sheet: str | None
    bbox: BBox | None


@dataclass
class Claim:
    text: str
    citations: list[Citation] = field(default_factory=list)
    verified: bool = True  # False when the cited span was checked and did not entail the claim


@dataclass
class ChatAnswer:
    state: Literal["Grounded", "Partial", "Rejected"]
    text: str
    claims: list[Claim]
    used_chunks: list[str]  # citation_ids actually retrieved/considered
    refusal_reason: str | None = None  # set when state == "Rejected"


@dataclass
class GroundingResult:
    """Common shape both grounding paths (JSON-schema, Anthropic Citations) produce."""

    claims: list[Claim]
    raw_text: str
    usage: dict
    unresolved_ids: list[str] = field(default_factory=list)
    insufficient_evidence: bool = False
    refusal_reason: str | None = None
