"""Typed loaders for the gold datasets section-04 produces (eval/datasets/*.jsonl). Mirrors the
JSONL record shapes; does not duplicate section-04's writer.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from delta_chat.canonical.model import BBox


@dataclass
class GoldChange:
    change_id: str
    type: str
    page: int
    sheet: str | None
    bbox: BBox
    old: str | None
    new: str | None
    description: str
    hard_slice: bool
    pair_id: str
    format: str
    item_type: str = "tag"


@dataclass
class QAGold:
    question: str
    reference_answer: str
    expected_citation_ids: list[str]
    answerable: bool  # False -> should abstain (INSUFFICIENT_EVIDENCE)
    citation_class: str  # "delta" | "doc_span" — both classes required by rubric
    pair_id: str


@dataclass
class RetrievalGold:
    query: str
    relevant_chunk_ids: list[str]
    pair_id: str


def _read_jsonl(path: str | Path) -> list[dict]:
    with Path(path).open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_delta_gold(path: str | Path) -> list[GoldChange]:
    out = []
    for r in _read_jsonl(path):
        bbox = r.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(
                f"malformed gold record {r.get('change_id', '<unknown>')!r} in {path}: "
                f"'bbox' must be a 4-element list, got {bbox!r}"
            )
        x0, y0, x1, y1 = bbox
        out.append(GoldChange(
            change_id=r["change_id"], type=r["type"], page=r["page"], sheet=r.get("sheet"),
            bbox=BBox(x0, y0, x1, y1), old=r.get("old"), new=r.get("new"),
            description=r["description"], hard_slice=bool(r.get("hard_slice", False)),
            pair_id=r["pair_id"], format=r.get("format", "pdf_native"),
            item_type=r.get("item_type", "tag"),
        ))
    return out


def load_qa_gold(path: str | Path) -> list[QAGold]:
    return [
        QAGold(
            question=r["question"], reference_answer=r.get("reference_answer", ""),
            expected_citation_ids=list(r.get("expected_citation_ids", [])),
            answerable=bool(r["answerable"]), citation_class=r.get("citation_class", "doc_span"),
            pair_id=r.get("pair_id", ""),
        )
        for r in _read_jsonl(path)
    ]


def load_retrieval_gold(path: str | Path) -> list[RetrievalGold]:
    return [
        RetrievalGold(
            query=r["query"], relevant_chunk_ids=list(r["relevant_chunk_ids"]), pair_id=r.get("pair_id", "")
        )
        for r in _read_jsonl(path)
    ]


def qa_gold_as_retrieval_gold(qa: list[QAGold]) -> list[RetrievalGold]:
    """section-04 does not produce a dedicated retrieval_gold.jsonl — derive retrieval gold from
    the QA set's expected_citation_ids (the chunks that should be retrieved to answer each
    answerable question)."""
    return [
        RetrievalGold(query=q.question, relevant_chunk_ids=list(q.expected_citation_ids), pair_id=q.pair_id)
        for q in qa if q.answerable
    ]


def dataset_hash(paths: list[str | Path]) -> str:
    """Stable sha256 over the raw JSONL bytes of the given files, recorded in the scorecard for
    regression provenance. Paths are sorted so hash order doesn't depend on caller iteration order."""
    h = hashlib.sha256()
    for p in sorted(str(p) for p in paths):
        h.update(Path(p).read_bytes())
    return h.hexdigest()
