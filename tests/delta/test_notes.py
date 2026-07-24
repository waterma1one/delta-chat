"""Tests for delta/notes.py — ordered-list LCS/Myers diff + calibrated semantic gate."""

from __future__ import annotations

from unittest.mock import Mock

from delta_chat.delta.align import normalize_text
from delta_chat.delta.notes import _MARKER_RE, diff_notes


def _note_item(id_, text, y0, *, source_rev="A"):
    from tests.delta.conftest import make_item

    return make_item(
        id_, text, source_rev=source_rev, type="text", region="notes", bbox=(0.1, y0, 0.5, y0 + 0.02)
    )


def _word(id_, text, x0, y0, *, source_rev="A"):
    from tests.delta.conftest import make_item

    return make_item(
        id_, text, source_rev=source_rev, type="text", region="notes", bbox=(x0, y0, x0 + 0.05, y0 + 0.02)
    )


def test_marker_regex_does_not_misparse_decimal_content():
    assert _MARKER_RE.match('1.5" MINIMUM CLEARANCE') is None
    assert _MARKER_RE.match("1. SUCTION SCRUBBER") is not None


def test_row_clustering_joins_multi_word_lines_like_real_ingestion(cfg):
    # Native-PDF ingestion emits one Item per word (not per line), out of any guaranteed x-order --
    # this exercises _cluster_rows/_build_lines rather than a hand-assembled single-Item-per-line
    # shortcut.
    notes_a = [
        _word("a3", "SCRUBBER", 0.23, 0.10),
        _word("a1", "1.", 0.10, 0.10),
        _word("a2", "SUCTION", 0.15, 0.10),
        _word("a6", "COOLER", 0.23, 0.15),
        _word("a4", "2.", 0.10, 0.15),
        _word("a5", "DISCHARGE", 0.15, 0.15),
    ]
    notes_b = [
        _word("b1", "1.", 0.10, 0.10, source_rev="B"),
        _word("b2", "SUCTION", 0.15, 0.10, source_rev="B"),
        _word("b3", "SCRUBBER", 0.23, 0.10, source_rev="B"),
    ]

    changes = diff_notes(notes_a, notes_b, cfg)

    assert len(changes) == 1
    assert changes[0].type == "removed"
    assert changes[0].old == "2. DISCHARGE COOLER"


def test_lcs_diff_produces_added_deleted_renumbered(cfg):
    notes_a = [
        _note_item("a1", "1. SUCTION SCRUBBER", 0.10),
        _note_item("a2", "2. DISCHARGE COOLER", 0.15),
        _note_item("a3", "3. RELIEF TO FLARE", 0.20),
    ]
    notes_b = [
        _note_item("b1", "1. SUCTION SCRUBBER", 0.10, source_rev="B"),
        _note_item("b2", "2. RELIEF TO FLARE", 0.15, source_rev="B"),
    ]

    changes = diff_notes(notes_a, notes_b, cfg)
    by_type = {c.type: c for c in changes}

    assert by_type["removed"].old == "2. DISCHARGE COOLER"
    assert by_type["renumbered"].old == "3. RELIEF TO FLARE"
    assert by_type["renumbered"].new == "2. RELIEF TO FLARE"
    assert "suction" not in " ".join(c.description.lower() for c in changes)  # unchanged line -> no Change


def test_cosine_above_high_is_cosmetic_no_llm_call(cfg):
    old_text, new_text = "1. COSMETIC OLD PHRASING", "1. COSMETIC NEW PHRASING"
    content_old = normalize_text(old_text[3:])
    content_new = normalize_text(new_text[3:])
    embedder = lambda t: {content_old: [1.0, 0.0], content_new: [1.0, 1e-6]}[t]  # noqa: E731
    llm = Mock()

    notes_a = [_note_item("a1", old_text, 0.10)]
    notes_b = [_note_item("b1", new_text, 0.10, source_rev="B")]

    changes = diff_notes(notes_a, notes_b, cfg, llm=llm, embedder=embedder)

    llm.assert_not_called()
    assert changes == []


def test_cosine_below_low_is_meaningful_no_llm_call(cfg):
    old_text, new_text = "1. MEANINGFUL OLD PHRASING", "1. MEANINGFUL NEW PHRASING"
    content_old = normalize_text(old_text[3:])
    content_new = normalize_text(new_text[3:])
    embedder = lambda t: {content_old: [1.0, 0.0], content_new: [0.0, 1.0]}[t]  # noqa: E731
    llm = Mock()

    notes_a = [_note_item("a1", old_text, 0.10)]
    notes_b = [_note_item("b1", new_text, 0.10, source_rev="B")]

    changes = diff_notes(notes_a, notes_b, cfg, llm=llm, embedder=embedder)

    llm.assert_not_called()
    assert len(changes) == 1
    assert changes[0].type == "modified"
    assert changes[0].old == old_text and changes[0].new == new_text


def test_ambiguous_band_calls_llm_exactly_once_and_consumes_verdict(cfg):
    old_text, new_text = "1. AMBIG OLD PHRASING", "1. AMBIG NEW PHRASING"
    content_old = normalize_text(old_text[3:])
    content_new = normalize_text(new_text[3:])
    embedder = lambda t: {content_old: [1.0, 0.0], content_new: [0.6, 0.8]}[t]  # cosine 0.6  # noqa: E731
    llm = Mock(return_value={"is_meaningful": True, "summary": "Phrasing changed materially", "severity": "info"})

    notes_a = [_note_item("a1", old_text, 0.10)]
    notes_b = [_note_item("b1", new_text, 0.10, source_rev="B")]

    changes = diff_notes(notes_a, notes_b, cfg, llm=llm, embedder=embedder)

    assert llm.call_count == 1
    assert len(changes) == 1
    ch = changes[0]
    assert "Phrasing changed materially" in ch.description
    assert "info" in ch.description


def test_llm_never_supplies_old_new_values(cfg):
    old_text, new_text = "1. AMBIG OLD PHRASING", "1. AMBIG NEW PHRASING"
    content_old = normalize_text(old_text[3:])
    content_new = normalize_text(new_text[3:])
    embedder = lambda t: {content_old: [1.0, 0.0], content_new: [0.6, 0.8]}[t]  # noqa: E731
    llm = Mock(return_value={"is_meaningful": True, "old": "LLM-MADE-UP-OLD", "new": "LLM-MADE-UP-NEW"})

    notes_a = [_note_item("a1", old_text, 0.10)]
    notes_b = [_note_item("b1", new_text, 0.10, source_rev="B")]

    changes = diff_notes(notes_a, notes_b, cfg, llm=llm, embedder=embedder)

    assert changes[0].old == old_text
    assert changes[0].new == new_text
