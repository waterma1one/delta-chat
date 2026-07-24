"""Tests for markup/overlay.py — Delta annotation overlay onto the source PDF. No LLM, no network."""

from __future__ import annotations

import copy
import json

import fitz
import pytest

from delta_chat.canonical.model import BBox, Change, Delta
from delta_chat.config import Config, load_config
from delta_chat.markup.overlay import render_markup


def _blank_pdf(n_pages: int = 2, size: tuple[float, float] = (200.0, 150.0)) -> bytes:
    doc = fitz.open()
    for _ in range(n_pages):
        doc.new_page(width=size[0], height=size[1])
    data = doc.tobytes()
    doc.close()
    return data


def _change(id_, type_, *, page=0, bbox=(0.1, 0.1, 0.3, 0.2), change_set_id="CS-0-1", **overrides) -> Change:
    x0, y0, x1, y1 = bbox
    defaults = dict(
        item_type="tag", sheet=None, description=f"{type_} change {id_}",
        old=None, new=None, rev_a_ref=None, rev_b_ref=None,
        match_conf=0.9, class_conf=0.9,
    )
    defaults.update(overrides)
    return Change(id=id_, type=type_, page=page, bbox=BBox(x0, y0, x1, y1), change_set_id=change_set_id, **defaults)


def _close(a, b, tol=0.02) -> bool:
    return all(abs(x - y) < tol for x, y in zip(a, b))


@pytest.fixture
def source_pdf() -> bytes:
    return _blank_pdf(2)


def test_render_markup_returns_valid_pdf_with_one_annot_per_change(source_pdf):
    delta = Delta(
        pid_a="A", pid_b="B",
        changes=[_change("CHG-0001", "added"), _change("CHG-0002", "removed", page=1)],
        counts={"added": 1, "removed": 1},
    )

    out = render_markup(delta, source_pdf)

    doc = fitz.open(stream=out, filetype="pdf")
    assert doc.page_count >= 1
    total_annots = sum(len(list(page.annots())) for page in doc)
    assert total_annots == len(delta.changes)


def test_colors_map_by_change_type(source_pdf):
    delta = Delta(
        pid_a="A", pid_b="B",
        changes=[
            _change("CHG-0001", "added", bbox=(0.1, 0.1, 0.2, 0.15), change_set_id="CS-0-1"),
            _change("CHG-0002", "removed", bbox=(0.3, 0.1, 0.4, 0.15), change_set_id="CS-0-2"),
            _change("CHG-0003", "modified", bbox=(0.5, 0.1, 0.6, 0.15), change_set_id="CS-0-3"),
            _change("CHG-0004", "moved", bbox=(0.7, 0.1, 0.8, 0.15), change_set_id="CS-0-4"),
        ],
        counts={},
    )

    out = render_markup(delta, source_pdf)
    doc = fitz.open(stream=out, filetype="pdf")
    colors_by_title = {a.info["title"]: a.colors["stroke"] for a in doc[0].annots()}

    assert _close(colors_by_title["CHG-0001"], (0.0, 0.7, 0.0))
    assert _close(colors_by_title["CHG-0002"], (0.85, 0.0, 0.0))
    assert _close(colors_by_title["CHG-0003"], (1.0, 0.65, 0.0))
    assert _close(colors_by_title["CHG-0004"], (0.0, 0.35, 0.85))


def test_change_bbox_lands_on_correct_page_and_region(source_pdf):
    delta = Delta(
        pid_a="A", pid_b="B",
        changes=[_change("CHG-0001", "added", page=1, bbox=(0.4, 0.4, 0.6, 0.5))],
        counts={"added": 1},
    )

    out = render_markup(delta, source_pdf)
    doc = fitz.open(stream=out, filetype="pdf")

    assert list(doc[0].annots()) == []

    page1 = doc[1]
    annots = list(page1.annots())
    assert len(annots) == 1
    expected = fitz.Rect(
        0.4 * page1.rect.width, 0.4 * page1.rect.height,
        0.6 * page1.rect.width, 0.5 * page1.rect.height,
    )
    assert annots[0].rect.intersects(expected)


def test_empty_delta_returns_valid_pdf_with_no_annotations(source_pdf):
    delta = Delta(pid_a="A", pid_b="B", changes=[], counts={})

    out = render_markup(delta, source_pdf)

    doc = fitz.open(stream=out, filetype="pdf")
    assert doc.page_count >= 1
    assert sum(len(list(p.annots())) for p in doc) == 0


def test_legend_lists_color_to_type_mapping(source_pdf):
    delta = Delta(pid_a="A", pid_b="B", changes=[_change("CHG-0001", "added")], counts={"added": 1})

    out = render_markup(delta, source_pdf)

    doc = fitz.open(stream=out, filetype="pdf")
    text = doc[0].get_text().lower()
    assert "added" in text


def test_only_pages_with_changes_are_annotated_and_change_sets_grouped(source_pdf):
    delta = Delta(
        pid_a="A", pid_b="B",
        changes=[
            _change("CHG-0001", "added", page=0, bbox=(0.1, 0.1, 0.2, 0.15), change_set_id="CS-0-1"),
            _change("CHG-0002", "modified", page=0, bbox=(0.22, 0.1, 0.32, 0.15), change_set_id="CS-0-1"),
        ],
        counts={},
    )

    out = render_markup(delta, source_pdf)

    doc = fitz.open(stream=out, filetype="pdf")
    assert list(doc[1].annots()) == []
    # the change-set cloud enclosure is decorative vector graphics, not an annotation, so the
    # per-change annotation count stays exactly == len(changes) even for a grouped set.
    assert len(list(doc[0].annots())) == 2


def test_out_of_range_page_is_skipped_not_raised(source_pdf, capsys):
    delta = Delta(
        pid_a="A", pid_b="B",
        changes=[_change("CHG-0001", "added", page=0), _change("CHG-0099", "added", page=99)],
        counts={},
    )

    out = render_markup(delta, source_pdf)  # must not raise

    doc = fitz.open(stream=out, filetype="pdf")
    assert sum(len(list(p.annots())) for p in doc) == 1

    out_lines = capsys.readouterr().out.splitlines()
    logged = [json.loads(l) for l in out_lines if l.strip().startswith("{")]
    assert any(ln.get("event") == "markup.page_out_of_range" for ln in logged)


def test_inverted_bbox_is_normalized_not_collapsed_to_meaningless_point(source_pdf):
    # BBox.__post_init__ only clamps each coordinate into [0,1] independently; it does not
    # enforce x0<=x1/y0<=y1, so a directly-constructed inverted bbox is legal at the type level.
    delta = Delta(
        pid_a="A", pid_b="B",
        changes=[_change("CHG-0001", "added", bbox=(0.6, 0.5, 0.4, 0.3))],  # x1<x0, y1<y0
        counts={"added": 1},
    )

    out = render_markup(delta, source_pdf)

    doc = fitz.open(stream=out, filetype="pdf")
    page = doc[0]
    annots = list(page.annots())
    assert len(annots) == 1
    expected = fitz.Rect(
        0.4 * page.rect.width, 0.3 * page.rect.height,
        0.6 * page.rect.width, 0.5 * page.rect.height,
    )
    assert annots[0].rect.intersects(expected)


def test_config_color_override_merges_over_defaults(source_pdf):
    data = copy.deepcopy(load_config().data)
    # override only "added"; every other type must keep its documented default color.
    data["markup"]["colors"] = {"added": [0.1, 0.2, 0.3]}
    cfg = Config(data)

    delta = Delta(
        pid_a="A", pid_b="B",
        changes=[
            _change("CHG-0001", "added", bbox=(0.1, 0.1, 0.2, 0.15), change_set_id="CS-0-1"),
            _change("CHG-0002", "removed", bbox=(0.3, 0.1, 0.4, 0.15), change_set_id="CS-0-2"),
        ],
        counts={},
    )

    out = render_markup(delta, source_pdf, cfg=cfg)

    doc = fitz.open(stream=out, filetype="pdf")
    colors_by_title = {a.info["title"]: a.colors["stroke"] for a in doc[0].annots()}
    assert _close(colors_by_title["CHG-0001"], (0.1, 0.2, 0.3))
    assert _close(colors_by_title["CHG-0002"], (0.85, 0.0, 0.0))  # untouched default, not gray fallback
