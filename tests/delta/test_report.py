"""Tests for delta/report.py — the JSON retrieval-chunk contract + Markdown summary."""

from __future__ import annotations

import re

from delta_chat.canonical.model import BBox, Change, Delta
from delta_chat.delta.report import render_report


def _change(id_: str, type_: str, **overrides) -> Change:
    defaults = dict(
        item_type="tag",
        page=0,
        sheet=None,
        bbox=BBox(0.1, 0.1, 0.2, 0.12),
        description=f"{type_} change",
        old="old",
        new="new",
        rev_a_ref="a1",
        rev_b_ref="b1",
        match_conf=0.9,
        class_conf=0.9,
        change_set_id="CS-0-1",
    )
    defaults.update(overrides)
    return Change(id=id_, type=type_, **defaults)


def test_json_has_one_atomic_entry_per_change_with_location_and_confidences():
    delta = Delta(
        pid_a="A",
        pid_b="B",
        changes=[_change("CHG-0001", "added"), _change("CHG-0002", "removed")],
        counts={"added": 1, "removed": 1},
        dataset_version="v1",
    )

    _, json_dict = render_report(delta)

    assert set(json_dict["changes"]) == {"delta:CHG-0001", "delta:CHG-0002"}
    entry = json_dict["changes"]["delta:CHG-0001"]
    assert entry["location"] == {"page": 0, "sheet": None, "bbox": [0.1, 0.1, 0.2, 0.12]}
    assert entry["match_conf"] == 0.9 and entry["class_conf"] == 0.9


def test_markdown_counts_by_type_match_json():
    delta = Delta(
        pid_a="A",
        pid_b="B",
        changes=[_change("CHG-0001", "added"), _change("CHG-0002", "added"), _change("CHG-0003", "removed")],
        counts={"added": 2, "removed": 1},
        dataset_version="v1",
    )

    markdown, json_dict = render_report(delta)

    found = {k: int(v) for k, v in re.findall(r"- \*\*(\w+)\*\*: (\d+)", markdown)}
    assert found == json_dict["counts"]
