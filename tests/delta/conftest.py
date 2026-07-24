"""Shared fixtures for the delta-engine test suite."""

from __future__ import annotations

import copy

import pytest

from delta_chat.canonical.model import BBox, Item
from delta_chat.config import Config, load_config


@pytest.fixture
def cfg():
    """Default config with the note-gate cosine cutoffs filled in (calibration is §10's job; the
    delta-engine tests need concrete numbers to exercise the gate's three branches)."""
    data = copy.deepcopy(load_config().data)
    data["delta"]["notes"]["semantic_low"] = 0.35
    data["delta"]["notes"]["semantic_high"] = 0.85
    return Config(data)


def make_item(
    id_,
    text,
    *,
    source_rev="A",
    type="tag",
    region="drawing",
    page=0,
    sheet=None,
    bbox=(0.1, 0.1, 0.2, 0.12),
    parsed_fields=None,
    conf=1.0,
):
    x0, y0, x1, y1 = bbox
    return Item(
        id=id_,
        source_rev=source_rev,
        type=type,
        text=text,
        parsed_fields=parsed_fields or {},
        bbox=BBox(x0, y0, x1, y1),
        page=page,
        sheet=sheet,
        region=region,
        conf=conf,
    )


@pytest.fixture
def item_factory():
    return make_item
