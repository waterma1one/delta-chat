"""ISA-5.1 tag / line-number / equipment extraction.

Config-driven (house standards vary): the area-instrument, equipment, and line-number patterns come from
``config/default.yaml`` (``tagging.*_regex``); the instrument/equipment split is driven by
``tagging.isa_first_letters`` + ``tagging.equipment_first_letters``. Hits carry ``parsed_fields`` so the
delta engine can do deterministic value diffs (e.g. ``FV-9038`` -> ``FV-9040``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from ..canonical.model import ItemType

# Module defaults (used when no cfg is passed). Config overrides these at runtime.
_DEF_AREA = r"(\d{1,2})-([A-Z]{2,4})-(\d{3,5})([A-Z]?)"
_DEF_EQUIP = r"([A-Z]{1,3})-(\d{2,4})([A-Z]?)"
_DEF_LINE = r'\d{1,3}(?:/\d)?"-[A-Z0-9]+(?:-[A-Z0-9]+){2,}'
# Classic ISA tag (FIC-101, LT201A) — secondary matcher, not currently a config key.
_CLASSIC = re.compile(r"^([A-Z]{2,4})-?(\d{2,4})([A-Z]?)$")
_DIMENSION = re.compile(r'^(\d+(?:\.\d+)?)\s?(mm|cm|m|in|"|deg)$')

_DEFAULT_EQUIP_LETTERS = {"P", "V", "E", "T", "K", "C", "KA", "KZ", "CX", "CK"}
_DEFAULT_ISA_LETTERS = {"F", "L", "P", "T", "A", "D", "I", "S", "W", "H"}


@dataclass
class TagHit:
    type: ItemType  # "tag" | "line_number" | "dimension"
    parsed_fields: dict


@lru_cache(maxsize=64)
def _anchored(pattern: str) -> re.Pattern:
    body = pattern.strip()
    if body.startswith("^"):
        body = body[1:]
    if body.endswith("$"):
        body = body[:-1]
    return re.compile(f"^(?:{body})$")


def _patterns(cfg):
    if cfg is None:
        area, equip, line = _DEF_AREA, _DEF_EQUIP, _DEF_LINE
        equip_letters, isa_letters = _DEFAULT_EQUIP_LETTERS, _DEFAULT_ISA_LETTERS
    else:
        area = cfg.get("tagging.instrument_regex", _DEF_AREA)
        equip = cfg.get("tagging.equipment_regex", _DEF_EQUIP)
        line = cfg.get("tagging.line_number_regex", _DEF_LINE)
        equip_letters = set(cfg.get("tagging.equipment_first_letters", [])) | _DEFAULT_EQUIP_LETTERS
        isa_letters = set(cfg.get("tagging.isa_first_letters", [])) | _DEFAULT_ISA_LETTERS
    try:
        return _anchored(area), _anchored(equip), _anchored(line), equip_letters, isa_letters
    except re.error:
        return _anchored(_DEF_AREA), _anchored(_DEF_EQUIP), _anchored(_DEF_LINE), equip_letters, isa_letters


def _kind(cls: str, equip_letters: set, isa_letters: set) -> str:
    if cls in equip_letters:
        return "equipment"
    if cls and cls[0] in isa_letters:
        return "instrument"
    return "instrument"


def extract_tags(word: str, cfg=None) -> list[TagHit]:
    """Return typed tag hits for a single token (0 or 1). Order: line-number, area-tag, classic-tag,
    equipment, dimension."""
    w = word.strip()
    if not w:
        return []
    area_re, equip_re, line_re, equip_letters, isa_letters = _patterns(cfg)

    if line_re.match(w):
        return [TagHit("line_number", {"raw": w})]

    m = area_re.match(w)
    if m and len(m.groups()) >= 3:
        g = m.groups()
        area, cls, loop = g[0], g[1], g[2]
        suffix = g[3] if len(g) > 3 else ""
        return [TagHit("tag", {"area": area, "tag_class": cls, "loop": loop, "suffix": suffix or "",
                               "kind": _kind(cls, equip_letters, isa_letters)})]

    m = _CLASSIC.match(w)
    if m:
        cls, loop, suffix = m.groups()
        return [TagHit("tag", {"tag_class": cls, "loop": loop, "suffix": suffix or "",
                               "kind": _kind(cls, equip_letters, isa_letters)})]

    m = equip_re.match(w)
    if m and len(m.groups()) >= 2:
        g = m.groups()
        cls, loop = g[0], g[1]
        suffix = g[2] if len(g) > 2 else ""
        return [TagHit("tag", {"tag_class": cls, "loop": loop, "suffix": suffix or "",
                               "kind": _kind(cls, equip_letters, isa_letters)})]

    m = _DIMENSION.match(w)
    if m:
        return [TagHit("dimension", {"value": float(m.group(1)), "unit": m.group(2)})]
    return []
