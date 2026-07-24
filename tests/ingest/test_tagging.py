"""ISA-5.1 tag extraction."""

from __future__ import annotations

from delta_chat.config import load_config
from delta_chat.ingest.tagging import extract_tags


def test_instrument_tags_area_and_classic():
    (h,) = extract_tags("26-PDI-9054")
    assert h.type == "tag" and h.parsed_fields["tag_class"] == "PDI" and h.parsed_fields["loop"] == "9054"
    (h,) = extract_tags("FIC-101")
    assert h.type == "tag" and h.parsed_fields["tag_class"] == "FIC" and h.parsed_fields["loop"] == "101"
    (h,) = extract_tags("LT201A")
    assert h.parsed_fields["tag_class"] == "LT" and h.parsed_fields["loop"] == "201" and h.parsed_fields["suffix"] == "A"


def test_line_number_is_one_token():
    (h,) = extract_tags('3/4"-DC-26-9026-FC11S-00')
    assert h.type == "line_number" and h.parsed_fields["raw"] == '3/4"-DC-26-9026-FC11S-00'


def test_equipment_disambiguated_by_leading_letter():
    (p,) = extract_tags("P-101")
    (v,) = extract_tags("V-201")
    assert p.type == "tag" and p.parsed_fields["kind"] == "equipment"
    assert v.parsed_fields["kind"] == "equipment"
    # area-prefixed equipment (compressor KA) still a tag
    (k,) = extract_tags("26-KA-902")
    assert k.type == "tag" and k.parsed_fields["kind"] == "equipment"


def test_non_tags_not_matched():
    assert extract_tags("PUMP") == []
    assert extract_tags("150") == []          # bare number, no unit
    (d,) = extract_tags("150mm")
    assert d.type == "dimension" and d.parsed_fields == {"value": 150.0, "unit": "mm"}


def test_config_drives_line_number_regex(monkeypatch):
    tok = '3/4"-DC-26-9026-FC11S-00'
    assert extract_tags(tok, load_config())[0].type == "line_number"
    monkeypatch.setenv("DELTA_CHAT__TAGGING__LINE_NUMBER_REGEX", "ZZZNOMATCH")
    hits = extract_tags(tok, load_config())
    assert not hits or hits[0].type != "line_number"


def test_config_drives_instrument_regex(monkeypatch):
    assert extract_tags("26-PDI-9054", load_config())[0].type == "tag"
    monkeypatch.setenv("DELTA_CHAT__TAGGING__INSTRUMENT_REGEX", "ZZZNOMATCH")
    hits = extract_tags("26-PDI-9054", load_config())
    # with the area pattern disabled, "26-PDI-9054" no longer matches as a tag
    assert not hits or hits[0].type != "tag"
