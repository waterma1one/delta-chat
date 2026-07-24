"""Sample synthesis: pairs, disjoint split, edit-log==gold, OCR truth, anti-leakage, schema."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
import synth_samples  # noqa: E402


@pytest.fixture
def generated(tmp_path):
    samples, datasets = tmp_path / "samples", tmp_path / "datasets"
    (samples / "source").mkdir(parents=True)
    synth_samples.main(samples_dir=samples, datasets_dir=datasets)
    gold = [json.loads(l) for l in (datasets / "delta_gold.jsonl").read_text().splitlines()]
    qa = [json.loads(l) for l in (datasets / "qa_gold.jsonl").read_text().splitlines()]
    lock = json.loads((datasets / "dataset.lock.json").read_text())
    return {"samples": samples, "datasets": datasets, "gold": gold, "qa": qa, "lock": lock}


def test_pairs_split_hard_and_multisheet(generated):
    lock = generated["lock"]
    train, test = set(lock["split"]["train"]), set(lock["split"]["test"])
    assert train and test and not (train & test) and len(train | test) >= 4
    assert any(g["hard_slice"] for g in generated["gold"])
    ms = json.loads((generated["samples"] / "pair_04_multisheet" / "meta.json").read_text())
    assert any(e["page"] >= 2 for e in ms["edits"])


def test_every_edit_has_matching_gold_record(generated):
    gold_ids = {g["change_id"] for g in generated["gold"]}
    for meta in generated["samples"].glob("pair_*/meta.json"):
        for e in json.loads(meta.read_text())["edits"]:
            assert e["change_id"] in gold_ids


def test_gold_schema_and_bbox_in_unit_interval(generated):
    required = {"pair_id", "change_id", "type", "item_type", "page", "bbox", "old", "new", "description"}
    for g in generated["gold"]:
        assert required <= set(g)
        assert g["type"] in {"added", "removed", "modified", "moved", "renumbered"}
        assert len(g["bbox"]) == 4 and all(0.0 <= v <= 1.0 for v in g["bbox"])
    # delete + renumber both exercised
    types = {g["type"] for g in generated["gold"]}
    assert "renumbered" in types and "removed" in types


def test_scanned_pre_degrade_text_equals_ocr_truth(generated):
    from delta_chat.ingest.pdf_native import parse_native_pdf

    p = generated["samples"] / "pair_03_scanned"
    native = parse_native_pdf((p / "rev_b_native.pdf").read_bytes(), "t")
    assert " ".join(it.text for it in native.items) == (p / "ocr_truth.txt").read_text()


def test_qa_gold_unanswerable_and_both_citation_classes(generated):
    qa = generated["qa"]
    assert any(not q["answerable"] for q in qa)
    classes = {q["citation_class"] for q in qa}
    assert "delta" in classes and "doc_span" in classes


def test_calibration_reads_train_ids_only():
    assert synth_samples.select_calibration_pairs(["pair_01_native", "pair_02_hard"]) == ["pair_01_native"]
    assert synth_samples.select_calibration_pairs(["pair_03_scanned"]) == []


def test_generation_is_deterministic(tmp_path):
    def run(sub):
        s, d = tmp_path / f"s{sub}", tmp_path / f"d{sub}"
        (s / "source").mkdir(parents=True)
        synth_samples.main(samples_dir=s, datasets_dir=d)
        return json.loads((d / "dataset.lock.json").read_text())["content_hash"]

    assert run(1) == run(2)
