"""Synthesize revision pairs with logged, known ground truth.

The two source P&IDs are different drawings (not revisions), so we author controlled Rev A -> Rev B
pairs. Every authored edit is recorded as a gold change, so the edit log *is* the delta ground truth
(reused as eval gold and pytest fixtures). Deterministic: config seed, sorted JSONL, sorted-key JSON.

Pairs: 01 native (tag/dimension edits + note delete&renumber), 02 hard (duplicate-tag disambiguation +
note split), 03 scanned (rasterized + skewed + noised + JPEG rev B; native rev B = OCR ground truth),
04 multi-sheet (edit on sheet 3), 05 dxf (added LINE / changed MTEXT / moved INSERT). Provenance in
data/samples/provenance.md.

DXF gold bboxes are derived from real geometry via ``ezdxf.bbox`` normalized against the drawing extents
(the same convention the DWG adapter must use), so they are not fabricated.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import fitz  # PyMuPDF

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))  # allow standalone `python scripts/synth_samples.py`

from delta_chat.canonical.model import normalize_bbox  # noqa: E402
from delta_chat.config import load_config  # noqa: E402
from delta_chat.ingest.pdf_native import parse_native_pdf  # noqa: E402

DEF_SAMPLES = REPO / "data" / "samples"
DEF_DATASETS = REPO / "eval" / "datasets"
PAGE = (595.0, 842.0)  # A4 portrait points
FONT = 10.0


@dataclass
class Tok:
    text: str
    x: float
    y: float
    page: int = 0


@dataclass
class GoldChange:
    pair_id: str
    change_id: str
    type: str          # added|removed|modified|moved|renumbered
    item_type: str
    page: int
    sheet: str | None
    bbox: list          # normalized [x0,y0,x1,y1] on rev B (or A for removed)
    old: str | None
    new: str | None
    description: str
    format: str         # extension for segmentation (not in the minimal schema)
    hard_slice: bool = False


def _bbox(text: str, x: float, y: float) -> list:
    w = fitz.get_text_length(text, fontname="helv", fontsize=FONT)
    return [round(x / PAGE[0], 6), round((y - FONT) / PAGE[1], 6),
            round((x + w) / PAGE[0], 6), round(y / PAGE[1], 6)]


def _render(tokens: list[Tok], n_pages: int = 1) -> bytes:
    doc = fitz.open()
    for _ in range(n_pages):
        doc.new_page(width=PAGE[0], height=PAGE[1])
    for t in tokens:
        doc[t.page].insert_text((t.x, t.y), t.text, fontsize=FONT)
    data = doc.tobytes()
    doc.close()
    return data


# --- pairs ---------------------------------------------------------------------------------------

def _pair_01_native():
    pid = "pair_01_native"
    a = [
        Tok("26-FV-9038", 80, 100), Tok("26-PIT-9055", 80, 130), Tok("26-KA-902", 80, 160),
        Tok("150mm", 300, 100), Tok("26-PDI-9054", 300, 130),
        Tok("1. SUCTION SCRUBBER", 80, 500),
        Tok("2. DISCHARGE COOLER", 80, 520),
        Tok("3. RELIEF TO FLARE", 80, 540),
    ]
    b = [Tok(t.text, t.x, t.y, t.page) for t in a]
    gold = []
    b[0].text = "26-FV-9040"
    gold.append(GoldChange(pid, "GOLD-0001", "modified", "tag", 0, None, _bbox(b[0].text, 80, 100),
                           "26-FV-9038", "26-FV-9040", "Instrument tag 9038 -> 9040", "pdf_native"))
    b[3].text = "200mm"
    gold.append(GoldChange(pid, "GOLD-0002", "modified", "dimension", 0, None, _bbox(b[3].text, 300, 100),
                           "150mm", "200mm", "Dimension 150mm -> 200mm", "pdf_native"))
    # delete note 2, renumber note 3 -> 2  (single-token notes => no orphan removal)
    del b[6]  # "2. DISCHARGE COOLER"
    gold.append(GoldChange(pid, "GOLD-0003", "removed", "note", 0, None, _bbox("2. DISCHARGE COOLER", 80, 520),
                           "2. DISCHARGE COOLER", None, "Note 2 deleted", "pdf_native"))
    b[-1].text = "2. RELIEF TO FLARE"  # was "3. ..."
    gold.append(GoldChange(pid, "GOLD-0004", "renumbered", "note", 0, None, _bbox(b[-1].text, 80, 540),
                           "3. RELIEF TO FLARE", "2. RELIEF TO FLARE", "Note 3 renumbered to 2", "pdf_native"))
    added = Tok("26-TIT-9211", 300, 160)
    b.append(added)
    gold.append(GoldChange(pid, "GOLD-0005", "added", "tag", 0, None, _bbox(added.text, 300, 160),
                           None, "26-TIT-9211", "Instrument 26-TIT-9211 added", "pdf_native"))
    return _render(a), _render(b), gold, [("26-FV-9040", 0), ("200mm", 0), ("26-TIT-9211", 0)]


def _pair_02_hard():
    pid = "pair_02_hard"
    a = [
        Tok("26-FV-9040", 80, 100), Tok("26-FV-9040", 400, 100),  # duplicate tag
        Tok("1. PURGE BEFORE START AND VENT", 80, 400),
        Tok("26-KA-902", 80, 200),
    ]
    b = [Tok(t.text, t.x, t.y, t.page) for t in a]
    gold = []
    b[1].y = 300  # move the SECOND duplicate down (disambiguation-by-position)
    gold.append(GoldChange(pid, "GOLD-0101", "moved", "tag", 0, None, _bbox(b[1].text, 400, 300),
                           "26-FV-9040", "26-FV-9040", "Duplicate FV-9040 (2nd instance) moved down",
                           "pdf_native", hard_slice=True))
    b[2].text = "1. PURGE BEFORE START"  # split note
    b.append(Tok("2. AND VENT TO ATMOSPHERE", 80, 420))
    gold.append(GoldChange(pid, "GOLD-0102", "modified", "note", 0, None, _bbox(b[2].text, 80, 400),
                           "1. PURGE BEFORE START AND VENT", "1. PURGE BEFORE START", "Note 1 split (part 1)",
                           "pdf_native", hard_slice=True))
    gold.append(GoldChange(pid, "GOLD-0103", "added", "note", 0, None, _bbox(b[-1].text, 80, 420),
                           None, "2. AND VENT TO ATMOSPHERE", "Note 1 split (new part 2)",
                           "pdf_native", hard_slice=True))
    return _render(a), _render(b), gold, [("26-KA-902", 0)]


def _pair_04_multisheet():
    pid = "pair_04_multisheet"
    a = [Tok("26-AA-101", 80, 100, 0), Tok("26-BB-202", 80, 100, 1),
         Tok("26-CC-303", 80, 100, 2), Tok("Sheet 3", 450, 60, 2)]
    b = [Tok(t.text, t.x, t.y, t.page) for t in a]
    b[2].text = "26-CC-404"
    gold = [GoldChange(pid, "GOLD-0301", "modified", "tag", 2, "3", _bbox(b[2].text, 80, 100),
                       "26-CC-303", "26-CC-404", "Tag on sheet 3 changed 303 -> 404", "pdf_native")]
    return _render(a, 3), _render(b, 3), gold, [("26-CC-404", 2)]


def _degrade_to_scanned(native_pdf: bytes, cfg) -> bytes:
    import cv2
    import numpy as np

    rng = np.random.default_rng(int(cfg.get("samples.seed", 1234)))
    dpi = int(cfg.get("samples.scanned.dpi", 220))
    sigma = float(cfg.get("samples.scanned.noise_sigma", 8))
    skew = float(cfg.get("samples.scanned.skew_deg", 2.0))
    quality = int(cfg.get("samples.scanned.jpeg_quality", 60))

    src = fitz.open(stream=native_pdf, filetype="pdf")
    out = fitz.open()
    for page in src:
        pix = page.get_pixmap(dpi=dpi, colorspace=fitz.csGRAY)
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).copy()
        # skew (affine rotation about centre)
        h, w = img.shape
        m = cv2.getRotationMatrix2D((w / 2, h / 2), skew, 1.0)
        img = cv2.warpAffine(img, m, (w, h), borderValue=255)
        # noise
        img = np.clip(img.astype(np.int16) + rng.normal(0, sigma, img.shape).astype(np.int16), 0, 255).astype(np.uint8)
        # JPEG re-encode (compression artifacts)
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpg = buf.tobytes() if ok else pix.tobytes("png")
        newpage = out.new_page(width=page.rect.width, height=page.rect.height)
        newpage.insert_image(page.rect, stream=jpg)
    data = out.tobytes()
    src.close()
    out.close()
    return data


def _dxf_norm(entity_ext, doc_ext) -> list:
    """Normalize a DXF entity bounding box against the drawing extents (the DWG adapter must match this)."""
    dw = doc_ext.extmax.x - doc_ext.extmin.x or 1.0
    dh = doc_ext.extmax.y - doc_ext.extmin.y or 1.0
    raw = (entity_ext.extmin.x - doc_ext.extmin.x, entity_ext.extmin.y - doc_ext.extmin.y,
           entity_ext.extmax.x - doc_ext.extmin.x, entity_ext.extmax.y - doc_ext.extmin.y)
    b = normalize_bbox(raw, "dxf_wcs", (dw, dh))
    return [round(v, 6) for v in b.as_tuple()]


def _pair_05_dxf():
    import ezdxf
    from ezdxf import bbox

    pid = "pair_05_dxf"

    def build(edited: bool):
        doc = ezdxf.new()
        if "SYM" not in doc.blocks:
            blk = doc.blocks.new(name="SYM")
            blk.add_circle((0, 0), radius=5)
        msp = doc.modelspace()
        msp.add_mtext("26-KA-903" if edited else "26-KA-902").set_location((10, 10))
        msp.add_line((0, 0), (120, 0))
        msp.add_blockref("SYM", (90, 90) if edited else (10, 90))
        if edited:
            msp.add_line((0, 60), (120, 60))  # added LINE
        buf = io.StringIO()
        doc.write(buf)
        return doc, msp, buf.getvalue().encode("utf-8")

    doc_a, msp_a, a = build(False)
    doc_b, msp_b, b = build(True)
    ext_b = bbox.extents(msp_b)
    mtext_b = _dxf_norm(bbox.extents([e for e in msp_b if e.dxftype() == "MTEXT"]), ext_b)
    added_line = [e for e in msp_b if e.dxftype() == "LINE"][-1]
    line_bb = _dxf_norm(bbox.extents([added_line]), ext_b)
    insert_b = _dxf_norm(bbox.extents([e for e in msp_b if e.dxftype() == "INSERT"]), ext_b)
    gold = [
        GoldChange(pid, "GOLD-0401", "modified", "text", 0, None, mtext_b,
                   "26-KA-902", "26-KA-903", "MTEXT tag 902 -> 903", "dwg"),
        GoldChange(pid, "GOLD-0402", "added", "symbol", 0, None, line_bb,
                   None, "LINE", "Horizontal line added", "dwg"),
        GoldChange(pid, "GOLD-0403", "moved", "symbol", 0, None, insert_b,
                   "SYM", "SYM", "Block INSERT 'SYM' moved (10,90) -> (90,90)", "dwg"),
    ]
    return a, b, gold


# --- gold assembly -------------------------------------------------------------------------------

def _qa_gold(dim_item_id: str) -> list[dict]:
    return [
        {"pair_id": "pair_01_native", "question": "What happened to valve FV-9038?",
         "reference_answer": "It was renumbered to FV-9040.", "expected_citation_ids": ["delta:GOLD-0001"],
         "citation_class": "delta", "answerable": True},
        {"pair_id": "pair_01_native", "question": "What is the new dimension on the suction line?",
         "reference_answer": "200mm.", "expected_citation_ids": ["delta:GOLD-0002"],
         "citation_class": "delta", "answerable": True},
        {"pair_id": "pair_01_native", "question": "Which note was deleted?",
         "reference_answer": "Note 2, DISCHARGE COOLER.", "expected_citation_ids": ["delta:GOLD-0003"],
         "citation_class": "delta", "answerable": True},
        {"pair_id": "pair_01_native", "question": "What tag was added in Rev B?",
         "reference_answer": "26-TIT-9211.", "expected_citation_ids": ["delta:GOLD-0005"],
         "citation_class": "delta", "answerable": True},
        {"pair_id": "pair_01_native", "question": "What does note 1 say in Rev A?",
         "reference_answer": "SUCTION SCRUBBER.", "expected_citation_ids": [f"doc_a:{dim_item_id}"],
         "citation_class": "doc_span", "answerable": True},
        {"pair_id": "pair_01_native", "question": "What is the design pressure of the vessel?",
         "reference_answer": "Not stated in the documents.", "expected_citation_ids": [],
         "citation_class": "delta", "answerable": False},
        {"pair_id": "pair_01_native", "question": "Who approved the drawing?",
         "reference_answer": "Not stated.", "expected_citation_ids": [], "citation_class": "delta",
         "answerable": False},
        {"pair_id": "pair_02_hard", "question": "Did any duplicate tag move?",
         "reference_answer": "Yes, the second FV-9040 moved down.", "expected_citation_ids": ["delta:GOLD-0101"],
         "citation_class": "delta", "answerable": True},
        {"pair_id": "pair_02_hard", "question": "How did note 1 change?",
         "reference_answer": "It was split into two notes.", "expected_citation_ids": ["delta:GOLD-0102", "delta:GOLD-0103"],
         "citation_class": "delta", "answerable": True},
        {"pair_id": "pair_04_multisheet", "question": "What changed on sheet 3?",
         "reference_answer": "Tag 26-CC-303 became 26-CC-404.", "expected_citation_ids": ["delta:GOLD-0301"],
         "citation_class": "delta", "answerable": True},
        {"pair_id": "pair_05_dxf", "question": "What changed in the DXF drawing?",
         "reference_answer": "A line was added, the tag became 903, and a block moved.",
         "expected_citation_ids": ["delta:GOLD-0401", "delta:GOLD-0402", "delta:GOLD-0403"],
         "citation_class": "delta", "answerable": True},
        {"pair_id": "pair_04_multisheet", "question": "What is the flow rate on sheet 2?",
         "reference_answer": "Not stated.", "expected_citation_ids": [], "citation_class": "delta",
         "answerable": False},
    ]


def select_calibration_pairs(requested_ids: list[str], cfg=None) -> list[str]:
    """Anti-leakage gate: return only TRAIN pair ids from the requested set; drop any test id."""
    cfg = cfg or load_config()
    train = set(cfg.get("eval.datasets.split.train", []))
    return [pid for pid in requested_ids if pid in train]


def _write_pair(pair_dir: Path, files: dict[str, bytes], edits: list[GoldChange]) -> None:
    pair_dir.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (pair_dir / name).write_bytes(data)
    (pair_dir / "meta.json").write_text(
        json.dumps({"edits": [asdict(e) for e in edits]}, indent=2, sort_keys=True), encoding="utf-8"
    )


def _validate_extractable(rev_b: bytes, expect: list[tuple[str, int]], pair: str) -> None:
    """Round-trip: authored tokens must be extractable at the right page, else fail loudly."""
    doc = parse_native_pdf(rev_b, f"{pair}:revB")
    texts = {(it.text, it.page) for it in doc.items}
    for text, page in expect:
        if (text, page) not in texts:
            raise AssertionError(f"{pair}: authored token {text!r} not extractable on page {page}")


def main(samples_dir: Path | None = None, datasets_dir: Path | None = None) -> int:
    cfg = load_config()
    samples = Path(samples_dir) if samples_dir else DEF_SAMPLES
    datasets = Path(datasets_dir) if datasets_dir else DEF_DATASETS
    samples.mkdir(parents=True, exist_ok=True)
    datasets.mkdir(parents=True, exist_ok=True)
    all_gold: list[GoldChange] = []

    a1, b1, g1, exp1 = _pair_01_native()
    _validate_extractable(b1, exp1, "pair_01_native")
    _write_pair(samples / "pair_01_native", {"rev_a.pdf": a1, "rev_b.pdf": b1}, g1)
    all_gold += g1

    a2, b2, g2, exp2 = _pair_02_hard()
    _validate_extractable(b2, exp2, "pair_02_hard")
    _write_pair(samples / "pair_02_hard", {"rev_a.pdf": a2, "rev_b.pdf": b2}, g2)
    all_gold += g2

    scanned = _degrade_to_scanned(b1, cfg)
    ocr_truth = " ".join(it.text for it in parse_native_pdf(b1, "pair_03_scanned:revB").items)
    g3 = [GoldChange("pair_03_scanned", g.change_id.replace("GOLD-00", "GOLD-20"), g.type, g.item_type,
                     g.page, g.sheet, g.bbox, g.old, g.new, g.description, "pdf_scanned", g.hard_slice)
          for g in g1]
    p3 = samples / "pair_03_scanned"
    _write_pair(p3, {"rev_a.pdf": a1, "rev_b_scanned.pdf": scanned, "rev_b_native.pdf": b1}, g3)
    (p3 / "ocr_truth.txt").write_text(ocr_truth, encoding="utf-8")
    all_gold += g3

    a4, b4, g4, exp4 = _pair_04_multisheet()
    _validate_extractable(b4, exp4, "pair_04_multisheet")
    _write_pair(samples / "pair_04_multisheet", {"rev_a.pdf": a4, "rev_b.pdf": b4}, g4)
    all_gold += g4

    a5, b5, g5 = _pair_05_dxf()
    _write_pair(samples / "pair_05_dxf", {"rev_a.dxf": a5, "rev_b.dxf": b5}, g5)
    all_gold += g5

    # resolve a real doc-span citation id for the qa_gold (Item.id of note 1 in rev A)
    rev_a_items = parse_native_pdf(a1, "pair_01_native:revA").items
    note1 = next((it for it in rev_a_items if it.text.startswith("1. SUCTION")), None)
    dim_item_id = note1.id if note1 else rev_a_items[0].id

    gold_sorted = sorted(all_gold, key=lambda g: (g.pair_id, g.change_id))
    with (datasets / "delta_gold.jsonl").open("w", encoding="utf-8") as fh:
        for g in gold_sorted:
            fh.write(json.dumps(asdict(g), sort_keys=True) + "\n")
    with (datasets / "qa_gold.jsonl").open("w", encoding="utf-8") as fh:
        for q in _qa_gold(dim_item_id):
            fh.write(json.dumps(q, sort_keys=True) + "\n")

    source_bytes = b""
    for sp in sorted((samples / "source").glob("*.pdf")):
        source_bytes += sp.read_bytes()
    content = hashlib.sha256(
        (datasets / "delta_gold.jsonl").read_bytes()
        + (datasets / "qa_gold.jsonl").read_bytes()
        + source_bytes
    ).hexdigest()
    (datasets / "dataset.lock.json").write_text(
        json.dumps({"dataset_version": cfg.get("eval.datasets.dataset_version", "v1"),
                    "content_hash": f"sha256:{content}",
                    "split": {"train": cfg.get("eval.datasets.split.train", []),
                              "test": cfg.get("eval.datasets.split.test", [])}},
                   indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (samples / "provenance.md").write_text(_provenance(), encoding="utf-8")
    print(f"synthesized {len(gold_sorted)} gold changes across 5 pairs; hash sha256:{content[:12]}")
    return 0


def _provenance() -> str:
    return (
        "# Sample provenance\n\n"
        "The two source P&IDs used in development are different drawings, not revisions, and are not\n"
        "redistributed here, so revision pairs are\n"
        "synthesized with logged edits. Each edit is a gold change (`eval/datasets/delta_gold.jsonl`);\n"
        "every authored token is round-trip validated as extractable by the native adapter.\n\n"
        "| pair | split | edits |\n|---|---|---|\n"
        "| pair_01_native | train | tag 9038->9040; dim 150->200mm; delete note 2; renumber note 3->2; add 26-TIT-9211 |\n"
        "| pair_02_hard | test | move 2nd of duplicate FV-9040 (disambiguation); split a note into two |\n"
        "| pair_03_scanned | test | pair_01 rev B rasterized + skewed + noised + JPEG; native rev B = OCR truth |\n"
        "| pair_04_multisheet | test | 3 sheets; tag edit on sheet 3 |\n"
        "| pair_05_dxf | test | DXF: add LINE; change MTEXT tag; move block INSERT (gold bbox from ezdxf extents) |\n\n"
        "Anti-leakage: TRAIN is reserved for future threshold/note-gate calibration (see `select_calibration_pairs`);\n"
        "TEST pairs headline the scorecard. Split + seed are read from `config/default.yaml`. No PII, no secrets.\n\n"
        "Known limitation: the hard slice uses a duplicate-tag positional move (text), not a true vector-path\n"
        "move; a genuine geometry-move synthesizer is future work. Q&A gold is a representative ~12-item set\n"
        "(a fuller 50-200 set is future work).\n"
    )


if __name__ == "__main__":
    sys.exit(main())
