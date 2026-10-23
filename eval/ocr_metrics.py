"""OCR error measurement : word/character error rate against a stored ground truth.

The scanned sample pair ships `ocr_truth.txt` — the exact token sequence extracted from the
*native* rev B before it was rasterized — so the OCR path can be scored against what the page
provably said, not against a hand transcription.

Reported in the scorecard but deliberately **not** part of the CI-gated `deterministic` block:
the number depends on the installed Tesseract build, so gating it would make CI fail on an
environment upgrade rather than on a real regression. It is a reported, honest quality signal —
which is what the README's failure taxonomy claims — not a gate.
"""

from __future__ import annotations

from rapidfuzz.distance import Levenshtein


def _rate(hyp_seq, ref_seq) -> float:
    """Levenshtein edit distance over a sequence, normalized by reference length and capped at 1.0.

    Capped because insertions are unbounded: a hypothesis far longer than the reference would
    otherwise report an error rate above 100%, which reads as a bug in a scorecard.
    """
    if not len(ref_seq):
        return 0.0 if not len(hyp_seq) else 1.0
    return min(1.0, Levenshtein.distance(hyp_seq, ref_seq) / len(ref_seq))


def word_error_rate(hypothesis: str, reference: str) -> float:
    """Substitutions + insertions + deletions at the word level, over reference word count."""
    return _rate(hypothesis.split(), reference.split())


def char_error_rate(hypothesis: str, reference: str) -> float:
    """Same, at the character level (whitespace-normalized so layout doesn't dominate)."""
    return _rate(" ".join(hypothesis.split()), " ".join(reference.split()))


def score_ocr(scanned_pdf: bytes, truth_text: str, cfg=None) -> dict:
    """Run the scanned-PDF adapter over `scanned_pdf` and score its text against `truth_text`."""
    from delta_chat.ingest.pdf_scanned import parse_scanned_pdf

    doc = parse_scanned_pdf(scanned_pdf, "ocr-eval", cfg=cfg)
    hypothesis = " ".join(item.text for item in doc.items if item.text)
    return {
        "wer": word_error_rate(hypothesis, truth_text),
        "cer": char_error_rate(hypothesis, truth_text),
        "n_ref_words": len(truth_text.split()),
        "n_hyp_words": len(hypothesis.split()),
        "degraded": bool(doc.raw_meta.get("degraded", False)),
    }
