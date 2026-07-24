"""OCR error metrics — the README's failure taxonomy claims OCR error is measured and reported,
so it has to actually be computed against the `ocr_truth.txt` fixture rather than asserted."""

from __future__ import annotations

from pathlib import Path

import pytest

from eval.ocr_metrics import char_error_rate, score_ocr, word_error_rate

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_identical_text_has_zero_error():
    assert word_error_rate("26-FV-9040 200mm", "26-FV-9040 200mm") == 0.0
    assert char_error_rate("26-FV-9040", "26-FV-9040") == 0.0


def test_one_wrong_word_in_four_is_quarter_wer():
    assert word_error_rate("A B C X", "A B C D") == pytest.approx(0.25)


def test_missing_and_extra_words_both_count():
    assert word_error_rate("A B", "A B C D") == pytest.approx(0.5)      # 2 deletions / 4
    assert word_error_rate("A B C D E F", "A B C D") == pytest.approx(0.5)  # 2 insertions / 4


def test_char_error_rate_is_character_level():
    # one substituted character out of ten
    assert char_error_rate("26-FV-9041", "26-FV-9040") == pytest.approx(0.1)


def test_empty_reference_is_not_a_division_by_zero():
    assert word_error_rate("anything", "") == 1.0
    assert char_error_rate("anything", "") == 1.0
    assert word_error_rate("", "") == 0.0


def test_error_rate_is_capped_at_one():
    """A hypothesis far longer than the reference must not report >100% error."""
    assert word_error_rate("x " * 50, "A B") <= 1.0


@pytest.mark.slow
def test_score_ocr_on_the_real_scanned_sample_reports_a_rate():
    """End-to-end against the synthesized scanned pair; needs the tesseract binary."""
    pair = REPO_ROOT / "data" / "samples" / "pair_03_scanned"
    scanned, truth = pair / "rev_b_scanned.pdf", pair / "ocr_truth.txt"
    if not scanned.exists() or not truth.exists():
        pytest.skip("scanned sample not synthesized")

    result = score_ocr(scanned.read_bytes(), truth.read_text(encoding="utf-8"))

    assert 0.0 <= result["wer"] <= 1.0
    assert 0.0 <= result["cer"] <= 1.0
    assert result["n_ref_words"] > 0
