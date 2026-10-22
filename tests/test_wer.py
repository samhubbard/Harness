import pytest

from harness.wer import WordErrors, corpus_wer, wer, word_errors

REF = "turn right heading zero niner zero"


@pytest.mark.parametrize("hypothesis, expected", [
    (REF, WordErrors(reference_words=6)),
    ("Turn right, heading zero niner zero.", WordErrors(reference_words=6)),             # case/punctuation ignored
    ("turn right heading zero nine zero", WordErrors(substitutions=1, reference_words=6)),
    ("turn right heading zero zero", WordErrors(deletions=1, reference_words=6)),
    ("turn right uh heading zero niner zero", WordErrors(insertions=1, reference_words=6)),
    ("turn left heading zero niner", WordErrors(substitutions=1, deletions=1, reference_words=6)),
])
def test_word_errors(hypothesis, expected):
    assert word_errors(REF, hypothesis) == expected


def test_wer_can_exceed_100_percent():
    assert wer("climb", "uh climb climb climb") == 3.0


@pytest.mark.parametrize("hypothesis", ["Climb and maintain 5,000", "climb and maintain five zero zero zero"])
def test_number_normalization_ignores_formatting(hypothesis):
    reference = "Climb and maintain five thousand"
    assert wer(reference, hypothesis) > 0                  # raw: formatting counts as errors
    assert wer(reference, hypothesis, numbers=True) == 0   # normalized: same number, no error


def test_number_normalization_still_counts_wrong_numbers():
    assert word_errors("heading two seven zero", "heading two seven five", numbers=True) == \
        WordErrors(substitutions=1, reference_words=2)  # "270" vs "275": one wrong number, two words


def test_corpus_wer_weights_by_length():
    pairs = [("climb", "descend"), (REF, REF)]  # 1 error in 7 words, not average(100%, 0%) = 50%
    assert corpus_wer(pairs) == pytest.approx(1 / 7)
