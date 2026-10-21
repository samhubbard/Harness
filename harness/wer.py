"""Word Error Rate: the standard accuracy measure for speech recognition (ASR).

WER = (substitutions + deletions + insertions) / words in the reference
  reference  (what was said):  turn right heading zero niner zero
  hypothesis (what ASR heard): turn right heading zero nine zero      -> 1 substitution / 6 words = 16.7%

It is the minimum number of word edits to turn the hypothesis into the reference (edit distance,
like a diff), so it can exceed 100% when the recognizer adds lots of extra words.
"""

import re
from dataclasses import dataclass, replace


@dataclass(frozen=True)
class WordErrors:
    substitutions: int = 0
    deletions: int = 0    # said, but missing from the transcript
    insertions: int = 0   # in the transcript, but never said
    reference_words: int = 0

    @property
    def total(self) -> int:
        return self.substitutions + self.deletions + self.insertions

    @property
    def rate(self) -> float:
        return self.total / max(self.reference_words, 1)

    def add(self, **counts: int) -> "WordErrors":
        return replace(self, **{name: getattr(self, name) + n for name, n in counts.items()})


def normalize(text: str) -> list[str]:
    """Lowercase and drop punctuation, so "Heading," and "heading" match. Real systems also normalize
    numbers ("5000" vs "five thousand"); without that, formatting differences count as errors."""
    return re.sub(r"[^a-z0-9.' ]+", " ", text.lower()).replace(". ", " ").strip(" .").split()


def word_errors(reference: str, hypothesis: str) -> WordErrors:
    """Edit distance over words (dynamic programming): row i holds the cheapest way to match the
    first i reference words against each prefix of the hypothesis."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    previous = [WordErrors(insertions=j) for j in range(len(hyp) + 1)]
    for i, ref_word in enumerate(ref, start=1):
        current = [WordErrors(deletions=i)]
        for j, hyp_word in enumerate(hyp, start=1):
            diagonal = previous[j - 1] if ref_word == hyp_word else previous[j - 1].add(substitutions=1)
            current.append(min(diagonal, previous[j].add(deletions=1), current[j - 1].add(insertions=1),
                               key=lambda errors: errors.total))
        previous = current
    return replace(previous[-1], reference_words=len(ref))


def wer(reference: str, hypothesis: str) -> float:
    return word_errors(reference, hypothesis).rate


def corpus_wer(pairs: list[tuple[str, str]]) -> float:
    """Total errors / total reference words. Not the average of per-sentence WERs, which would let
    a 2-word sentence count as much as a 20-word one."""
    results = [word_errors(ref, hyp) for ref, hyp in pairs]
    return sum(r.total for r in results) / max(sum(r.reference_words for r in results), 1)
