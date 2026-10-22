"""Spoken numbers -> values, so code can check the model's numbers against the words actually said.

Three ways the same number arrives:
  digit by digit (ATC style):  "one two zero" -> 120, "zero niner zero" -> 90, "one two four point three five" -> 124.35
  everyday words:              "two thousand five hundred" -> 2500, "twenty five thousand" -> 25000
  digits from a recognizer:    "060" -> 60, "5,000" -> 5000
Used by the wording check (validator.check_wording) and by WER normalization (wer.py).
"""

import re

DIGITS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
          "six": 6, "seven": 7, "eight": 8, "nine": 9, "niner": 9}
TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000}
NUMBER_WORDS = DIGITS | TEENS | TENS | SCALES
LITERAL = re.compile(r"\d+(\.\d+)?")


def words(text: str) -> list[str]:
    """Lowercase words without punctuation. Keeps decimal points ("124.35"), drops thousands commas ("5,000"),
    and reads the degree sign as the word, since dictation writes "turn 30° right"."""
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text.lower()).replace("°", " degrees ")
    return re.sub(r"[^a-z0-9.' ]+", " ", text).replace(". ", " ").strip(" .").split()


def collapse_numbers(tokens: list[str]) -> list[str]:
    """Replace every spoken number with one token holding its value: ["heading", "two", "seven", "zero"]
    -> ["heading", "270"]. Everything else passes through unchanged."""
    out, run = [], []
    for i, token in enumerate(tokens):
        next_token = tokens[i + 1] if i + 1 < len(tokens) else ""
        is_decimal_point = token == "point" and run and next_token in DIGITS
        if token in NUMBER_WORDS or is_decimal_point:
            run.append(token)
            continue
        if run:
            out.append(_format(_value(run)))
            run = []
        out.append(_format(float(token)) if LITERAL.fullmatch(token) else token)
    if run:
        out.append(_format(_value(run)))
    return out


def spoken_numbers(text: str) -> list[float]:
    """Every number in the text, as values."""
    return [float(token) for token in collapse_numbers(words(text)) if LITERAL.fullmatch(token)]


def _value(run: list[str]) -> float:
    if all(token in DIGITS or token == "point" for token in run):
        # Digit by digit: concatenate, like reading a display ("zero niner zero" -> "090" -> 90).
        return float("".join("." if token == "point" else str(DIGITS[token]) for token in run))
    # Everyday words: "two thousand five hundred" = (2 x 1000) + (5 x 100).
    total = current = 0
    for token in run:
        if token == "thousand":
            total, current = total + max(current, 1) * 1000, 0
        elif token == "hundred":
            current = max(current, 1) * 100
        else:
            current += NUMBER_WORDS.get(token, 0)
    return float(total + current)


def _format(value: float) -> str:
    return str(int(value)) if value == int(value) else str(value)
