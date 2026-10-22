import pytest

from harness.numbers import collapse_numbers, spoken_numbers, words


@pytest.mark.parametrize("text, expected", [
    ("Turn right heading zero niner zero", [90]),
    ("Fly heading one two zero", [120]),
    ("Climb and maintain two thousand five hundred", [2500]),
    ("Climb and maintain twenty five thousand", [25000]),
    ("Climb and maintain five thousand", [5000]),
    ("Speed one hundred knots", [100]),
    ("Turn thirty degrees right", [30]),
    ("Tune UHF two eight five point four", [285.4]),
    ("Turn left adding 060", [60]),
    ("Climb 5,000 feet", [5000]),
    ("Heading two four zero, speed one two zero", [240, 120]),
    ("Turn right heading one five zero, descend and maintain two thousand five hundred, slow to eighty knots",
     [150, 2500, 80]),
    ("Climb a little", []),
])
def test_spoken_numbers(text, expected):
    assert spoken_numbers(text) == expected


def test_collapse_keeps_other_words():
    assert collapse_numbers(words("Heading two seven zero, climb 3,000.")) == ["heading", "270", "climb", "3000"]


def test_degree_sign_reads_as_the_word():
    assert words("Turn 30° right") == ["turn", "30", "degrees", "right"]  # how dictation writes it


def test_point_outside_a_number_is_a_word():
    assert collapse_numbers(["point", "two"]) == ["point", "2"]
