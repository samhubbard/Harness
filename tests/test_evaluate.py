"""The scorer must be right, and so must the golden data: a wrong label silently corrupts every metric."""

import pytest

from harness.evaluate import (
    CORRECT, FALSE_REJECT, UNSAFE_ACCEPT, WRONG_COMMAND, load_cases, normalize, score,
)
from harness.schema import CommandBatch

CLIMB = [{"action": "set_altitude", "altitude_ft": 5000, "altitude_direction": "climb"}]


@pytest.mark.parametrize("expect, outcome, resolved, verdict", [
    ("accept", "accepted", CLIMB, CORRECT),
    ("accept", "accepted", [{"action": "set_altitude", "altitude_ft": 4000, "altitude_direction": "climb"}], WRONG_COMMAND),
    ("accept", "rejected", None, FALSE_REJECT),
    ("reject", "rejected", None, CORRECT),
    ("reject", "accepted", CLIMB, UNSAFE_ACCEPT),
])
def test_score(expect, outcome, resolved, verdict):
    case = {"expect": expect, "commands": CLIMB}
    assert score(case, {"outcome": outcome, "resolved": resolved}) == verdict


def test_normalize_ignores_order_and_unset_fields():
    a = [{"action": "set_airspeed", "airspeed_kt": 80}, {"action": "set_heading", "heading_deg": 90, "turn_direction": None}]
    b = [{"heading_deg": 90, "action": "set_heading"}, {"action": "set_airspeed", "airspeed_kt": 80}]
    assert normalize(a) == normalize(b)


@pytest.mark.parametrize("case", load_cases(), ids=lambda case: case["id"])
def test_golden_data_is_well_formed(case):
    assert case["expect"] in ("accept", "reject")
    if case["expect"] == "accept":
        CommandBatch.model_validate({"commands": case["commands"]})  # expected commands must be real commands
