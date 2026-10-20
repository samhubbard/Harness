"""Validator tests: no model involved, so they are fast and fully repeatable.

We feed hand-written JSON (exactly what a model might return) and check the decision.
"""

import pytest
from pydantic import ValidationError

from harness.schema import CommandBatch
from harness.state import AircraftState
from harness.validator import validate


def decide(json_text: str, heading_deg: int = 360, altitude_ft: int = 0) -> list[str]:
    return validate(CommandBatch.model_validate_json(json_text), AircraftState(heading_deg=heading_deg, altitude_ft=altitude_ft))


@pytest.mark.parametrize("json_text", [
    '{"commands":[{"action":"set_altitude","altitude_ft":5000}]}',
    '{"commands":[{"action":"set_airspeed","airspeed_kt":0}]}',  # hover is legal
    '{"commands":[{"action":"set_heading","heading_deg":360}]}',
    '{"commands":[{"action":"set_comm_frequency","radio":"VHF","frequency_mhz":121.5}]}',
    '{"commands":[{"action":"set_heading","heading_deg":270},{"action":"set_altitude","altitude_ft":3000}]}',
])
def test_accepts_valid_commands(json_text):
    assert decide(json_text) == []


@pytest.mark.parametrize("json_text, expected", [
    ('{"commands":[{"action":"set_altitude","altitude_ft":25000}]}', "altitude_ft"),
    ('{"commands":[{"action":"set_heading","heading_deg":0}]}', "heading_deg"),
    ('{"commands":[{"action":"set_airspeed","airspeed_kt":999}]}', "airspeed_kt"),
    ('{"commands":[{"action":"set_comm_frequency","radio":"UHF","frequency_mhz":121.5}]}', "UHF"),
    ('{"commands":[{"action":"clarify","question":"Say altitude."}]}', "clarification"),
    ('{"commands":[]}', "no commands"),
    ('{"commands":[{"action":"set_altitude","altitude_ft":3000},{"action":"set_altitude","altitude_ft":5000}]}', "duplicate"),
])
def test_rejects_invalid_commands(json_text, expected):
    reasons = decide(json_text)
    assert reasons and expected in reasons[0]


def test_one_bad_command_rejects_whole_batch():
    reasons = decide('{"commands":[{"action":"set_heading","heading_deg":90},{"action":"set_airspeed","airspeed_kt":999}]}')
    assert len(reasons) == 1 and "airspeed_kt" in reasons[0]


@pytest.mark.parametrize("current, target, direction, accepted", [
    (360, 180, None, False),     # reversal, no direction: ambiguous
    (360, 180, "left", True),    # reversal, direction stated
    (270, 90, None, False),      # reversal across north
    (360, 179, None, True),      # 179° has a shortest side
    (10, 350, None, True),       # 20° left across north
])
def test_reversal_needs_turn_direction(current, target, direction, accepted):
    command = {"action": "set_heading", "heading_deg": target, "turn_direction": direction}
    reasons = decide(CommandBatch(commands=[command]).model_dump_json(), heading_deg=current)
    assert (reasons == []) == accepted


@pytest.mark.parametrize("current, target, direction, accepted", [
    (5000, 1000, "climb", False),
    (5000, 1000, "descend", True),
    (5000, 10000, "climb", True),
    (5000, 10000, "descend", False),
    (1000, 5000, None, True),
    (5000, 5000, "climb", False)
])
def test_check_altitude_request(current, target, direction, accepted):
    command = {"action": "set_altitude", "altitude_ft": target, "altitude_direction": direction}
    reasons = decide(CommandBatch(commands=[command]).model_dump_json(), altitude_ft=current)
    assert (reasons == []) == accepted


@pytest.mark.parametrize("json_text", [
    '{"commands":[{"action":"self_destruct"}]}',                           # unknown action
    '{"commands":[{"action":"set_altitude","altitude_ft":"five thousand"}]}',  # wrong type
    '{"commands":[{"action":"set_altitude"}]}',                            # missing field
    '{"commands":[{"action":"set_altitude","altitude_ft":5000',           # truncated JSON
])
def test_schema_rejects_malformed_output(json_text):
    with pytest.raises(ValidationError):
        CommandBatch.model_validate_json(json_text)
