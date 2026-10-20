"""Validator tests: no model involved, so they are fast and fully repeatable.

We feed hand-written JSON (exactly what a model might return) and check the decision.
"""

import pytest
from pydantic import ValidationError

from harness.schema import CommandBatch
from harness.validator import validate


def decide(json_text: str) -> list[str]:
    return validate(CommandBatch.model_validate_json(json_text))


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


@pytest.mark.parametrize("json_text", [
    '{"commands":[{"action":"self_destruct"}]}',                           # unknown action
    '{"commands":[{"action":"set_altitude","altitude_ft":"five thousand"}]}',  # wrong type
    '{"commands":[{"action":"set_altitude"}]}',                            # missing field
    '{"commands":[{"action":"set_altitude","altitude_ft":5000',           # truncated JSON
])
def test_schema_rejects_malformed_output(json_text):
    with pytest.raises(ValidationError):
        CommandBatch.model_validate_json(json_text)
