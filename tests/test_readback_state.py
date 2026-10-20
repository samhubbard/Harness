"""Readback wording and state updates: deterministic, no model."""

from harness.readback import readback
from harness.schema import CommandBatch
from harness.state import AircraftState


def batch(*commands: dict) -> CommandBatch:
    return CommandBatch(commands=list(commands))


def test_readback_says_climb_or_descend_from_current_state():
    state = AircraftState(altitude_ft=3000)
    assert readback(batch({"action": "set_altitude", "altitude_ft": 5000}), state) == "climb and maintain 5,000 feet"
    assert readback(batch({"action": "set_altitude", "altitude_ft": 1000}), state) == "descend and maintain 1,000 feet"


def test_readback_keeps_turn_direction_and_pads_heading():
    state = AircraftState()
    assert readback(batch({"action": "set_heading", "heading_deg": 90, "turn_direction": "left"}), state) == "turn left heading 090"
    assert readback(batch({"action": "set_heading", "heading_deg": 90}), state) == "fly heading 090"


def test_readback_joins_multiple_commands():
    commands = batch(
        {"action": "set_heading", "heading_deg": 270, "turn_direction": "left"},
        {"action": "set_comm_frequency", "radio": "VHF", "frequency_mhz": 124.35},
    )
    assert readback(commands, AircraftState()) == "turn left heading 270, tune VHF 124.350"


def test_apply_updates_only_commanded_fields():
    state = AircraftState()
    state.apply(batch({"action": "set_altitude", "altitude_ft": 3000}, {"action": "set_comm_frequency", "radio": "UHF", "frequency_mhz": 251.0}))
    assert state.altitude_ft == 3000
    assert state.radios_mhz == {"VHF": 121.5, "UHF": 251.0}
    assert state.heading_deg == 360  # untouched


def test_each_state_gets_its_own_radios():
    first, second = AircraftState(), AircraftState()
    first.radios_mhz["VHF"] = 130.0
    assert second.radios_mhz["VHF"] == 121.5  # proves default_factory, not a shared default
