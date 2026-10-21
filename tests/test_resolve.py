"""Relative -> absolute: the arithmetic the model is no longer trusted with."""

import pytest

from harness.resolve import resolve_command
from harness.schema import ChangeAltitudeBy, SetAirspeed, SetAltitude, SetHeading, TurnBy
from harness.state import AircraftState


@pytest.mark.parametrize("current, degrees, direction, expected", [
    (360, 30, "right", 30),   # the live bug: the model said 330
    (360, 30, "left", 330),
    (350, 20, "right", 10),   # across north
    (10, 10, "left", 360),    # lands on north: 360, never 000
    (90, 180, "right", 270),
])
def test_turn_by_resolves_heading(current, degrees, direction, expected):
    cmd = resolve_command(TurnBy(action="turn_by", degrees=degrees, turn_direction=direction),
                          AircraftState(heading_deg=current))
    assert cmd == SetHeading(action="set_heading", heading_deg=expected, turn_direction=direction)


@pytest.mark.parametrize("current, feet, direction, expected", [
    (3000, 2000, "climb", 5000),
    (3000, 1000, "descend", 2000),
])
def test_change_altitude_by_resolves_altitude(current, feet, direction, expected):
    cmd = resolve_command(ChangeAltitudeBy(action="change_altitude_by", feet=feet, altitude_direction=direction),
                          AircraftState(altitude_ft=current))
    assert cmd == SetAltitude(action="set_altitude", altitude_ft=expected, altitude_direction=direction)


def test_absolute_commands_pass_through():
    cmd = SetAirspeed(action="set_airspeed", airspeed_kt=90)
    assert resolve_command(cmd, AircraftState()) is cmd
