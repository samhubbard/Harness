"""Deterministic guardrails: plain Python rules, no model involved.

The model proposes; this module decides. A batch is all-or-nothing: if any command
fails, nothing is applied (like a pilot refusing a clearance they can't fully comply with).
"""

from harness.numbers import spoken_numbers, words
from harness.resolve import resolve
from harness.schema import (
    ChangeAltitudeBy, Clarify, CommandBatch, SetAirspeed, SetAltitude, SetFrequency, SetHeading, TurnBy,
)
from harness.state import AircraftState

# FICTIONAL training limits, not real CH-47F data.
ALTITUDE_FT = (0, 15_000)
AIRSPEED_KT = (0, 160)  # 0 is legal: helicopters hover
HEADING_DEG = (1, 360)  # 360 = north; 000 is not issued
RADIO_BANDS_MHZ = {"VHF": (118.000, 136.975), "UHF": (225.000, 399.975)}
MAX_COMMANDS = 4
TURN_BY_DEG = (1, 180)  # a bigger relative turn is unusual enough to say the heading instead
ALTITUDE_CHANGE_FT = (1, 15_000)
RELATIVE_ALTITUDE_WORDS = {"another", "by", "more"}  # "climb another 2000", "descend by 500", "500 more"


def _out_of_range(name: str, value: float, limits: tuple[float, float]) -> str | None:
    low, high = limits
    if low <= value <= high:
        return None
    return f"{name} {value} outside allowed range {low}-{high}"


def _shortest_turn_deg(current: int, target: int) -> int:
    delta = (target - current) % 360
    return min(delta, 360 - delta)


def _turn_deg(current: int, target: int, direction: str) -> int:
    """How far the aircraft actually turns when it goes the stated way."""
    delta = (target - current) % 360
    return delta if direction == "right" else (360 - delta) % 360


def _check_heading(cmd: SetHeading, state: AircraftState) -> str | None:
    if reason := _out_of_range("heading_deg", cmd.heading_deg, HEADING_DEG):
        return reason
    # State-aware rule: a 180° reversal has no "shortest" side, so the direction must be stated.
    if cmd.turn_direction is None and _shortest_turn_deg(state.heading_deg, cmd.heading_deg) == 180:
        return f"heading {cmd.heading_deg:03d} is a 180° turn from {state.heading_deg:03d}: say turn left or right"
    # Long-way rule: 360 "turn right heading 330" is a 330° turn, almost always a slip. Ask, don't fly it.
    if cmd.turn_direction and (turn := _turn_deg(state.heading_deg, cmd.heading_deg, cmd.turn_direction)) > 180:
        return (f"turn {cmd.turn_direction} heading {cmd.heading_deg:03d} from {state.heading_deg:03d} "
                f"is a {turn}° turn the long way: confirm direction")
    return None


def _check_altitude(cmd: SetAltitude, state: AircraftState) -> str | None:
    if reason := _out_of_range("altitude_ft", cmd.altitude_ft, ALTITUDE_FT):
        return reason
    if cmd.altitude_direction == "climb" and state.altitude_ft >= cmd.altitude_ft:
        return f"climb to {cmd.altitude_ft:,} ft but already at {state.altitude_ft:,} ft"
    elif cmd.altitude_direction == "descend" and state.altitude_ft <= cmd.altitude_ft:
        return f"descend to {cmd.altitude_ft:,} ft but already at {state.altitude_ft:,} ft"
    return None


def _check_command(cmd, state: AircraftState) -> str | None:
    match cmd:
        case SetAltitude():
            return _check_altitude(cmd, state)
        case SetHeading():
            return _check_heading(cmd, state)
        case SetAirspeed():
            return _out_of_range("airspeed_kt", cmd.airspeed_kt, AIRSPEED_KT)
        case SetFrequency():
            # Cross-field rule: the valid range depends on which radio was chosen.
            return _out_of_range(f"{cmd.radio} frequency_mhz", cmd.frequency_mhz, RADIO_BANDS_MHZ[cmd.radio])
        case Clarify():
            return f"model requested clarification: {cmd.question}"
    return f"unknown command type {type(cmd).__name__}"


def _check_relative(cmd) -> str | None:
    """Check the amount as spoken BEFORE doing math with it, so a wild value can't wrap into a legal one
    (turn 390 right would otherwise quietly become a 30° turn)."""
    match cmd:
        case TurnBy():
            return _out_of_range("turn_by degrees", cmd.degrees, TURN_BY_DEG)
        case ChangeAltitudeBy():
            return _out_of_range("change_altitude_by feet", cmd.feet, ALTITUDE_CHANGE_FT)
    return None


def _spoken_value(cmd) -> int | None:
    """The number the speaker must have said for this command. Frequencies are left out on purpose:
    a facility lookup supplies them, not the speaker (the band check still covers them)."""
    match cmd:
        case SetHeading():
            return cmd.heading_deg
        case SetAltitude():
            return cmd.altitude_ft
        case SetAirspeed():
            return cmd.airspeed_kt
        case TurnBy():
            return cmd.degrees
        case ChangeAltitudeBy():
            return cmd.feet
    return None


def check_wording(text: str, batch: CommandBatch) -> list[str]:
    """Cross-check the commands against the words actually said: the rules that need the input text.

    1. Every number in a command must appear in what was said. The model transcribes; this verifies the
       transcription ("two thousand five hundred" read as 500 is caught here, not by a limit check).
    2. Absolute and relative instructions must not be confused. "Heading 060" is not "60 degrees", and
       "descend and maintain 2,500" is not "descend another 2,500" (both mix-ups were seen live). A relative
       command is only believed if the speaker used a relative word; otherwise ask, don't guess."""
    heard, said = spoken_numbers(text), set(words(text))
    reasons = [f"say again: {value} is not in what was said" for cmd in batch.commands
               if (value := _spoken_value(cmd)) is not None and value not in heard]
    if not said & {"degree", "degrees"}:
        reasons += [f"say again: heading or degrees? (heard a {cmd.degrees}° {cmd.turn_direction} turn, but no 'degrees')"
                    for cmd in batch.commands if isinstance(cmd, TurnBy)]
    if not said & RELATIVE_ALTITUDE_WORDS:
        reasons += [f"say again: maintain {cmd.feet} or {cmd.altitude_direction} another {cmd.feet}?"
                    for cmd in batch.commands if isinstance(cmd, ChangeAltitudeBy)]
    return reasons


def validate(batch: CommandBatch, state: AircraftState) -> list[str]:
    """Return a list of rejection reasons. Empty list means the whole batch is accepted."""
    if not batch.commands:
        return ["no commands found"]
    if len(batch.commands) > MAX_COMMANDS:
        return [f"{len(batch.commands)} commands exceeds limit of {MAX_COMMANDS}"]
    if relative_reasons := [reason for cmd in batch.commands if (reason := _check_relative(cmd))]:
        return relative_reasons

    batch = resolve(batch, state)  # from here on every command is absolute, so the rules below cover both
    actions = [cmd.action for cmd in batch.commands]
    duplicates = sorted({a for a in actions if actions.count(a) > 1})
    if duplicates:
        return [f"conflicting duplicate actions: {', '.join(duplicates)}"]

    return [reason for cmd in batch.commands if (reason := _check_command(cmd, state))]
