"""Deterministic guardrails: plain Python rules, no model involved.

The model proposes; this module decides. A batch is all-or-nothing: if any command
fails, nothing is applied (like a pilot refusing a clearance they can't fully comply with).
"""

from harness.schema import Clarify, CommandBatch, SetAirspeed, SetAltitude, SetFrequency, SetHeading

# FICTIONAL training limits, not real CH-47F data.
ALTITUDE_FT = (0, 15_000)
AIRSPEED_KT = (0, 160)  # 0 is legal: helicopters hover
HEADING_DEG = (1, 360)  # 360 = north; 000 is not issued
RADIO_BANDS_MHZ = {"VHF": (118.000, 136.975), "UHF": (225.000, 399.975)}
MAX_COMMANDS = 4


def _out_of_range(name: str, value: float, limits: tuple[float, float]) -> str | None:
    low, high = limits
    if low <= value <= high:
        return None
    return f"{name} {value} outside allowed range {low}-{high}"


def _check_command(cmd) -> str | None:
    match cmd:
        case SetAltitude():
            return _out_of_range("altitude_ft", cmd.altitude_ft, ALTITUDE_FT)
        case SetHeading():
            return _out_of_range("heading_deg", cmd.heading_deg, HEADING_DEG)
        case SetAirspeed():
            return _out_of_range("airspeed_kt", cmd.airspeed_kt, AIRSPEED_KT)
        case SetFrequency():
            # Cross-field rule: the valid range depends on which radio was chosen.
            return _out_of_range(f"{cmd.radio} frequency_mhz", cmd.frequency_mhz, RADIO_BANDS_MHZ[cmd.radio])
        case Clarify():
            return f"model requested clarification: {cmd.question}"
    return f"unknown command type {type(cmd).__name__}"


def validate(batch: CommandBatch) -> list[str]:
    """Return a list of rejection reasons. Empty list means the whole batch is accepted."""
    if not batch.commands:
        return ["no commands found"]
    if len(batch.commands) > MAX_COMMANDS:
        return [f"{len(batch.commands)} commands exceeds limit of {MAX_COMMANDS}"]

    actions = [cmd.action for cmd in batch.commands]
    duplicates = sorted({a for a in actions if actions.count(a) > 1})
    if duplicates:
        return [f"conflicting duplicate actions: {', '.join(duplicates)}"]

    return [reason for cmd in batch.commands if (reason := _check_command(cmd))]
