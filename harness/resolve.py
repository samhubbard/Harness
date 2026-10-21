"""Turn relative commands into absolute ones. Code does the arithmetic, never the model.

The model is good at language ("turn 30 right" means turn_by 30, right) and unreliable at math
(it once turned 360 + 30 into 330). So it transcribes the amount and this module computes the target.
"""

from harness.schema import ChangeAltitudeBy, CommandBatch, SetAltitude, SetHeading, TurnBy
from harness.state import AircraftState


def resolve(batch: CommandBatch, state: AircraftState) -> CommandBatch:
    return CommandBatch(commands=[resolve_command(cmd, state) for cmd in batch.commands])


def resolve_command(cmd, state: AircraftState):
    """Absolute commands pass through unchanged."""
    match cmd:
        case TurnBy():
            sign = 1 if cmd.turn_direction == "right" else -1
            heading = (state.heading_deg + sign * cmd.degrees) % 360 or 360  # 000 is issued as 360
            return SetHeading(action="set_heading", heading_deg=heading, turn_direction=cmd.turn_direction)
        case ChangeAltitudeBy():
            sign = 1 if cmd.altitude_direction == "climb" else -1
            altitude = state.altitude_ft + sign * cmd.feet
            return SetAltitude(action="set_altitude", altitude_ft=altitude, altitude_direction=cmd.altitude_direction)
    return cmd
