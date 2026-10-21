"""Simulated aircraft state. Synthetic only: no real equipment or simulator.

A dataclass, not Pydantic: this data is created by our own code, so it is trusted and needs
no parsing. Pydantic guards the trust boundary (model output); dataclasses hold internal state.
"""

from dataclasses import dataclass, field

from harness.schema import (
    ChangeAltitudeBy, Clarify, CommandBatch, SetAirspeed, SetAltitude, SetFrequency, SetHeading, TurnBy,
)


@dataclass
class AircraftState:
    heading_deg: int = 360
    altitude_ft: int = 0
    airspeed_kt: int = 0
    # default_factory: each instance gets its OWN dict (a plain `= {...}` default would be shared).
    radios_mhz: dict[str, float] = field(default_factory=lambda: {"VHF": 121.5, "UHF": 243.0})

    def apply(self, batch: CommandBatch) -> None:
        """Apply an already-validated batch. Never call this on unvalidated input."""
        for cmd in batch.commands:
            match cmd:
                case SetAltitude():
                    self.altitude_ft = cmd.altitude_ft
                case SetHeading():
                    self.heading_deg = cmd.heading_deg
                case SetAirspeed():
                    self.airspeed_kt = cmd.airspeed_kt
                case SetFrequency():
                    self.radios_mhz[cmd.radio] = cmd.frequency_mhz
                case Clarify():
                    raise ValueError("clarify is not executable; the validator should have rejected it")
                case TurnBy() | ChangeAltitudeBy():
                    raise ValueError("relative commands must be resolved before they are applied")
