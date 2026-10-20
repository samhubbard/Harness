"""The contract between the model and the application.

These Pydantic classes do two jobs from one definition:
  1. Generate the JSON Schema we send to the model (CommandBatch.model_json_schema()).
  2. Parse and type-check the model's reply (CommandBatch.model_validate_json(...)).
One source of truth means the prompt and the parser can never drift apart.

Deliberately NO range limits here (no max altitude, etc.). If the schema told the
model "altitude <= 15000", a request for 25,000 ft could be silently rewritten to a
legal value. We want the model to transcribe faithfully and the validator to refuse.
"""

from typing import Annotated, Literal, Union

from pydantic import BaseModel, Field


class SetAltitude(BaseModel):
    action: Literal["set_altitude"]
    altitude_ft: int
    # Optional on purpose "climb, descend are not necessarily required here."
    altitude_direction: Literal["climb", "descend"] | None = None


class SetHeading(BaseModel):
    action: Literal["set_heading"]
    heading_deg: int
    # Optional on purpose: "fly heading 270" has no direction. Required would force the model to invent one.
    turn_direction: Literal["left", "right"] | None = None


class SetAirspeed(BaseModel):
    action: Literal["set_airspeed"]
    airspeed_kt: int


class SetFrequency(BaseModel):
    action: Literal["set_comm_frequency"]
    radio: Literal["VHF", "UHF"]
    frequency_mhz: float


class Clarify(BaseModel):
    """The model's way to say "I'm not sure what you meant" instead of guessing."""

    action: Literal["clarify"]
    question: str


# "Discriminated union": the `action` value decides which class applies,
# like a C# switch on a type tag. Each action carries only its own fields.
Command = Annotated[
    Union[SetAltitude, SetHeading, SetAirspeed, SetFrequency, Clarify],
    Field(discriminator="action"),
]


class CommandBatch(BaseModel):
    commands: list[Command]
