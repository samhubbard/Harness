"""The tool menu the model sees, and the dispatcher that runs the read-only tools.

The model never runs anything. It sends back a tool *name* and JSON *arguments*; run_tool()
decides what happens. A name that isn't handled here runs nothing.
"""

import json
from dataclasses import asdict

from harness.schema import CommandBatch
from harness.state import AircraftState

# Fictional facilities: synthetic data only.
DIRECTORY = {
    "alpha tower": {"radio": "VHF", "frequency_mhz": 124.35},
    "bravo approach": {"radio": "UHF", "frequency_mhz": 285.4},
}

SUBMIT = "submit_commands"  # the one "action" tool: the pipeline handles it, not run_tool()


def _tool(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOLS = [
    _tool("get_aircraft_state", "Current heading, altitude, airspeed and radio frequencies.",
          {"type": "object", "properties": {}}),
    _tool("lookup_frequency", "Radio and frequency for a named facility.",
          {"type": "object", "properties": {"facility": {"type": "string"}}, "required": ["facility"]}),
    _tool(SUBMIT, "Submit the final commands. Call exactly once, last.", CommandBatch.model_json_schema()),
]


def run_tool(name: str, arguments: str, state: AircraftState) -> dict:
    """Run one read-only tool.

    Problems go back to the model as data ({"error": ...}), not exceptions, so it can recover
    (for example, clarify an unknown facility). The agent's round cap stops it retrying forever.
    """
    try:
        args = json.loads(arguments)  # parse #3: arguments arrive as a JSON *string*
    except json.JSONDecodeError:
        return {"error": "arguments were not valid JSON"}
    if not isinstance(args, dict):
        return {"error": "arguments must be a JSON object"}

    match name:
        case "get_aircraft_state":
            return asdict(state)
        case "lookup_frequency":
            facility = str(args.get("facility", "")).strip().lower()
            return DIRECTORY.get(facility, {"error": f"unknown facility: {args.get('facility')}"})
    return {"error": f"unknown tool: {name}"}
