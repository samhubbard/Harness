"""Step 5: tool calling on the wire, one round trip at a time.

The model can't run anything. It replies with a *request* to call a tool; this script runs it,
appends the result as a "tool" message, and resends the WHOLE conversation (the API is stateless).
Watch prompt_tokens grow every round.

Run:  uv run examples/05_tool_calling.py "Tune Alpha Tower and climb another 2000 feet"
"""

import json
import sys

import httpx

from harness.schema import CommandBatch

ENDPOINT = "http://localhost:11434/v1/chat/completions"
MAX_ROUNDS = 4

# Fictional, synthetic data only.
STATE = {"heading_deg": 360, "altitude_ft": 3000, "airspeed_kt": 90, "radios_mhz": {"VHF": 121.5, "UHF": 243.0}}
DIRECTORY = {"alpha tower": ("VHF", 124.35), "bravo approach": ("UHF", 285.4)}


def tool(name: str, description: str, parameters: dict) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": parameters}}


TOOLS = [
    tool("get_aircraft_state", "Current heading, altitude, airspeed and radio frequencies.",
         {"type": "object", "properties": {}}),
    tool("lookup_frequency", "Radio and frequency for a named facility.",
         {"type": "object", "properties": {"facility": {"type": "string"}}, "required": ["facility"]}),
    tool("submit_commands", "Submit the final commands. Call exactly once, last.",
         CommandBatch.model_json_schema()),
]


def run_tool(name: str, args: dict) -> dict:
    """The dispatcher. The model only ever sees what this returns."""
    match name:
        case "get_aircraft_state":
            return STATE
        case "lookup_frequency":
            radio, mhz = DIRECTORY.get(args.get("facility", "").lower(), (None, None))
            return {"radio": radio, "frequency_mhz": mhz} if radio else {"error": "unknown facility"}
    return {"error": f"unknown tool {name}"}


messages = [
    {"role": "system", "content": "You convert flight commands for a SIMULATED helicopter. "
                                  "Use tools to look up anything you don't know, then call submit_commands."},
    {"role": "user", "content": sys.argv[1] if len(sys.argv) > 1 else "Tune Alpha Tower and climb another 2000 feet"},
]

for round_no in range(1, MAX_ROUNDS + 1):
    body = {"model": "qwen3.5:9b", "messages": messages, "tools": TOOLS, "temperature": 0, "reasoning_effort": "none"}
    envelope = httpx.post(ENDPOINT, json=body, timeout=120).json()
    choice = envelope["choices"][0]
    reply = choice["message"]

    print("=" * 25, f"ROUND {round_no}: finish_reason={choice['finish_reason']}, "
          f"prompt_tokens={envelope['usage']['prompt_tokens']}", "=" * 25)
    print(json.dumps(reply, indent=2))

    messages.append(reply)  # the model's own tool request becomes part of the history
    calls = reply.get("tool_calls") or []
    if not calls:
        break

    for call in calls:
        name = call["function"]["name"]
        args = json.loads(call["function"]["arguments"])  # parse #3: arguments is a JSON *string*
        if name == "submit_commands":
            print("\nPROPOSED (would go to validator -> readback -> y/N):", json.dumps(args))
            sys.exit()
        result = run_tool(name, args)
        print(f"\n  -> ran {name}({args}) = {result}")
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": json.dumps(result)})
else:
    print("\nRound cap reached without submit_commands: fail closed.")
