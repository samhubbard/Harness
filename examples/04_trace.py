"""Step 4: trace ONE command through every stage and print the real data at each step.

Run:  uv run examples/04_trace.py "Turn left heading two seven zero and climb to 3,000."
"""

import json
import sys

from harness.llm import build_request, parse_reply, send
from harness.state import AircraftState
from harness.validator import validate

text = sys.argv[1] if len(sys.argv) > 1 else "Turn left heading two seven zero and climb to 3,000."


def stage(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * (70 - len(title))}")


stage("0. USER INPUT (a plain Python string)")
print(repr(text))

stage("1. REQUEST BODY we will POST (schema shown separately below)")
body = build_request(text)
schema = body["response_format"]["json_schema"]["schema"]
print(json.dumps({**body, "response_format": "<see 1b>"}, indent=2))

stage("1b. JSON SCHEMA generated from harness/schema.py")
print(json.dumps(schema, indent=2))

stage("2. RESPONSE ENVELOPE from the server (after parse #1)")
envelope = send(body)
print(json.dumps(envelope, indent=2))

stage("3. message.content: what the model wrote (still a str)")
content = envelope["choices"][0]["message"]["content"]
print(type(content).__name__, "->", repr(content))

stage("4. PARSED into Pydantic objects (parse #2)")
batch = parse_reply(envelope)
for cmd in batch.commands:
    print(f"{type(cmd).__name__:<14} {cmd!r}")

stage("5. VALIDATOR decision")
reasons = validate(batch, AircraftState())
print("ACCEPTED" if not reasons else f"REJECTED: {reasons}")
