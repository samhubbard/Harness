"""Evaluation: run labelled commands through the real pipeline and score them.

Answers "is it good enough, and did my change make it better or worse?" with numbers instead of
impressions. Re-run it after every prompt edit, model swap or fine-tune.

Run:  uv run python -m harness.evaluate            (both model paths)
      uv run python -m harness.evaluate agent      (one path)
"""

import json
import sys
import time
from collections import Counter
from functools import partial
from pathlib import Path

from harness.agent import interpret_with_tools
from harness.llm import interpret
from harness.pipeline import process
from harness.state import AircraftState

DATASET = Path("evals/commands.jsonl")
AUDIT_LOG = Path("logs/eval_audit.jsonl")  # kept apart from the interactive audit log
BASE_STATE = {"heading_deg": 360, "altitude_ft": 3000, "airspeed_kt": 90}
MODEL_PATHS = ("structured", "agent")

# Scores, worst first. An unsafe accept is the one that matters most: the guardrails let a bad command through.
UNSAFE_ACCEPT = "unsafe_accept"    # should have been rejected, was accepted
WRONG_COMMAND = "wrong_command"    # accepted, but not what was said: valid-but-wrong, only a human catches it
FALSE_REJECT = "false_reject"      # a clear command refused: annoying, not dangerous
CORRECT = "correct"


def load_cases(path: Path = DATASET) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def normalize(commands: list[dict]) -> list[str]:
    """Ignore field order, command order and unset (None) fields: compare meaning, not formatting."""
    return sorted(json.dumps({k: v for k, v in cmd.items() if v is not None}, sort_keys=True) for cmd in commands)


def score(case: dict, result: dict) -> str:
    accepted = result["outcome"] == "accepted"
    if case["expect"] == "reject":
        return UNSAFE_ACCEPT if accepted else CORRECT
    if not accepted:
        return FALSE_REJECT
    return CORRECT if normalize(result["resolved"]) == normalize(case["commands"]) else WRONG_COMMAND


def run_case(case: dict, model_path: str) -> dict:
    state = AircraftState(**{**BASE_STATE, **case.get("state", {})})
    model = interpret if model_path == "structured" else partial(interpret_with_tools, state=state)
    started = time.perf_counter()
    # confirm always says yes: we are measuring the machine, so the human safety net is switched off.
    result = process(case["input"], state, confirm=lambda spoken: True, model=model, log_path=AUDIT_LOG)
    return {**result, "seconds": time.perf_counter() - started}


def evaluate(model_path: str, cases: list[dict]) -> Counter:
    print(f"\n=== {model_path} ({len(cases)} cases) ===")
    scores, seconds = Counter(), 0.0
    for case in cases:
        result = run_case(case, model_path)
        verdict = score(case, result)
        scores[verdict] += 1
        seconds += result["seconds"]
        if verdict != CORRECT:
            got = result.get("resolved") or result.get("reasons") or result.get("commands")
            print(f"  {verdict:14} {case['id']:20} got: {json.dumps(got)}")
    accuracy = scores[CORRECT] / len(cases)
    print(f"  accuracy {accuracy:.0%}  |  " + "  ".join(f"{k}={scores[k]}" for k in (UNSAFE_ACCEPT, WRONG_COMMAND, FALSE_REJECT))
          + f"  |  avg {seconds / len(cases):.1f}s/command")
    return scores


def main() -> None:
    cases = load_cases()
    for model_path in sys.argv[1:] or MODEL_PATHS:
        evaluate(model_path, cases)


if __name__ == "__main__":
    main()
