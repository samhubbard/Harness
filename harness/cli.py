"""Interactive command loop.

Run:  uv run python -m harness.cli
"""

from functools import partial

from harness.agent import interpret_with_tools
from harness.pipeline import process
from harness.state import AircraftState


def ask_confirm(readback_text: str) -> bool:
    answer = input(f'  Readback: "{readback_text}". Execute? [y/N] ')
    return answer.strip().lower() == "y"  # anything but "y" means no: the safe default


def main() -> None:
    state = AircraftState()
    # partial() pre-fills `state`, turning interpret_with_tools(text, state) into model(text):
    # the shape process() expects. Same object every call, so the tools always see the live state.
    model = partial(interpret_with_tools, state=state)
    print("Simulated CH-47 command harness (synthetic only). Empty line to quit.")
    while text := input("\n> ").strip():
        result = process(text, state, confirm=ask_confirm, model=model)
        print(f"  {result['outcome'].upper()}", *result.get("reasons", []), sep="\n  - ")
        print(f"  state: {state}")


# Runs only when executed directly, not when imported: Python's equivalent of Main().
if __name__ == "__main__":
    main()
