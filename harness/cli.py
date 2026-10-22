"""Interactive command loop.

Run:  uv run python -m harness.cli            (type commands)
      uv run python -m harness.cli --voice    (speak commands: Windows' built-in recognizer, default microphone)
"""

import sys
from functools import partial

from harness.agent import interpret_with_tools
from harness.pipeline import process
from harness.speech import listen
from harness.state import AircraftState


def ask_confirm(readback_text: str) -> bool:
    answer = input(f'  Readback: "{readback_text}". Execute? [y/N] ')
    return answer.strip().lower() == "y"  # anything but "y" means no: the safe default


def typed_commands():
    while text := input("\n> ").strip():
        yield text, None  # typed input has no recognizer confidence


def spoken_commands():
    while input("\nPress Enter, then speak a command (q + Enter to quit) ").strip().lower() != "q":
        heard = listen()
        print(f'  Heard: "{heard.text}" (confidence {heard.confidence:.2f})')
        yield heard.text, heard.confidence


def main() -> None:
    state = AircraftState()
    # partial() pre-fills `state`, turning interpret_with_tools(text, state) into model(text):
    # the shape process() expects. Same object every call, so the tools always see the live state.
    model = partial(interpret_with_tools, state=state)
    voice = "--voice" in sys.argv
    print("Simulated CH-47 command harness (synthetic only).", "Voice mode." if voice else "Empty line to quit.")
    for text, confidence in spoken_commands() if voice else typed_commands():
        result = process(text, state, confirm=ask_confirm, model=model, confidence=confidence)
        print(f"  {result['outcome'].upper()}", *result.get("reasons", []), sep="\n  - ")
        print(f"  state: {state}")


# Runs only when executed directly, not when imported: Python's equivalent of Main().
if __name__ == "__main__":
    main()
