"""One command, end to end: (confidence gate) -> interpret -> check wording + validate -> resolve
-> readback -> confirm -> apply -> audit."""

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import httpx

from harness import audit
from harness.llm import ModelOutputError, interpret
from harness.readback import readback
from harness.resolve import resolve
from harness.schema import CommandBatch
from harness.speech import MIN_CONFIDENCE
from harness.state import AircraftState
from harness.validator import check_wording, validate


def process(
    text: str,
    state: AircraftState,
    confirm: Callable[[str], bool],
    model: Callable[[str], CommandBatch] = interpret,
    log_path: Path = audit.DEFAULT_LOG,
    confidence: float | None = None,
) -> dict:
    """Run one command and return its audit record.

    `confirm` and `model` are passed in (dependency injection), so tests can swap in a fake
    model and an automatic yes/no instead of a live LLM and a human at the keyboard.
    `confidence` is the speech recognizer's score when the text came from a voice (None when typed).
    """
    entry = {"input": text, "state_before": asdict(state)}
    if confidence is not None:
        entry["confidence"] = confidence
        if confidence < MIN_CONFIDENCE or not text.strip():
            # Unclear audio never reaches the model: it would only guess, and a guess can be a valid command.
            return _finish(entry, "say_again", log_path,
                           reasons=[f"say again: recognizer confidence {confidence:.2f} below {MIN_CONFIDENCE}"])
    try:
        batch = model(text)
    except ModelOutputError as err:
        return _finish(entry, "model_error", log_path, reasons=[str(err)])
    except httpx.HTTPError as err:
        return _finish(entry, "unavailable", log_path, reasons=[f"{type(err).__name__}: {err}"])

    entry["commands"] = batch.model_dump()["commands"]
    if batch.via:
        entry["via"] = batch.via
    if reasons := check_wording(text, batch) or validate(batch, state):
        return _finish(entry, "rejected", log_path, reasons=reasons)

    # Relative -> absolute, by code: the same math validate() just checked. Readback and apply
    # only ever see absolute commands. The log keeps both: what the model said, and what we computed.
    batch = resolve(batch, state)
    entry["resolved"] = batch.model_dump()["commands"]

    spoken = readback(batch, state)
    entry["readback"] = spoken
    if not confirm(spoken):
        return _finish(entry, "declined", log_path)

    state.apply(batch)
    return _finish(entry, "accepted", log_path, state_after=asdict(state))


def _finish(entry: dict, outcome: str, log_path: Path, **extra) -> dict:
    result = {**entry, "outcome": outcome, **extra}
    audit.record(result, log_path)
    return result
