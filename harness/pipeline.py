"""One command, end to end: interpret -> validate -> readback -> confirm -> apply -> audit."""

from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import httpx

from harness import audit
from harness.llm import ModelOutputError, interpret
from harness.readback import readback
from harness.resolve import resolve
from harness.schema import CommandBatch
from harness.state import AircraftState
from harness.validator import validate


def process(
    text: str,
    state: AircraftState,
    confirm: Callable[[str], bool],
    model: Callable[[str], CommandBatch] = interpret,
    log_path: Path = audit.DEFAULT_LOG,
) -> dict:
    """Run one command and return its audit record.

    `confirm` and `model` are passed in (dependency injection), so tests can swap in a fake
    model and an automatic yes/no instead of a live LLM and a human at the keyboard.
    """
    entry = {"input": text, "state_before": asdict(state)}
    try:
        batch = model(text)
    except ModelOutputError as err:
        return _finish(entry, "model_error", log_path, reasons=[str(err)])
    except httpx.HTTPError as err:
        return _finish(entry, "unavailable", log_path, reasons=[f"{type(err).__name__}: {err}"])

    entry["commands"] = batch.model_dump()["commands"]
    if reasons := validate(batch, state):
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
