# Aviation AI Command Harness

A learning project: a small harness that turns spoken or typed flight commands for a **simulated** helicopter
into validated actions, using a local language model.

> **Synthetic only.** No real aircraft, avionics or flight simulator is connected. All limits are fictional
> training values, and all frequencies and facilities are made up.

## The idea

**The model proposes, deterministic code decides, a human confirms, and everything is logged and measured.**

The language model is the least trustworthy part of the system, so nothing safety-relevant depends on it
behaving. It turns words into a proposed command. Plain Python checks that proposal against the words that
were actually said, the limits and the current state. The human then confirms a readback that code generated
from the exact objects that will execute.

```
voice (optional) ─► confidence gate ─► model / agent + tools ─► schema parse ─► wording check
   ─► validator ─► resolve relative → absolute ─► readback ─► human y/N ─► apply ─► audit log
```

| Layer | What it guarantees | What it can't |
|---|---|---|
| Prompt | Good output is *likely* | Anything |
| Schema (Pydantic) | Correct shape and types | Correct values |
| Wording check | Every number was actually said; relative vs absolute matches the words | A misheard but valid number |
| Validator | Within limits, consistent with state, all-or-nothing | A legal command that isn't what was meant |
| Readback + human | The human approves exactly what will execute | Only works if the human listens |

The gap no automatic layer closes is the **valid-but-wrong** command: a legal command that isn't what was
meant. The readback is the defense, and evaluation measures how often it happens.

## Components

| File | Job |
|---|---|
| `harness/llm.py` | One request to the model with a JSON schema (structured output); retries with backoff; context-window guards |
| `harness/agent.py`, `harness/tools.py` | Tool-calling loop: the model may read state or look up a frequency before submitting. Submitting is only a proposal |
| `harness/schema.py` | The contract: Pydantic classes that generate the schema and parse the reply |
| `harness/validator.py` | Deterministic rules: wording check, ranges, turn and climb/descend consistency, duplicates, command limit |
| `harness/numbers.py` | Reads spoken numbers ("two thousand five hundred", "zero niner zero", "060") as values |
| `harness/resolve.py` | Relative commands ("turn 30 degrees right") → absolute ones, computed by code |
| `harness/readback.py` | ATC-style readback generated from the commands that will execute |
| `harness/pipeline.py` | The whole flow, with the model and the human confirmation injected (so tests can fake both) |
| `harness/audit.py` | One JSON line per decision |
| `harness/speech.py`, `harness/speech.ps1` | Windows' built-in speech recognizer: recorded files or the microphone, with confidence scores |
| `harness/wer.py` | Word error rate, raw or with numbers normalized |
| `harness/evaluate.py`, `evals/commands.jsonl` | 33 labelled cases scored by severity: unsafe accept, wrong command, false reject, correct |
| `harness/cli.py` | Interactive loop, typed or `--voice` |
| `examples/` | Small step-by-step scripts: first request, HTTP inspection, structured output, tool calling, inference parameters, speech |

## Running it

Requirements: [uv](https://docs.astral.sh/uv/), [Ollama](https://ollama.com/) with `qwen3.5:9b` pulled, and
Windows for the speech parts (they use the built-in `System.Speech`; nothing extra is installed).

```
uv run pytest                          # unit tests (no model needed)
uv run python -m harness.cli           # type commands
uv run python -m harness.cli --voice   # speak commands (default microphone)
uv run python -m harness.evaluate      # labelled eval, both model paths (needs Ollama)
uv run examples/07_speech.py           # speech eval: synthetic voice -> recognizer -> harness
```

## Measured results

Local model `qwen3.5:9b` through Ollama, temperature 0, on one consumer GPU.

**Text eval (33 cases):**

| Path | Accuracy | Unsafe accepts | Wrong commands | False rejects | Avg time |
|---|---|---|---|---|---|
| Structured output (one call) | 91% | 0 | 0 | 3 | 2.7 s |
| Agent with tools | 100% | 0 | 0 | 0 | 3.2 s |

The structured path can't look up facility frequencies (it has no tools), and it asks rather than invents.

**Speech eval (20 phrases, built-in Windows voice → built-in recognizer → agent → pipeline):**

| Recognizer mode | WER raw | WER, numbers normalized | Commands correct | Wrong commands |
|---|---|---|---|---|
| General dictation | 59.2% | 37.5% | 6/20 | 0 |
| Domain vocabulary only | 3.2% | 8.0% | 16/20 | 0 |

**What the numbers mean, and their limits:**
- 33 and 20 cases are small samples: one case moves accuracy by 3–5 points. They're regression checks, not
  benchmarks.
- The speech audio is a clean synthetic voice. Real audio (rotor noise, accents, radio static) would score worse.
- Everything that isn't correct is a "say again" or a rejection, not a wrong action. Before the wording check and
  the confidence gate, the speech run produced valid-but-wrong commands (1 knot from "speed 1 to 0", heading 030
  from "turn left adding 060").
- WER and task success diverge: a 75% WER transcript produced a correct command, and a 6% WER transcript
  ("slow two eighty" for "slow to eighty") produced a wrong number. Normalizing numbers *raised* the
  domain-vocabulary WER, because a single wrong digit word makes the whole number wrong.
- Recognizer confidence is a weak signal on clean audio. Misheard phrases scored up to 0.95, and a perfect one scored 0.28.

## Known limitations

- The context window on the test machine is 4,096 tokens. Requests that might not fit are refused, not truncated.
- No async: the harness serves one user at a time.
- The speech recognizer is Windows' classic engine, not a modern neural model.
- One model provider is wired up. The model is injected, so a hosted endpoint is a configuration change, but
  that hasn't been built or tested.
