"""Step 7: voice in. Speech -> text (ASR) -> the harness, measured end to end.

1. A built-in Windows voice reads each "accept" phrase from the eval set into a WAV file (synthetic audio).
2. The built-in Windows recognizer transcribes each WAV twice:
     dictation: general English, any word allowed
     domain:    only our aviation vocabulary allowed (the simplest form of domain adaptation)
3. We score each transcript with WER, then run it through the agent + pipeline and score the RESULT.

Clean computer speech is far easier than a real cockpit (rotor noise, accents, radio static):
treat these numbers as a best case. Nothing is installed; everything is Windows' own System.Speech.

Run:  uv run examples/07_speech.py
"""

import json
import subprocess
from pathlib import Path

from harness.evaluate import CORRECT, load_cases, run_case, score
from harness.wer import corpus_wer, wer

WORK_DIR = Path("logs/speech")  # git-ignored
SCRIPT = Path(__file__).with_name("speech_transcribe.ps1")

# The domain's words, written once by hand (NOT pulled from the test phrases, which would be cheating).
VOCABULARY = """zero one two three four five six seven eight nine niner ten twenty thirty forty fifty sixty seventy
eighty ninety hundred thousand point turn left right heading fly climb descend maintain and to another feet
degrees knots speed airspeed slow tune contact VHF UHF tower approach alpha bravo charlie""".split()


def transcribe(phrases: list[str]) -> list[dict]:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    job, results = WORK_DIR / "job.json", WORK_DIR / "results.json"
    job.write_text(json.dumps({"phrases": phrases, "vocabulary": VOCABULARY}), encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                    str(job), str(WORK_DIR), str(results)], check=True)
    return json.loads(results.read_text(encoding="utf-8-sig"))  # PowerShell 5 writes a BOM


def main() -> None:
    cases = [case for case in load_cases() if case["expect"] == "accept"]
    transcripts = transcribe([case["input"] for case in cases])

    totals = {"dictation": 0, "domain": 0}
    print(f"{'case':20} {'mode':9} {'WER':>5}  {'result':14} transcript")
    for case, heard in zip(cases, transcripts):
        for mode in totals:
            # The same scoring as the text-only eval, but the input is what the recognizer heard.
            verdict = score(case, run_case({**case, "input": heard[mode]}, "agent"))
            totals[mode] += verdict == CORRECT
            print(f"{case['id']:20} {mode:9} {wer(case['input'], heard[mode]):5.0%}  {verdict:14} {heard[mode]!r}")

    for mode, correct in totals.items():
        pairs = [(case["input"], heard[mode]) for case, heard in zip(cases, transcripts)]
        print(f"\n{mode:9}: corpus WER {corpus_wer(pairs):.1%}   commands correct {correct}/{len(cases)}")


if __name__ == "__main__":
    main()
