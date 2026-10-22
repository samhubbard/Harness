"""Step 7: voice in. Speech -> text (ASR) -> the harness, measured end to end.

1. A built-in Windows voice reads each "accept" phrase from the eval set into a WAV file (synthetic audio).
2. The built-in Windows recognizer transcribes each WAV twice:
     dictation: general English, any word allowed
     domain:    only our aviation vocabulary allowed (the simplest form of domain adaptation)
3. We score each transcript with WER (raw, and with numbers normalized), then run it through the
   agent + pipeline (with the confidence gate) and score the RESULT.

Clean computer speech is far easier than a real cockpit (rotor noise, accents, radio static):
treat these numbers as a best case. Nothing is installed; everything is Windows' own System.Speech.

Run:  uv run examples/07_speech.py
"""

from harness.evaluate import CORRECT, load_cases, run_case, score
from harness.speech import transcribe_files
from harness.wer import corpus_wer, wer


def main() -> None:
    cases = [case for case in load_cases() if case["expect"] == "accept"]
    transcripts = transcribe_files([case["input"] for case in cases])

    totals = {"dictation": 0, "domain": 0}
    print(f"{'case':20} {'mode':9} {'WER':>5} {'conf':>5}  {'result':14} transcript")
    for case, heard in zip(cases, transcripts):
        for mode in totals:
            # The same scoring as the text-only eval, but the input is what the recognizer heard.
            spoken = heard[mode]
            result = run_case({**case, "input": spoken.text}, "agent", confidence=spoken.confidence)
            verdict = score(case, result)
            totals[mode] += verdict == CORRECT
            print(f"{case['id']:20} {mode:9} {wer(case['input'], spoken.text):5.0%} {spoken.confidence:5.2f}  "
                  f"{verdict:14} {spoken.text!r}")

    for mode, correct in totals.items():
        pairs = [(case["input"], heard[mode].text) for case, heard in zip(cases, transcripts)]
        print(f"\n{mode:9}: corpus WER {corpus_wer(pairs):.1%} raw, {corpus_wer(pairs, numbers=True):.1%} numbers "
              f"normalized   commands correct {correct}/{len(cases)}")


if __name__ == "__main__":
    main()
