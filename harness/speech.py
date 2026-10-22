"""Voice in: Windows' built-in recognizer (System.Speech), driven through harness/speech.ps1. Nothing installed.

Two ways in:
  transcribe_files(phrases)  a built-in voice reads each phrase into a WAV, then it is transcribed (the speech eval)
  listen()                   one command from the default microphone (cli.py --voice)

Every transcript carries the recognizer's confidence (0-1). Below MIN_CONFIDENCE the pipeline answers
"say again" and the words never reach the model: a guess from bad audio is how valid-but-wrong commands start.
"""

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

SCRIPT = Path(__file__).with_name("speech.ps1")
WORK_DIR = Path("logs/speech")  # git-ignored
# Chosen from the speech eval (gate off, 20 phrases): domain-vocabulary transcripts scored 0.66-0.95, so 0.6
# blocks none of them on clean audio; its job there is noise and mumbling. On dictation it stopped the one
# remaining wrong command (0.54) at the cost of 3 correct ones. Confidence is a weak signal: misheard
# phrases scored up to 0.95 and a perfect one 0.28. The wording checks and the human readback still matter.
MIN_CONFIDENCE = 0.6

# The domain's words, written once by hand (NOT pulled from the test phrases, which would be cheating).
VOCABULARY = """zero one two three four five six seven eight nine niner ten twenty thirty forty fifty sixty seventy
eighty ninety hundred thousand point turn left right heading fly climb descend maintain and to another feet
degrees knots speed airspeed slow tune contact VHF UHF tower approach alpha bravo charlie""".split()


@dataclass(frozen=True)
class Heard:
    text: str
    confidence: float  # the lowest confidence of the transcript's pieces; 0 when nothing was heard


def transcribe_files(phrases: list[str]) -> list[dict]:
    """Per phrase: {"phrase", "dictation": Heard, "domain": Heard}."""
    results = _run("file", {"phrases": phrases, "vocabulary": VOCABULARY})
    return [{"phrase": r["phrase"], "dictation": Heard(**r["dictation"]), "domain": Heard(**r["domain"])}
            for r in results]


def listen(timeout_s: int = 8) -> Heard:
    """Wait up to timeout_s for speech on the default microphone; return one command."""
    return Heard(**_run("mic", {"vocabulary": VOCABULARY, "timeout_s": timeout_s}))


def _run(mode: str, job: dict):
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    job_path, result_path = WORK_DIR / f"{mode}_job.json", WORK_DIR / f"{mode}_result.json"
    job_path.write_text(json.dumps(job), encoding="utf-8")
    subprocess.run(["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(SCRIPT),
                    "-Mode", mode, "-JobPath", str(job_path), "-OutDir", str(WORK_DIR), "-ResultPath", str(result_path)],
                   check=True)
    return json.loads(result_path.read_text(encoding="utf-8-sig"))  # PowerShell 5 writes a BOM
