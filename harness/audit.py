"""Append-only audit log: one JSON object per line ("JSON Lines").

Every decision is recorded, accepted or not, so failures can be counted, diagnosed and replayed later.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_LOG = Path("logs/audit.jsonl")  # logs/ is git-ignored


def record(entry: dict, log_path: Path = DEFAULT_LOG) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    line = {"time": datetime.now(timezone.utc).isoformat(), **entry}
    # `with` closes the file even if an exception occurs: C#'s `using` block.
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(line) + "\n")
