#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path


def timestamp_utc() -> str:
    return datetime.now(UTC).isoformat()


def append_event(payload: dict[str, object]) -> None:
    metrics_path = Path(os.environ["HARNESS_METRICS_FILE"])
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with metrics_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=False) + "\n")


def main() -> int:
    real_rsync = os.environ["HARNESS_REAL_RSYNC_BIN"]
    workflow_id = os.environ.get("HARNESS_WORKFLOW_ID", "default")
    args = sys.argv[1:]
    proc = subprocess.run([real_rsync, *args], capture_output=True)
    sys.stdout.buffer.write(proc.stdout)
    sys.stderr.buffer.write(proc.stderr)
    append_event(
        {
            "ts": timestamp_utc(),
            "workflow_id": workflow_id,
            "kind": "rsync",
            "argv": args,
            "rc": proc.returncode,
        }
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
