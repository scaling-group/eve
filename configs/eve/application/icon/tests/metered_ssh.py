#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
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
    real_ssh = os.environ["HARNESS_REAL_SSH_BIN"]
    workflow_id = os.environ.get("HARNESS_WORKFLOW_ID", "default")
    args = sys.argv[1:]
    debug_dir = Path(os.environ["HARNESS_DEBUG_LOG_DIR"])
    debug_dir.mkdir(parents=True, exist_ok=True)
    debug_log = debug_dir / f"ssh-{workflow_id}-{uuid.uuid4().hex}.log"

    command = [real_ssh, "-vvv", "-E", str(debug_log), *args]
    proc = subprocess.run(
        command,
        stdin=sys.stdin.buffer,
        stdout=sys.stdout.buffer,
        stderr=sys.stderr.buffer,
        check=False,
    )

    debug_text = (
        debug_log.read_text(encoding="utf-8", errors="replace") if debug_log.exists() else ""
    )
    fresh_handshake = bool(
        re.search(r"(Connecting to |Connection established|Authenticating to )", debug_text)
    )
    is_config_only = "-G" in args
    append_event(
        {
            "ts": timestamp_utc(),
            "workflow_id": workflow_id,
            "kind": "ssh",
            "argv": args,
            "rc": proc.returncode,
            "fresh_handshake": fresh_handshake,
            "config_only": is_config_only,
            "debug_log": str(debug_log),
        }
    )
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
