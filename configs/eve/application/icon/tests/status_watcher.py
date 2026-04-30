#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def timestamp_utc() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def parse_qstat_output(output: str) -> tuple[str, str | None]:
    job_state = ""
    exit_status: str | None = None
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("job_state = "):
            job_state = stripped.split(" = ", 1)[1]
        elif stripped.startswith("Exit_status = "):
            exit_status = stripped.split(" = ", 1)[1]
    return job_state, exit_status


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--status-path", required=True)
    parser.add_argument("--result-path", required=True)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--max-seconds", type=int, default=1800)
    args = parser.parse_args()

    status_path = Path(args.status_path)
    result_path = Path(args.result_path)
    deadline = time.time() + args.max_seconds
    heartbeat = 0
    observed_state = "Q"

    atomic_write_json(
        status_path,
        {
            "updated_at": timestamp_utc(),
            "pbs_job_id": args.job_id,
            "job_state": "Q",
            "exit_status": None,
            "heartbeat": heartbeat,
            "watcher_status": "queued",
            "result_present": result_path.is_file(),
            "result_size": result_path.stat().st_size if result_path.exists() else 0,
        },
    )
    print(f"{timestamp_utc()} state=queued heartbeat={heartbeat}", flush=True)

    while time.time() < deadline:
        proc = subprocess.run(
            ["qstat", "-fx", args.job_id],
            capture_output=True,
            text=True,
            check=False,
        )
        heartbeat += 1
        if proc.returncode == 0:
            observed_state, exit_status = parse_qstat_output(proc.stdout)
            watcher_status = {
                "Q": "queued",
                "R": "running",
                "C": "finished",
                "F": "finished",
            }.get(observed_state or "Q", "unknown")
        else:
            exit_status = None
            watcher_status = "poll_error"

        result_present = result_path.is_file()
        result_size = result_path.stat().st_size if result_present else 0

        atomic_write_json(
            status_path,
            {
                "updated_at": timestamp_utc(),
                "pbs_job_id": args.job_id,
                "job_state": observed_state or "Q",
                "exit_status": exit_status,
                "heartbeat": heartbeat,
                "watcher_status": watcher_status,
                "result_present": result_present,
                "result_size": result_size,
            },
        )
        print(
            f"{timestamp_utc()} state={watcher_status} job_state={observed_state or 'Q'} "
            f"heartbeat={heartbeat} result_present={int(result_present)}",
            flush=True,
        )

        if observed_state in {"C", "F"} and result_present:
            return 0
        time.sleep(args.poll_seconds)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
