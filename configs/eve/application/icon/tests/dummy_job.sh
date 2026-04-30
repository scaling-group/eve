#!/bin/bash
# ADAPT: your PBS project allocation
#PBS -P REPLACE_WITH_YOUR_PROJECT_CODE
#PBS -j oe
#PBS -k oed
#PBS -N eve_transport_harness
#PBS -q interactive_cpu
#PBS -l walltime=00:20:00
#PBS -l select=1:ncpus=1

set -euo pipefail

STAGING_DIR="${STAGING_DIR:?Must set STAGING_DIR}"
HARNESS_ATTEMPT_ID="${HARNESS_ATTEMPT_ID:?Must set HARNESS_ATTEMPT_ID}"
HARNESS_SLEEP_SECONDS="${HARNESS_SLEEP_SECONDS:-180}"
RESULT_PATH="${STAGING_DIR}/result.json"

printf 'harness job start host=%s job=%s attempt=%s\n' "$(hostname)" "${PBS_JOBID:-unknown}" "${HARNESS_ATTEMPT_ID}"
sleep "${HARNESS_SLEEP_SECONDS}"

python3 - "${RESULT_PATH}" "${HARNESS_ATTEMPT_ID}" "${PBS_JOBID:-}" <<'PY'
import json
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

result_path = Path(sys.argv[1])
attempt_id = sys.argv[2]
pbs_job_id = sys.argv[3]
payload = {
    "schema_version": 1,
    "task": "harness",
    "status": "pass",
    "attempt_id": attempt_id,
    "started_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "finished_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    "pbs_job_id": pbs_job_id,
    "remote_worktree": str(result_path.parent),
    "checkpoint_path": None,
    "score_metric": None,
    "score_components": None,
    "error": None,
    "host": socket.gethostname(),
}
tmp_path = result_path.with_suffix(".tmp")
tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
tmp_path.replace(result_path)
PY

printf 'harness job done result=%s\n' "${RESULT_PATH}"
