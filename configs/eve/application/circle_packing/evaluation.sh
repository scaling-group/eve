#!/usr/bin/env bash
# Run with stricter shell settings:
# - `-e` exits on an unhandled command failure
# - `-u` errors on unset variables
# - `-o pipefail` fails the pipeline if any stage fails
set -euo pipefail

# Ensure the log directory exists before writing any evaluation artifacts.
mkdir -p "$EVE_EVAL_LOG_ROOT"

# Standard output locations for evaluation artifacts:
# - score.yaml: final YAML payload consumed by Eve
# - error.txt: failure reason when evaluation cannot be parsed or run
score_path="$EVE_EVAL_LOG_ROOT/score.yaml"
error_path="$EVE_EVAL_LOG_ROOT/error.txt"
tmp_stdout="$(mktemp)"
tmp_stderr="$(mktemp)"
trap 'rm -f "$tmp_stdout" "$tmp_stderr"' EXIT

# Assume success first, then capture the real exit code if the subprocess fails.
status=0
(
  # Run from the candidate repo root provided by the outer workflow.
  cd "$EVE_OUTPUT_ROOT"
  # `evaluate.py` is expected to print a JSON object to stdout,
  # for example {"score": ..., "summary": ...}.
  python3 - <<'PY'
import subprocess
import sys

try:
    completed = subprocess.run(
        [sys.executable, "evaluate.py"],
        check=False,
        timeout=600,
    )
except subprocess.TimeoutExpired:
    print("evaluate.py timed out after 600 seconds", file=sys.stderr)
    raise SystemExit(124)

raise SystemExit(completed.returncode)
PY
) >"$tmp_stdout" 2>"$tmp_stderr" || status=$?

# Forward the evaluator's raw output through this script's own stdout/stderr so
# the outer evaluation step can capture it as step-level logs.
if [[ -s "$tmp_stdout" ]]; then
  cat "$tmp_stdout"
fi
if [[ -s "$tmp_stderr" ]]; then
  cat "$tmp_stderr" >&2
fi

# If the evaluation command itself fails, do not fail the shell step outright.
# Instead, follow the Eve contract by writing a failure score payload and an error log.
if [[ $status -ne 0 ]]; then
  cat >"$score_path" <<'EOF'
score: 0.0
summary: evaluation command failed
EOF
  {
    # Record the process exit code first.
    printf 'Evaluation command failed with exit code %s\n' "$status"
    # Append stderr if present so the root cause is preserved.
    if [[ -s "$tmp_stderr" ]]; then
      printf '\n'
      cat "$tmp_stderr"
    fi
  } >"$error_path"
  # Exit successfully here because the outer workflow reads failure state
  # from `score.yaml` and `error.txt`, not from this script's exit code.
  exit 0
fi

# Use a short Python helper to parse the evaluator stdout and
# materialize the files required by the outer workflow.
python3 - "$tmp_stdout" "$score_path" "$error_path" <<'PY'
import json
import sys
from pathlib import Path

import yaml

# Receive all file paths as positional arguments from the shell wrapper.
stdout_path = Path(sys.argv[1])
score_path = Path(sys.argv[2])
error_path = Path(sys.argv[3])

# Read the evaluator's raw stdout and trim surrounding whitespace before parsing.
raw = stdout_path.read_text(encoding="utf-8").strip()
try:
    # Stdout is expected to be a complete JSON payload.
    payload = json.loads(raw)
    # `score` is required and must be convertible to float.
    score = float(payload["score"])
except Exception as exc:  # noqa: BLE001
    # Parsing failures still follow the same contract:
    # write a minimal failure score payload and record the parse error for diagnosis.
    score_path.write_text(
        yaml.safe_dump(
            {
                "score": 0.0,
                "summary": "failed to parse evaluator output",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    error_path.write_text(f"Failed to parse evaluator output: {exc}\n", encoding="utf-8")
    # Exit cleanly so the outer workflow can handle failure via the written files.
    raise SystemExit(0)

# On success, write the same payload as YAML without reshaping it.
score_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
PY
