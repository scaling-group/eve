#!/usr/bin/env bash
# Programmatic evaluation step for circle_packing.
#
# Writes the `performance` dimension into logs/evaluate/score.yaml. It KEEPS any dimensions
# earlier steps already wrote and ADDS its own, so it can run at ANY position in the pipeline
# (first, or after a judge) without clobbering other steps' dimensions. There is no engine-side
# merge: every step shares the one logs/evaluate/score.yaml (never cleared), and each step keeps
# the prior dimensions and adds its own (design doc §5, §6).
#
# Env (provided by the eval phase):
#   $EVE_SOLVER_ROOT    candidate repo root (contains candidate.py / evaluate.py); read-only
#   $EVE_EVAL_LOG_ROOT  where this step writes its contribution
set -euo pipefail
mkdir -p "$EVE_EVAL_LOG_ROOT"

# Run the task's own evaluator; it prints a JSON object like {"score": ..., "summary": ...}.
raw="$(cd "$EVE_SOLVER_ROOT" && python3 evaluate.py)"

python3 - "$raw" "$EVE_EVAL_LOG_ROOT/score.yaml" <<'PY'
import json
import sys
from pathlib import Path

import yaml

raw, out = sys.argv[1], sys.argv[2]
payload = json.loads(raw)
out_path = Path(out)

# Keep any dimensions earlier steps already wrote, then add ours.
score = {}
if out_path.exists():
    score = yaml.safe_load(out_path.read_text(encoding="utf-8")) or {}
score["performance"] = float(payload["score"])
score["summary"] = payload.get("summary", "")
out_path.write_text(yaml.safe_dump(score, sort_keys=False), encoding="utf-8")
PY
