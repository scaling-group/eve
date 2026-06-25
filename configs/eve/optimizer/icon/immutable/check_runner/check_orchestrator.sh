#!/bin/bash
# ADAPT: your PBS project allocation
#PBS -P REPLACE_WITH_YOUR_PROJECT_CODE
#PBS -j oe
#PBS -k oed
#PBS -N eve_icon_check_orch
#PBS -l walltime=00:30:00
#PBS -l select=1:ngpus=2

set -euo pipefail

STAGING_DIR="${STAGING_DIR:?Must set STAGING_DIR}"
SCRIPT_DIR="${PBS_O_WORKDIR:-${STAGING_DIR}}"

set +u
source ~/.bashrc
set -euo pipefail

if command -v python3 >/dev/null 2>&1; then
  ORCHESTRATOR_PYTHON="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  ORCHESTRATOR_PYTHON="$(command -v python)"
else
  echo "Failed to resolve python for orchestrator" >&2
  exit 1
fi

"${ORCHESTRATOR_PYTHON}" \
  "${SCRIPT_DIR}/orchestrator.py" \
  --task check \
  --staging-dir "${STAGING_DIR}"
