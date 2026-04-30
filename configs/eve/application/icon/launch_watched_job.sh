#!/usr/bin/env bash

set -euo pipefail

STAGING_DIR="${STAGING_DIR:?Must set STAGING_DIR}"
PBS_SCRIPT_NAME="${PBS_SCRIPT_NAME:?Must set PBS_SCRIPT_NAME}"
WATCHER_SCRIPT_NAME="${WATCHER_SCRIPT_NAME:?Must set WATCHER_SCRIPT_NAME}"
WATCHER_STATUS_PATH="${WATCHER_STATUS_PATH:-${STAGING_DIR}/status.json}"
WATCHER_RESULT_PATH="${WATCHER_RESULT_PATH:-${STAGING_DIR}/result.json}"
WATCHER_POLL_SECONDS="${WATCHER_POLL_SECONDS:-5}"
WATCHER_MAX_SECONDS="${WATCHER_MAX_SECONDS:-2400}"
ORCHESTRATOR_STDOUT_PATH="${ORCHESTRATOR_STDOUT_PATH:-${STAGING_DIR}/orchestrator_stdout.log}"
ORCHESTRATOR_STDERR_PATH="${ORCHESTRATOR_STDERR_PATH:-${STAGING_DIR}/orchestrator_stderr.log}"
WATCHER_LOG_PATH="${WATCHER_LOG_PATH:-${STAGING_DIR}/watcher.log}"

cd "${STAGING_DIR}"

JOB_ID="$(timeout 180 qsub -v STAGING_DIR="${STAGING_DIR}" -o "${ORCHESTRATOR_STDOUT_PATH}" -e "${ORCHESTRATOR_STDERR_PATH}" "${PBS_SCRIPT_NAME}" 2>&1)"
JOB_ID="$(printf '%s\n' "${JOB_ID}" | awk '/[0-9]+\./ {print $1; exit}')"
if [[ -z "${JOB_ID}" ]]; then
  exit 1
fi

nohup python3 "${STAGING_DIR}/${WATCHER_SCRIPT_NAME}" \
  --job-id "${JOB_ID}" \
  --status-path "${WATCHER_STATUS_PATH}" \
  --result-path "${WATCHER_RESULT_PATH}" \
  --poll-seconds "${WATCHER_POLL_SECONDS}" \
  --max-seconds "${WATCHER_MAX_SECONDS}" \
  > "${WATCHER_LOG_PATH}" 2>&1 < /dev/null &

printf '%s\n' "${JOB_ID}"
