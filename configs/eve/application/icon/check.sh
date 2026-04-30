#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${EVE_OUTPUT_ROOT:-}" ]]; then
  repo_root="$(cd "${EVE_OUTPUT_ROOT}" && pwd)"
  workspace_root="${EVE_WORKSPACE_ROOT:-$(cd "${repo_root}/.." && pwd)}"
elif [[ -n "${EVE_WORKSPACE_ROOT:-}" ]]; then
  workspace_root="$(cd "${EVE_WORKSPACE_ROOT}" && pwd)"
  repo_root="$(cd "${workspace_root}/output" && pwd)"
else
  repo_root="$(cd "${script_dir}/../../../.." && pwd)"
  workspace_root="$(cd "${repo_root}/.." && pwd)"
fi
check_log_root="${EVE_CHECK_LOG_ROOT:-${workspace_root}/logs/check-runner}"

EVE_REMOTE_HOST="${EVE_REMOTE_HOST:-remote-cluster}"
REMOTE_REPO="${REMOTE_REPO:-~/repos/icon-core}"
REMOTE_BRANCH="${REMOTE_BRANCH:-main}"
MAX_STEPS="${MAX_STEPS:-50}"
VAL_EVERY="${VAL_EVERY:-50}"
SAVE_EVERY="${SAVE_EVERY:-50}"
TASK_NAME_PREFIX="${TASK_NAME_PREFIX:-eve_icon_evolve_iter}"
BS="${BS:-32}"
LR="${LR:-5e-4}"
NUM_WORKERS="${NUM_WORKERS:-32}"
DATA_DIR="${DATA_DIR:-${EVE_REMOTE_DATA_DIR:-}}"
TRAIN_RUNNING_TIMEOUT_SECONDS="${TRAIN_RUNNING_TIMEOUT_SECONDS:-1200}"
DEMO_NUMS="${DEMO_NUMS:-1 2 3 4 5 6 7 8 9 10}"
CHECK_EVAL_TIMEOUT_SECONDS="${CHECK_EVAL_TIMEOUT_SECONDS:-90}"
EDITABLE_FILES=(
  configs/experiment/evolve_base.yaml
  configs/model/icon_evolve.yaml
  src/models/icon/icon_evolve.py
  src/models/base/transformer_evolve.py
  src/models/icon/pe_evolve.py
)
EDITABLE_FOLDERS=()

SSH_BIN="${SSH_BIN:-ssh}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"
SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-}"
SSH_OPTIONS=()
SSH_RSYNC_CMD=""
RSYNC_COMMON_OPTIONS=()
RSYNC_INPLACE_OPTIONS=()

L2_RETRY_ATTEMPTS="${L2_RETRY_ATTEMPTS:-4}"
L2_BACKOFF_FLOOR_SECONDS="${L2_BACKOFF_FLOOR_SECONDS:-5}"
L2_BACKOFF_MAX_SECONDS="${L2_BACKOFF_MAX_SECONDS:-40}"
L2_JITTER_MAX_SECONDS="${L2_JITTER_MAX_SECONDS:-3}"
BREAKER_COOLDOWN_SECONDS="${BREAKER_COOLDOWN_SECONDS:-60}"
BREAKER_HALF_OPEN_LEASE_SECONDS="${BREAKER_HALF_OPEN_LEASE_SECONDS:-15}"
BREAKER_HALT_WINDOW_SECONDS="${BREAKER_HALT_WINDOW_SECONDS:-900}"
BREAKER_DIR="${EVE_REMOTE_BREAKER_DIR:-/tmp/remote_transport_breaker_${USER}}"
BREAKER_STATE_FILE="${BREAKER_DIR}/state.json"
BREAKER_LOCK_FILE="${BREAKER_DIR}/state.lock"
PROBE_CONNECT_TIMEOUT_SECONDS="${PROBE_CONNECT_TIMEOUT_SECONDS:-10}"
RESULT_JSON_FETCH_ATTEMPTS="${RESULT_JSON_FETCH_ATTEMPTS:-9}"
RESULT_JSON_FETCH_INITIAL_DELAY_SECONDS="${RESULT_JSON_FETCH_INITIAL_DELAY_SECONDS:-2}"
RESULT_JSON_FETCH_MAX_DELAY_SECONDS="${RESULT_JSON_FETCH_MAX_DELAY_SECONDS:-15}"
TRANSPORT_RECONCILE_GRACE_SECONDS="${TRANSPORT_RECONCILE_GRACE_SECONDS:-600}"
TRANSPORT_RECONCILE_POLL_INTERVAL_SECONDS="${TRANSPORT_RECONCILE_POLL_INTERVAL_SECONDS:-15}"

WAIT_PBS_CANDIDATE_FAILURE=1
WAIT_PBS_RETRYABLE_TIMEOUT=2
WAIT_PBS_TRANSPORT_HALT=3

LAST_QSTAT_OUTPUT=""
ERROR_MESSAGE=""
OUTCOME_IS_TRANSPORT_HALT=0
OUTCOME_IS_CANDIDATE_FAILURE=0
ATTEMPT_ID="${ATTEMPT_ID_OVERRIDE:-check_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}}"
LOGS_DIR="${check_log_root}"
RESULT_JSON_PATH="${LOGS_DIR}/result_${ATTEMPT_ID}.json"
ORCHESTRATOR_STDOUT_PATH="${LOGS_DIR}/orchestrator_${ATTEMPT_ID}.out.log"
ORCHESTRATOR_STDERR_PATH="${LOGS_DIR}/orchestrator_${ATTEMPT_ID}.err.log"
STATUS_JSON_PATH="${LOGS_DIR}/status_${ATTEMPT_ID}.json"
WATCHER_LOG_PATH="${LOGS_DIR}/watcher_${ATTEMPT_ID}.log"
META_JSON_PATH=""
STAGING_DIR=""
REMOTE_WORKTREE=""
REMOTE_REPO_PATH=""
WORKSPACE_BASE=""
REMOTE_USER=""
STAGE_BUNDLE_DIR=""
STAGE_ATTEMPT_ROOT=""
STATUS_POLL_INTERVAL_SECONDS="${STATUS_POLL_INTERVAL_SECONDS:-60}"

source "${script_dir}/helpers.sh"
init_remote_transport

cleanup_local_temp() {
  if [[ -n "${META_JSON_PATH:-}" ]]; then
    rm -f "${META_JSON_PATH}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${STAGE_BUNDLE_DIR:-}" ]]; then
    rm -rf "${STAGE_BUNDLE_DIR}" >/dev/null 2>&1 || true
  fi
}

cleanup_remote() {
  set +e
  if [[ -n "${STAGING_DIR:-}" ]]; then
    ssh_remote "rm -rf '${STAGING_DIR}'" >/dev/null 2>&1 || true
  fi
  close_ssh_master
  cleanup_local_temp
}

trap 'rc=$?; trap - EXIT; cleanup_remote; exit "$rc"' EXIT

print_traceback_excerpt() {
  local traceback_text="$1"
  if [[ -z "${traceback_text}" ]]; then
    return
  fi
  python3 - "${traceback_text}" <<'PY'
import sys

traceback_text = sys.argv[1]
lines = traceback_text.splitlines()
if not lines:
    raise SystemExit(0)
print("[CHECK-TRACEBACK] begin", file=sys.stderr)
for line in lines[:50]:
    print(line, file=sys.stderr)
if len(lines) > 50:
    print("[CHECK-TRACEBACK] truncated", file=sys.stderr)
print("[CHECK-TRACEBACK] end", file=sys.stderr)
PY
}

print_error_message() {
  if [[ -n "${ERROR_MESSAGE:-}" ]]; then
    printf '%s\n' "${ERROR_MESSAGE}" >&2
  fi
}

fail_code() {
  local stage="$1"
  shift

  printf '[CHECK-CODE] stage=%s %s\n' "${stage}" "$*" >&2
  exit 1
}

fail_pbs() {
  local stage="$1"
  shift

  printf '[CHECK-PBS] stage=%s %s\n' "${stage}" "$*" >&2
  print_error_message
  print_traceback_excerpt "${RESULT_ERROR_TRACEBACK:-}"
  if [[ -n "${LAST_QSTAT_OUTPUT:-}" ]]; then
    printf '%s\n' "${LAST_QSTAT_OUTPUT}" >&2
  fi
  exit 1
}

fail_infra() {
  local stage="$1"
  shift
  local fallback_message="$*"
  local message="${ERROR_MESSAGE:-${fallback_message}}"

  if [[ -z "${ERROR_MESSAGE:-}" ]]; then
    halt_transport "${fallback_message}" || true
  fi
  printf '[CHECK-INFRA] stage=%s %s\n' "${stage}" "${message}" >&2
  print_error_message
  if [[ -n "${LAST_QSTAT_OUTPUT:-}" ]]; then
    printf '%s\n' "${LAST_QSTAT_OUTPUT}" >&2
  fi
  exit 1
}

json_array() {
  python3 - "$@" <<'PY'
import json
import sys

values = sys.argv[1:]
if values == [""]:
    values = []
print(json.dumps(values))
PY
}

fetch_remote_file() {
  local label="$1"
  local remote_path="$2"
  local local_path="$3"
  local tmp_path="${local_path}.tmp"

  mkdir -p "$(dirname "${local_path}")"
  if ! ssh_remote_resilient "${label}" "
    set -euo pipefail
    test -f '${remote_path}'
    cat '${remote_path}'
  " >"${tmp_path}"; then
    rm -f "${tmp_path}" >/dev/null 2>&1 || true
    return 1
  fi
  mv "${tmp_path}" "${local_path}"
}

result_json_is_valid() {
  local local_path="$1"

  python3 - "${local_path}" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(1)
text = path.read_text(encoding="utf-8")
if not text.strip():
    raise SystemExit(1)
json.loads(text)
PY
}

fetch_result_json_with_retry() {
  local label="$1"
  local remote_path="$2"
  local local_path="$3"
  local attempt=1
  local sleep_seconds="${RESULT_JSON_FETCH_INITIAL_DELAY_SECONDS}"
  local max_attempts="${RESULT_JSON_FETCH_ATTEMPTS}"

  while (( attempt <= max_attempts )); do
    if ssh_remote "test -s '${remote_path}'" >/dev/null 2>&1; then
      if fetch_remote_file "${label}" "${remote_path}" "${local_path}" && result_json_is_valid "${local_path}"; then
        return 0
      fi
    fi

    rm -f "${local_path}" "${local_path}.tmp" >/dev/null 2>&1 || true
    if (( attempt == max_attempts )); then
      return 1
    fi

    printf '%s [REMOTE-RESULT] %s not ready; retry %d/%d in %ss\n' \
      "$(timestamp_utc)" "${label}" "${attempt}" "${max_attempts}" "${sleep_seconds}" >&2
    sleep "${sleep_seconds}"
    if (( sleep_seconds < RESULT_JSON_FETCH_MAX_DELAY_SECONDS )); then
      sleep_seconds=$((sleep_seconds * 2))
      if (( sleep_seconds > RESULT_JSON_FETCH_MAX_DELAY_SECONDS )); then
        sleep_seconds="${RESULT_JSON_FETCH_MAX_DELAY_SECONDS}"
      fi
    fi
    attempt=$((attempt + 1))
  done
}

prepare_stage_bundle() {
  STAGE_BUNDLE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/icon-check-stage-${ATTEMPT_ID}.XXXXXX")"
  STAGE_ATTEMPT_ROOT="${STAGE_BUNDLE_DIR}/eve_staging/${ATTEMPT_ID}"
  python3 - \
    "${STAGE_ATTEMPT_ROOT}" \
    "${repo_root}" \
    "${META_JSON_PATH}" \
    "${script_dir}/check_orchestrator.sh" \
    "${script_dir}/orchestrator.py" \
    "${script_dir}/status_watcher.py" \
    "${script_dir}/launch_watched_job.sh" \
    "${EDITABLE_FILES[@]}" <<'PY'
import shutil
import sys
from pathlib import Path

attempt_root = Path(sys.argv[1])
source_root = Path(sys.argv[2])
meta_path = Path(sys.argv[3])
extra_files = [Path(value) for value in sys.argv[4:8]]
editable_files = [Path(value) for value in sys.argv[8:]]

inputs_root = attempt_root / "inputs"
inputs_root.mkdir(parents=True, exist_ok=True)
shutil.copy2(meta_path, attempt_root / "meta.json")
for extra_file in extra_files:
    shutil.copy2(extra_file, attempt_root / extra_file.name)
for editable_file in editable_files:
    target = inputs_root / editable_file
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_root / editable_file, target)
PY
}

write_meta_json() {
  local editable_files_json editable_folders_json

  editable_files_json="$(json_array "${EDITABLE_FILES[@]}")"
  editable_folders_json="$(json_array "${EDITABLE_FOLDERS[@]-}")"
  META_JSON_PATH="$(mktemp "${TMPDIR:-/tmp}/icon-check-meta-${ATTEMPT_ID}.XXXXXX")"
  python3 - \
    "${META_JSON_PATH}" \
    "${ATTEMPT_ID}" \
    "$(timestamp_utc)" \
    "${workspace_root}" \
    "${REMOTE_REPO}" \
    "${REMOTE_BRANCH}" \
    "${MAX_STEPS}" \
    "${VAL_EVERY}" \
    "${SAVE_EVERY}" \
    "${TASK_NAME_PREFIX}" \
    "${BS}" \
    "${LR}" \
    "${NUM_WORKERS}" \
    "${DATA_DIR}" \
    "/scratch/${REMOTE_USER}/logs" \
    "${TRAIN_RUNNING_TIMEOUT_SECONDS}" \
    "${DEMO_NUMS}" \
    "${CHECK_EVAL_TIMEOUT_SECONDS}" \
    "${editable_files_json}" \
    "${editable_folders_json}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "attempt_id": sys.argv[2],
    "created_at": sys.argv[3],
    "task": "check",
    "workspace_hint": sys.argv[4],
    "remote_repo": sys.argv[5],
    "remote_branch": sys.argv[6],
    "max_steps": int(sys.argv[7]),
    "val_every": int(sys.argv[8]),
    "save_every": int(sys.argv[9]),
    "task_name_prefix": sys.argv[10],
    "bs": sys.argv[11],
    "lr": sys.argv[12],
    "num_workers": sys.argv[13],
    "data_dir": sys.argv[14],
    "log_dir": sys.argv[15],
    "train_timeout_seconds": int(sys.argv[16]),
    "demo_nums": sys.argv[17],
    "check_eval_timeout_seconds": int(sys.argv[18]),
    "editable_files": json.loads(sys.argv[19]),
    "editable_folders": json.loads(sys.argv[20]),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
PY
}

parse_result_env() {
  python3 - "${RESULT_JSON_PATH}" <<'PY'
import json
import shlex
import sys

payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
error = payload.get("error") or {}
fields = {
    "RESULT_STATUS": payload.get("status") or "",
    "RESULT_ERROR_STAGE": error.get("stage") or "",
    "RESULT_ERROR_MESSAGE": error.get("message") or "",
    "RESULT_ERROR_TRACEBACK": error.get("traceback") or "",
    "RESULT_CHECKPOINT_PATH": payload.get("checkpoint_path") or "",
    "RESULT_REMOTE_WORKTREE": payload.get("remote_worktree") or "",
    "RESULT_PBS_JOB_ID": payload.get("pbs_job_id") or "",
}
for key, value in fields.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
}

mkdir -p "${LOGS_DIR}"

if [[ -z "${EVE_BOUNDARY_CHECK_COMMAND:-}" ]]; then
  fail_infra "boundary_env_missing" "EVE_BOUNDARY_CHECK_COMMAND was empty"
fi
if ! bash -lc "${EVE_BOUNDARY_CHECK_COMMAND}"; then
  fail_code "boundary" "Boundary check failed"
fi

for relative_path in "${EDITABLE_FOLDERS[@]:-}"; do
  [[ -n "${relative_path}" ]] || continue
  if [[ ! -d "${repo_root}/${relative_path}" ]]; then
    fail_code "editable_surface" "Missing editable folder ${relative_path}"
  fi
done
for relative_path in "${EDITABLE_FILES[@]}"; do
  if [[ ! -f "${repo_root}/${relative_path}" ]]; then
    fail_code "editable_surface" "Missing editable file ${relative_path}"
  fi
done

if ! REMOTE_USER="$(resolve_remote_user)"; then
  fail_infra "resolve_remote_user" "Failed to resolve remote user on ${EVE_REMOTE_HOST}"
fi
REMOTE_USER="$(printf '%s' "${REMOTE_USER}" | tr -d '\r\n')"
DATA_DIR="${DATA_DIR:-/scratch/${REMOTE_USER}/data}"
WORKSPACE_BASE="/scratch/${REMOTE_USER}/eve_staging"
STAGING_DIR="${WORKSPACE_BASE}/${ATTEMPT_ID}"
REMOTE_REPO_PATH="${REMOTE_REPO/#\~/\$HOME}"
REMOTE_WORKTREE="${STAGING_DIR}"

write_meta_json
prepare_stage_bundle

if ! rsync_resilient \
  "stage check bundle" \
  -a "${RSYNC_COMMON_OPTIONS[@]}" -e "${SSH_RSYNC_CMD}" \
  "${STAGE_BUNDLE_DIR}/" \
  "${EVE_REMOTE_HOST}:/scratch/${REMOTE_USER}/"; then
  fail_infra "stage_bundle" "Failed to stage check bundle into /scratch/${REMOTE_USER}"
fi

JOB_ID=""
if ! JOB_ID="$(submit_watched_pbs_job "launch_watched_job.sh" "check_orchestrator.sh" "check orchestrator")"; then
  fail_infra "orchestrator_submit" "Failed to submit check orchestrator PBS job"
fi

pbs_wait_rc=0
if wait_for_status_file "${JOB_ID}" "${STAGING_DIR}/status.json" "${STATUS_POLL_INTERVAL_SECONDS}" "check orchestrator"; then
  pbs_wait_rc=0
else
  pbs_wait_rc=$?
fi

if [[ "${pbs_wait_rc}" -eq "${WAIT_PBS_TRANSPORT_HALT}" ]]; then
  if reconcile_pbs_after_transport_halt \
    "${JOB_ID}" \
    "${TRANSPORT_RECONCILE_POLL_INTERVAL_SECONDS}" \
    "${TRANSPORT_RECONCILE_GRACE_SECONDS}" \
    "check orchestrator"; then
    pbs_wait_rc=0
  else
    pbs_wait_rc=$?
  fi
fi

FETCH_DIR="$(mktemp -d "${TMPDIR:-/tmp}/icon-check-fetch-${ATTEMPT_ID}.XXXXXX")"
if fetch_remote_artifacts_batch \
  "fetch check artifacts batch" \
  "${STAGING_DIR}" \
  "${FETCH_DIR}" \
  "result.json" \
  "status.json" \
  "orchestrator_stdout.log" \
  "orchestrator_stderr.log" \
  "watcher.log"; then
  mv "${FETCH_DIR}/result.json" "${RESULT_JSON_PATH}" 2>/dev/null || true
  mv "${FETCH_DIR}/status.json" "${STATUS_JSON_PATH}" 2>/dev/null || true
  mv "${FETCH_DIR}/orchestrator_stdout.log" "${ORCHESTRATOR_STDOUT_PATH}" 2>/dev/null || true
  mv "${FETCH_DIR}/orchestrator_stderr.log" "${ORCHESTRATOR_STDERR_PATH}" 2>/dev/null || true
  mv "${FETCH_DIR}/watcher.log" "${WATCHER_LOG_PATH}" 2>/dev/null || true
fi
rm -rf "${FETCH_DIR}" >/dev/null 2>&1 || true

if [[ ! -s "${RESULT_JSON_PATH}" ]]; then
  emit_pbs_fetch_diagnostics "${JOB_ID}" "check_orchestrator" || true
  case "${pbs_wait_rc}" in
    "${WAIT_PBS_CANDIDATE_FAILURE}")
      fail_infra "result_fetch" "Orchestrator PBS job ${JOB_ID} exited non-zero before result.json could be fetched"
      ;;
    "${WAIT_PBS_RETRYABLE_TIMEOUT}")
      fail_infra "result_fetch" "Timed out while polling orchestrator PBS job ${JOB_ID} before result.json was available"
      ;;
    "${WAIT_PBS_TRANSPORT_HALT}")
      fail_infra "result_fetch" "Transport halted while polling or fetching result.json for orchestrator PBS job ${JOB_ID}"
      ;;
    *)
      fail_infra "result_fetch" "Failed to fetch ${STAGING_DIR}/result.json"
      ;;
  esac
fi

result_env=""
if ! result_env="$(parse_result_env)"; then
  fail_infra "result_parse" "Failed to parse ${RESULT_JSON_PATH}"
fi
eval "${result_env}"
ERROR_MESSAGE="${RESULT_ERROR_MESSAGE}"
if [[ -n "${RESULT_JSON_PATH}" ]]; then
  ERROR_MESSAGE="${ERROR_MESSAGE}; result_json=${RESULT_JSON_PATH}"
fi
if [[ -f "${ORCHESTRATOR_STDOUT_PATH}" ]]; then
  ERROR_MESSAGE="${ERROR_MESSAGE}; orchestrator_stdout=${ORCHESTRATOR_STDOUT_PATH}"
fi
if [[ -f "${ORCHESTRATOR_STDERR_PATH}" ]]; then
  ERROR_MESSAGE="${ERROR_MESSAGE}; orchestrator_stderr=${ORCHESTRATOR_STDERR_PATH}"
fi

case "${RESULT_STATUS}" in
  pass)
    printf '[CHECK-PASS] checkpoint=%s remote_worktree=%s job_id=%s result_json=%s\n' \
      "${RESULT_CHECKPOINT_PATH}" \
      "${RESULT_REMOTE_WORKTREE}" \
      "${RESULT_PBS_JOB_ID:-${JOB_ID}}" \
      "${RESULT_JSON_PATH}"
    ;;
  candidate_failure)
    case "${RESULT_ERROR_STAGE}" in
      import_check|overlay)
        fail_code "${RESULT_ERROR_STAGE}" "${RESULT_ERROR_MESSAGE}"
        ;;
      training|checkpoint_verify|eval)
        fail_pbs "${RESULT_ERROR_STAGE}" "${RESULT_ERROR_MESSAGE}"
        ;;
      *)
        fail_code "${RESULT_ERROR_STAGE:-candidate_failure}" "${RESULT_ERROR_MESSAGE:-Candidate failure without error.message}"
        ;;
    esac
    ;;
  infra_failure|unclear|"")
    fail_infra "${RESULT_ERROR_STAGE:-orchestrator_result}" "${RESULT_ERROR_MESSAGE:-Orchestrator did not report a classified result}"
    ;;
  *)
    fail_infra "orchestrator_result" "Unknown result status ${RESULT_STATUS}"
    ;;
esac
