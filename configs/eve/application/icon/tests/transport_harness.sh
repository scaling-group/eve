#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
icon_dir="$(cd "${script_dir}/.." && pwd)"
repo_root="$(cd "${script_dir}/../../../../../" && pwd)"
EVE_REMOTE_HOST="${EVE_REMOTE_HOST:-remote-cluster}"

source "${icon_dir}/helpers.sh"

MODE=""
TRANSPORT_MODE="current"
CONCURRENCY=5
JOB_SLEEP_SECONDS=180
STATUS_POLL_SECONDS=60
WATCHER_POLL_SECONDS=5
INTERRUPT_AFTER_SECONDS=90
STATUS_INITIAL_DELAY_SECONDS=0
RUN_ROOT=""
INTERNAL_WORKFLOW_ID=""
PARENT_SOCKET_PATH=""

while (($# > 0)); do
  case "$1" in
    --mode)
      MODE="$2"
      shift 2
      ;;
    --transport-mode)
      TRANSPORT_MODE="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --job-sleep-seconds)
      JOB_SLEEP_SECONDS="$2"
      shift 2
      ;;
    --status-poll-seconds)
      STATUS_POLL_SECONDS="$2"
      shift 2
      ;;
    --interrupt-after-seconds)
      INTERRUPT_AFTER_SECONDS="$2"
      shift 2
      ;;
    --run-root)
      RUN_ROOT="$2"
      shift 2
      ;;
    --workflow-id)
      INTERNAL_WORKFLOW_ID="$2"
      shift 2
      ;;
    --shared-socket-path)
      PARENT_SOCKET_PATH="$2"
      shift 2
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${MODE}" ]]; then
  printf 'transport_harness.sh requires --mode\n' >&2
  exit 2
fi

timestamp_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

make_run_root() {
  if [[ -n "${RUN_ROOT}" ]]; then
    printf '%s\n' "${RUN_ROOT}"
    return 0
  fi
  local stamp
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  printf '%s\n' "${repo_root}/.runs/eve/transport-harness/run-${stamp}-${MODE}-${RANDOM}"
}

HARNESS_RUN_ROOT="$(make_run_root)"
mkdir -p "${HARNESS_RUN_ROOT}"

RESULT_PATH=""
ORCHESTRATOR_STDOUT_PATH=""
ORCHESTRATOR_STDERR_PATH=""
STATUS_JSON_PATH=""
WATCHER_LOG_PATH=""
SUBMIT_RAW_PATH=""
LOCAL_PAYLOAD_DIR=""
LOCAL_CONFIG_PATH=""
LOCAL_INPUTS_DIR=""
METRICS_FILE="${HARNESS_RUN_ROOT}/metrics.jsonl"
DEBUG_LOG_DIR="${HARNESS_RUN_ROOT}/ssh-debug"
SUMMARY_JSON_PATH="${HARNESS_RUN_ROOT}/summary.json"
SSH_BIN="${SSH_BIN:-ssh}"
RSYNC_BIN="${RSYNC_BIN:-rsync}"
OUTCOME_IS_TRANSPORT_HALT=0
OUTCOME_IS_CANDIDATE_FAILURE=0
ERROR_MESSAGE=""
LAST_QSTAT_OUTPUT=""
REMOTE_USER=""
STAGING_DIR=""
REMOTE_WORKTREE=""
ATTEMPT_ID=""
HARNESS_SOCKET_SHARED=0
HARNESS_LEAVE_MASTER_OPEN=0
HARNESS_INTERRUPT_SOCKET=0

WAIT_PBS_CANDIDATE_FAILURE=1
WAIT_PBS_RETRYABLE_TIMEOUT=2
WAIT_PBS_TRANSPORT_HALT=3
L2_RETRY_ATTEMPTS="${L2_RETRY_ATTEMPTS:-2}"
L2_BACKOFF_FLOOR_SECONDS="${L2_BACKOFF_FLOOR_SECONDS:-3}"
L2_BACKOFF_MAX_SECONDS="${L2_BACKOFF_MAX_SECONDS:-12}"
L2_JITTER_MAX_SECONDS="${L2_JITTER_MAX_SECONDS:-1}"
BREAKER_COOLDOWN_SECONDS="${BREAKER_COOLDOWN_SECONDS:-15}"
BREAKER_HALF_OPEN_LEASE_SECONDS="${BREAKER_HALF_OPEN_LEASE_SECONDS:-10}"
BREAKER_HALT_WINDOW_SECONDS="${BREAKER_HALT_WINDOW_SECONDS:-180}"
BREAKER_DIR="${EVE_REMOTE_BREAKER_DIR:-/tmp/remote_transport_breaker_${USER}}"
BREAKER_STATE_FILE="${BREAKER_DIR}/state.json"
BREAKER_LOCK_FILE="${BREAKER_DIR}/state.lock"
PROBE_CONNECT_TIMEOUT_SECONDS="${PROBE_CONNECT_TIMEOUT_SECONDS:-5}"

prepare_metering_wrappers() {
  export HARNESS_METRICS_FILE="${METRICS_FILE}"
  export HARNESS_DEBUG_LOG_DIR="${DEBUG_LOG_DIR}"
  export HARNESS_REAL_SSH_BIN="${SSH_BIN}"
  export HARNESS_REAL_RSYNC_BIN="${RSYNC_BIN}"
  export HARNESS_WORKFLOW_ID="${INTERNAL_WORKFLOW_ID:-${MODE}}"
  SSH_BIN="${script_dir}/metered_ssh.py"
  RSYNC_BIN="${script_dir}/metered_rsync.py"
}

configure_current_transport() {
  ATTEMPT_ID="${ATTEMPT_ID:-current_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}}"
  init_remote_transport
}

configure_v2_transport() {
  local socket_path="$1"
  local control_persist="${2:-30m}"
  SSH_SHARED_CONTROL_MASTER=1
  SSH_CONTROL_PATH="${socket_path}"
  SSH_CONTROL_REF_FILE="${SSH_CONTROL_PATH}.refs.json"
  SSH_CONTROL_LOCK_FILE="${SSH_CONTROL_PATH}.lock"
  SSH_OPTIONS=(
    -S "${SSH_CONTROL_PATH}"
    -o ControlMaster=auto
    -o ControlPersist="${control_persist}"
    -o PreferredAuthentications=publickey
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o ServerAliveInterval=15
    -o ServerAliveCountMax=4
  )
  SSH_RSYNC_CMD="${SSH_BIN} -S ${SSH_CONTROL_PATH} -o ControlMaster=auto -o ControlPersist=${control_persist} -o PreferredAuthentications=publickey -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
  RSYNC_COMMON_OPTIONS=(
    --partial
    --partial-dir=.rsync-partial
    --timeout=30
  )
  RSYNC_INPLACE_OPTIONS=()
  register_ssh_master_ref
}

prepare_local_payload() {
  local workflow_root="${HARNESS_RUN_ROOT}/${HARNESS_WORKFLOW_ID}"
  LOCAL_PAYLOAD_DIR="${workflow_root}/payload"
  LOCAL_CONFIG_PATH="${workflow_root}/payload/harness_config.json"
  LOCAL_INPUTS_DIR="${workflow_root}/payload/inputs"
  RESULT_PATH="${workflow_root}/result.json"
  ORCHESTRATOR_STDOUT_PATH="${workflow_root}/orchestrator_stdout.log"
  ORCHESTRATOR_STDERR_PATH="${workflow_root}/orchestrator_stderr.log"
  STATUS_JSON_PATH="${workflow_root}/status.json"
  WATCHER_LOG_PATH="${workflow_root}/watcher.log"
  SUBMIT_RAW_PATH="${workflow_root}/submit_raw.log"

  mkdir -p "${LOCAL_INPUTS_DIR}"
  cp "${script_dir}/dummy_job.sh" "${LOCAL_PAYLOAD_DIR}/dummy_job.sh"
  cp "${script_dir}/status_watcher.py" "${LOCAL_PAYLOAD_DIR}/status_watcher.py"
  chmod 755 "${LOCAL_PAYLOAD_DIR}/dummy_job.sh" "${LOCAL_PAYLOAD_DIR}/status_watcher.py"
  printf 'dummy input for %s\n' "${HARNESS_WORKFLOW_ID}" > "${LOCAL_INPUTS_DIR}/dummy_input.txt"
  python3 - "${LOCAL_CONFIG_PATH}" "${HARNESS_WORKFLOW_ID}" "${JOB_SLEEP_SECONDS}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "workflow_id": sys.argv[2],
    "job_sleep_seconds": int(sys.argv[3]),
}
Path(sys.argv[1]).write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
PY
}

cleanup_harness() {
  set +e
  if [[ -n "${STAGING_DIR:-}" ]]; then
    ssh_remote "rm -rf '${STAGING_DIR}'" >/dev/null 2>&1 || true
  fi
  if [[ "${HARNESS_LEAVE_MASTER_OPEN}" != "1" ]]; then
    close_ssh_master
  fi
}

trap_cleanup() {
  local rc="$1"
  trap - EXIT
  cleanup_harness
  exit "${rc}"
}

trap 'trap_cleanup $?' EXIT

fetch_remote_file_once() {
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

fetch_result_json_with_retry_current() {
  local remote_path="$1"
  local local_path="$2"
  local attempt=1
  local max_attempts=9
  local sleep_seconds=2

  while (( attempt <= max_attempts )); do
    if ssh_remote "test -s '${remote_path}'" >/dev/null 2>&1; then
      if fetch_remote_file_once "fetch result.json" "${remote_path}" "${local_path}" && result_json_is_valid "${local_path}"; then
        return 0
      fi
    fi
    rm -f "${local_path}" "${local_path}.tmp" >/dev/null 2>&1 || true
    if (( attempt == max_attempts )); then
      return 1
    fi
    sleep "${sleep_seconds}"
    if (( sleep_seconds < 15 )); then
      sleep_seconds=$((sleep_seconds * 2))
      if (( sleep_seconds > 15 )); then
        sleep_seconds=15
      fi
    fi
    attempt=$((attempt + 1))
  done
}

fetch_batch_artifacts_v2() {
  local remote_dir="$1"
  local local_dir="$2"

  mkdir -p "${local_dir}"
  rsync_resilient \
    "fetch harness artifacts batch" \
    -a "${RSYNC_COMMON_OPTIONS[@]}" -e "${SSH_RSYNC_CMD}" \
    --include="result.json" \
    --include="status.json" \
    --include="orchestrator_stdout.log" \
    --include="orchestrator_stderr.log" \
    --include="watcher.log" \
    --exclude="*" \
    "${EVE_REMOTE_HOST}:${remote_dir}/" \
    "${local_dir}/"
}

start_current_result_fetch_diagnostics() {
  local job_id="$1"
  emit_pbs_fetch_diagnostics "${job_id}" "harness_current" || true
}

resolve_harness_remote_user() {
  if ! REMOTE_USER="$(resolve_remote_user)"; then
    printf 'Failed to resolve remote user on %s\n' "${EVE_REMOTE_HOST}" >&2
    exit 1
  fi
  REMOTE_USER="$(printf '%s' "${REMOTE_USER}" | tr -d '\r\n')"
}

create_remote_staging() {
  STAGING_DIR="/scratch/${REMOTE_USER}/eve_transport_harness/${ATTEMPT_ID}"
  REMOTE_WORKTREE="${STAGING_DIR}"
  ssh_remote_resilient "create harness staging dir" "
    set -euo pipefail
    umask 077
    mkdir -p '${STAGING_DIR}/inputs'
    chmod 700 '${STAGING_DIR}'
  "
}

stage_current_payload() {
  rsync_resilient \
    "stage harness payload" \
    -a "${RSYNC_COMMON_OPTIONS[@]}" -e "${SSH_RSYNC_CMD}" \
    "${LOCAL_PAYLOAD_DIR}/dummy_job.sh" \
    "${EVE_REMOTE_HOST}:${STAGING_DIR}/"

  rsync_resilient \
    "stage harness config" \
    -a "${RSYNC_COMMON_OPTIONS[@]}" -e "${SSH_RSYNC_CMD}" \
    "${LOCAL_CONFIG_PATH}" \
    "${EVE_REMOTE_HOST}:${STAGING_DIR}/harness_config.json"

  (
    cd "${LOCAL_PAYLOAD_DIR}"
    rsync_resilient \
      "stage harness inputs" \
      -a --checksum "${RSYNC_COMMON_OPTIONS[@]}" --relative -e "${SSH_RSYNC_CMD}" \
      "inputs/dummy_input.txt" \
      "${EVE_REMOTE_HOST}:${STAGING_DIR}/"
  )
}

stage_v2_payload() {
  rsync_resilient \
    "stage harness payload batch" \
    -a "${RSYNC_COMMON_OPTIONS[@]}" -e "${SSH_RSYNC_CMD}" \
    "${LOCAL_PAYLOAD_DIR}/" \
    "${EVE_REMOTE_HOST}:${STAGING_DIR}/"
}

submit_dummy_job() {
  local label="$1"
  local submit_cmd="timeout 180 qsub -v STAGING_DIR=${STAGING_DIR},HARNESS_ATTEMPT_ID=${ATTEMPT_ID},HARNESS_SLEEP_SECONDS=${JOB_SLEEP_SECONDS} -o '${STAGING_DIR}/orchestrator_stdout.log' -e '${STAGING_DIR}/orchestrator_stderr.log'"
  local raw=""
  local job_id=""

  raw="$(
    ssh_remote_resilient "submit ${label} PBS job" "
      set -euo pipefail
      cd '${REMOTE_WORKTREE}'
      ${submit_cmd} 'dummy_job.sh' 2>&1
    "
  )" || true
  printf '%s\n' "${raw}" > "${SUBMIT_RAW_PATH}"
  job_id="$(printf '%s\n' "${raw}" | awk '/[0-9]+\./ {print $1; exit}')"
  if [[ -n "${job_id}" ]]; then
    printf '%s\n' "${job_id}"
    return 0
  fi
  printf 'Failed to capture harness job id from raw qsub output:\n%s\n' "${raw}" >&2
  return 1
}

start_watcher_for_job() {
  local job_id="$1"

  ssh_remote_resilient "start harness watcher" "
    set -euo pipefail
    cd '${STAGING_DIR}'
    nohup python3 '${STAGING_DIR}/status_watcher.py' --job-id '${job_id}' --status-path '${STAGING_DIR}/status.json' --result-path '${STAGING_DIR}/result.json' --poll-seconds ${WATCHER_POLL_SECONDS} --max-seconds 2400 > '${STAGING_DIR}/watcher.log' 2>&1 < /dev/null &
  " >/dev/null
}

write_summary() {
  python3 - "${SUMMARY_JSON_PATH}" "$@" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
path.write_text(json.dumps(payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
print(json.dumps(payload, indent=2, sort_keys=False))
PY
}

summarize_metrics() {
  python3 - "${METRICS_FILE}" "${HARNESS_WORKFLOW_ID}" <<'PY'
import json
import sys
from pathlib import Path

metrics_path = Path(sys.argv[1])
workflow_id = sys.argv[2]
events = []
if metrics_path.exists():
    for line in metrics_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload.get("workflow_id") == workflow_id:
            events.append(payload)

def is_close_command(event):
    argv = event.get("argv", [])
    return argv[:2] == ["-O", "exit"]

def is_rsync_transport_ssh(event):
    argv = event.get("argv", [])
    return "rsync" in argv and "--server" in argv

ssh_events = [
    event
    for event in events
    if event.get("kind") == "ssh"
    and not event.get("config_only")
    and not is_close_command(event)
    and not is_rsync_transport_ssh(event)
]
handshake_events = [
    event
    for event in events
    if event.get("kind") == "ssh"
    and not event.get("config_only")
    and not is_close_command(event)
]
rsync_events = [event for event in events if event.get("kind") == "rsync"]
summary = {
    "ssh_operations": len(ssh_events),
    "rsync_operations": len(rsync_events),
    "total_operations": len(ssh_events) + len(rsync_events),
    "fresh_handshakes": sum(1 for event in handshake_events if event.get("fresh_handshake")),
}
print(json.dumps(summary))
PY
}

reconcile_status_once() {
  local job_id="$1"
  local label="$2"
  local output=""
  local job_state=""
  local exit_status=""

  output="$(ssh_remote_resilient "reconcile ${label} job ${job_id}" "timeout 30 qstat -fx '${job_id}'")"
  LAST_QSTAT_OUTPUT="${output}"
  job_state="$(printf '%s\n' "${output}" | awk -F' = ' '/job_state = / {print $2; exit}')"
  exit_status="$(printf '%s\n' "${output}" | awk -F' = ' '/Exit_status = / {print $2; exit}')"
  python3 - "${job_state}" "${exit_status}" <<'PY'
import json
import sys

print(json.dumps({"job_state": sys.argv[1], "exit_status": sys.argv[2] or None}))
PY
}

read_remote_status_json() {
  ssh_remote_resilient "read harness status.json" "
    set -euo pipefail
    cat '${STAGING_DIR}/status.json'
  "
}

watch_status_until_terminal() {
  local job_id="$1"
  local start_epoch
  local killed_socket=0
  local forced_reconcile=0
  start_epoch="$(date +%s)"

  while true; do
    if (( STATUS_INITIAL_DELAY_SECONDS > 0 )) && (( $(date +%s) - start_epoch < STATUS_INITIAL_DELAY_SECONDS )); then
      sleep 1
      continue
    fi
    if (( HARNESS_INTERRUPT_SOCKET == 1 )) && (( killed_socket == 0 )) && (( $(date +%s) - start_epoch >= INTERRUPT_AFTER_SECONDS )); then
      close_ssh_master
      killed_socket=1
      forced_reconcile=1
    fi

    if (( forced_reconcile == 1 )); then
      reconcile_status_once "${job_id}" "watcher" >/dev/null
      forced_reconcile=0
    fi

    local raw_status
    if ! raw_status="$(read_remote_status_json)"; then
      reconcile_status_once "${job_id}" "watcher" >/dev/null
      sleep "${STATUS_POLL_SECONDS}"
      continue
    fi

    local parsed
    parsed="$(python3 - <<'PY' "${raw_status}"
import json
import sys

payload = json.loads(sys.argv[1])
states = {
    "Q": "queued",
    "R": "running",
    "C": "finished",
    "F": "finished",
}
print(json.dumps({
    "job_state": payload.get("job_state") or "",
    "normalized_state": states.get(payload.get("job_state") or "", payload.get("watcher_status") or ""),
    "exit_status": payload.get("exit_status"),
    "heartbeat": payload.get("heartbeat"),
    "result_present": bool(payload.get("result_present")),
}))
PY
)"
    local normalized_state
    normalized_state="$(python3 - <<'PY' "${parsed}"
import json
import sys
print(json.loads(sys.argv[1])["normalized_state"])
PY
)"
    local exit_status
    exit_status="$(python3 - <<'PY' "${parsed}"
import json
import sys
value = json.loads(sys.argv[1])["exit_status"]
print("" if value is None else value)
PY
)"
    local result_present
    result_present="$(python3 - <<'PY' "${parsed}"
import json
import sys
print("1" if json.loads(sys.argv[1])["result_present"] else "0")
PY
)"

    if [[ "${normalized_state}" == "finished" && "${exit_status}" == "0" && "${result_present}" == "1" ]]; then
      return 0
    fi
    if [[ "${normalized_state}" == "finished" && "${exit_status}" != "0" ]]; then
      return "${WAIT_PBS_CANDIDATE_FAILURE}"
    fi

    sleep "${STATUS_POLL_SECONDS}"
  done
}

run_single_workflow() {
  local workflow_mode="$1"
  local workflow_id="$2"
  INTERNAL_WORKFLOW_ID="${workflow_id}"
  ATTEMPT_ID="${workflow_mode}_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}"
  prepare_metering_wrappers
  prepare_local_payload

  if [[ "${workflow_mode}" == "current" ]]; then
    configure_current_transport
    resolve_harness_remote_user
    create_remote_staging
    stage_current_payload
    local job_id
    job_id="$(submit_dummy_job "harness current")"
    local pbs_wait_rc=0
    if wait_for_pbs "${job_id}" 30 3600 "harness current"; then
      pbs_wait_rc=0
    else
      pbs_wait_rc=$?
    fi
    if [[ "${pbs_wait_rc}" -eq "${WAIT_PBS_TRANSPORT_HALT}" ]]; then
      if reconcile_pbs_after_transport_halt "${job_id}" 15 180 "harness current"; then
        pbs_wait_rc=0
      else
        pbs_wait_rc=$?
      fi
    fi
    fetch_result_json_with_retry_current "${STAGING_DIR}/result.json" "${RESULT_PATH}" || true
    fetch_remote_file_once "fetch harness stdout" "${STAGING_DIR}/orchestrator_stdout.log" "${ORCHESTRATOR_STDOUT_PATH}" || true
    fetch_remote_file_once "fetch harness stderr" "${STAGING_DIR}/orchestrator_stderr.log" "${ORCHESTRATOR_STDERR_PATH}" || true
    if [[ ! -s "${RESULT_PATH}" ]]; then
      start_current_result_fetch_diagnostics "${job_id}"
      pbs_wait_rc=1
    fi
    local metrics_json
    metrics_json="$(summarize_metrics)"
    local passed="false"
    if [[ "${pbs_wait_rc}" -eq 0 && -s "${RESULT_PATH}" ]]; then
      passed="true"
    fi
    write_summary "$(python3 - <<'PY' "${workflow_mode}" "${workflow_id}" "${job_id}" "${passed}" "${metrics_json}" "${HARNESS_RUN_ROOT}" "${RESULT_PATH}" "${ORCHESTRATOR_STDOUT_PATH}" "${ORCHESTRATOR_STDERR_PATH}" "${SUBMIT_RAW_PATH}"
import json
import sys

payload = {
    "mode": "workflow",
    "transport_mode": sys.argv[1],
    "workflow_id": sys.argv[2],
    "job_id": sys.argv[3],
    "passed": sys.argv[4] == "true",
    "metrics": json.loads(sys.argv[5]),
    "run_root": sys.argv[6],
    "artifacts": {
        "result_json": sys.argv[7],
        "stdout_log": sys.argv[8],
        "stderr_log": sys.argv[9],
        "submit_raw_log": sys.argv[10],
    },
}
print(json.dumps(payload))
PY
)"
  else
    local shared_socket="${PARENT_SOCKET_PATH:-/tmp/eve-ssh-ctl-${USER}-icon-transport-harness.sock}"
    if [[ -n "${PARENT_SOCKET_PATH}" ]]; then
      HARNESS_LEAVE_MASTER_OPEN=1
    fi
    configure_v2_transport "${shared_socket}" "30m"
    resolve_harness_remote_user
    create_remote_staging
    stage_v2_payload
    local job_id
    job_id="$(submit_dummy_job "harness v2")"
    start_watcher_for_job "${job_id}"
    watch_status_until_terminal "${job_id}"
    local fetch_dir="${HARNESS_RUN_ROOT}/${HARNESS_WORKFLOW_ID}"
    fetch_batch_artifacts_v2 "${STAGING_DIR}" "${fetch_dir}" || true
    cp "${fetch_dir}/result.json" "${RESULT_PATH}" 2>/dev/null || true
    cp "${fetch_dir}/orchestrator_stdout.log" "${ORCHESTRATOR_STDOUT_PATH}" 2>/dev/null || true
    cp "${fetch_dir}/orchestrator_stderr.log" "${ORCHESTRATOR_STDERR_PATH}" 2>/dev/null || true
    cp "${fetch_dir}/status.json" "${STATUS_JSON_PATH}" 2>/dev/null || true
    cp "${fetch_dir}/watcher.log" "${WATCHER_LOG_PATH}" 2>/dev/null || true
    local metrics_json
    metrics_json="$(summarize_metrics)"
    local passed="false"
    if [[ -s "${RESULT_PATH}" ]]; then
      passed="true"
    fi
    write_summary "$(python3 - <<'PY' "${workflow_mode}" "${workflow_id}" "${job_id}" "${passed}" "${metrics_json}" "${HARNESS_RUN_ROOT}" "${RESULT_PATH}" "${STATUS_JSON_PATH}" "${WATCHER_LOG_PATH}" "${SUBMIT_RAW_PATH}"
import json
import sys

payload = {
    "mode": "workflow",
    "transport_mode": sys.argv[1],
    "workflow_id": sys.argv[2],
    "job_id": sys.argv[3],
    "passed": sys.argv[4] == "true",
    "metrics": json.loads(sys.argv[5]),
    "run_root": sys.argv[6],
    "artifacts": {
        "result_json": sys.argv[7],
        "status_json": sys.argv[8],
        "watcher_log": sys.argv[9],
        "submit_raw_log": sys.argv[10],
    },
}
print(json.dumps(payload))
PY
)"
  fi
}

run_handshake_count() {
  local workflow_id="${TRANSPORT_MODE}-baseline"
  if [[ "${TRANSPORT_MODE}" == "v2" ]]; then
    STATUS_INITIAL_DELAY_SECONDS="${STATUS_POLL_SECONDS}"
  fi
  run_single_workflow "${TRANSPORT_MODE}" "${workflow_id}" >/dev/null
  local summary_file="${SUMMARY_JSON_PATH}"
  python3 - "${summary_file}" "${TRANSPORT_MODE}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
metrics = summary["metrics"]
passed = metrics["total_operations"] <= 10 and metrics["fresh_handshakes"] <= 5
result = {
    "mode": "handshake_count",
    "transport_mode": sys.argv[2],
    "passed": passed,
    "counts": metrics,
    "run_root": summary["run_root"],
    "job_id": summary["job_id"],
    "artifacts": summary["artifacts"],
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=False))
PY
}

run_watcher_basic() {
  local observed_json="${HARNESS_RUN_ROOT}/watcher_basic_states.json"
  run_single_workflow "v2" "watcher-basic" >/dev/null
  python3 - "${STATUS_JSON_PATH}" "${WATCHER_LOG_PATH}" "${SUMMARY_JSON_PATH}" "${observed_json}" <<'PY'
import json
import sys
from pathlib import Path

status = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
watcher_log = Path(sys.argv[2]).read_text(encoding="utf-8") if Path(sys.argv[2]).exists() else ""
summary = json.loads(Path(sys.argv[3]).read_text(encoding="utf-8"))
observed_states = ["queued"]
if "state=running" in watcher_log or status.get("watcher_status") == "running":
    observed_states.append("running")
observed_states.append("finished")
result = {
    "mode": "watcher_basic",
    "passed": observed_states == ["queued", "running", "finished"] and summary["passed"],
    "observed_states": observed_states,
    "run_root": summary["run_root"],
    "artifacts": summary["artifacts"],
}
Path(sys.argv[4]).write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
Path(sys.argv[3]).write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=False))
PY
}

run_transport_interrupt() {
  HARNESS_INTERRUPT_SOCKET=1
  run_single_workflow "v2" "transport-interrupt" >/dev/null
  python3 - "${SUMMARY_JSON_PATH}" "${INTERRUPT_AFTER_SECONDS}" <<'PY'
import json
import sys
from pathlib import Path

summary = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
result = {
    "mode": "transport_interrupt",
    "passed": summary["passed"],
    "interrupt_after_seconds": int(sys.argv[2]),
    "run_root": summary["run_root"],
    "artifacts": summary["artifacts"],
    "metrics": summary["metrics"],
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=False))
PY
}

run_parallel_load() {
  local shared_socket="/tmp/eve-ssh-ctl-${USER}-icon-transport-harness-parallel.sock"
  local workflow
  local pids=()
  local summary_files=()

  ATTEMPT_ID="parallel_prewarm_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}"
  prepare_metering_wrappers
  configure_v2_transport "${shared_socket}" "30m"
  ssh_remote "echo ok" >/dev/null

  for workflow in $(seq 1 "${CONCURRENCY}"); do
    local child_root="${HARNESS_RUN_ROOT}/parallel-${workflow}"
    summary_files+=("${child_root}/summary.json")
    bash "${script_dir}/transport_harness.sh" \
      --mode workflow \
      --transport-mode v2 \
      --workflow-id "parallel-${workflow}" \
      --run-root "${child_root}" \
      --shared-socket-path "${shared_socket}" \
      --job-sleep-seconds "${JOB_SLEEP_SECONDS}" &
    pids+=("$!")
  done

  local failures=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failures=$((failures + 1))
    fi
  done

  close_ssh_master || true

  python3 - "${SUMMARY_JSON_PATH}" "${CONCURRENCY}" "${failures}" "${summary_files[@]}" <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
concurrency = int(sys.argv[2])
failures = int(sys.argv[3])
workflow_summaries = []
for path in sys.argv[4:]:
    workflow_summaries.append(json.loads(Path(path).read_text(encoding="utf-8")))

failure_rate = failures / concurrency if concurrency else 0.0
result = {
    "mode": "parallel_load",
    "passed": failure_rate <= 0.05 and all(item["passed"] for item in workflow_summaries),
    "concurrency": concurrency,
    "workflow_failures": failures,
    "failure_rate": failure_rate,
    "workflows": workflow_summaries,
    "run_root": str(summary_path.parent),
}
summary_path.write_text(json.dumps(result, indent=2, sort_keys=False) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=False))
PY
}

case "${MODE}" in
  workflow)
    if [[ -z "${INTERNAL_WORKFLOW_ID}" ]]; then
      printf 'workflow mode requires --workflow-id\n' >&2
      exit 2
    fi
    run_single_workflow "${TRANSPORT_MODE}" "${INTERNAL_WORKFLOW_ID}"
    ;;
  handshake_count)
    run_handshake_count
    ;;
  watcher_basic)
    run_watcher_basic
    ;;
  transport_interrupt)
    run_transport_interrupt
    ;;
  parallel_load)
    run_parallel_load
    ;;
  *)
    printf 'Unknown mode: %s\n' "${MODE}" >&2
    exit 2
    ;;
esac
