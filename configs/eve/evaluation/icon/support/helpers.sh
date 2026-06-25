#!/usr/bin/env bash

timestamp_utc() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

mark_transport_halt() {
  OUTCOME_IS_TRANSPORT_HALT=1
  OUTCOME_IS_CANDIDATE_FAILURE=0
  ERROR_MESSAGE="$1"
}

halt_transport() {
  mark_transport_halt "$1"
  return 1
}

init_remote_transport() {
  local user_tag="${USER:-unknown}"
  local host_tag="${EVE_REMOTE_HOST//[^A-Za-z0-9_.-]/_}"

  : "${ATTEMPT_ID:?ATTEMPT_ID must be set before init_remote_transport}"

  SSH_SHARED_CONTROL_MASTER="${SSH_SHARED_CONTROL_MASTER:-1}"
  SSH_CONTROL_PERSIST="${SSH_CONTROL_PERSIST:-30m}"
  if [[ "${SSH_SHARED_CONTROL_MASTER}" == "1" ]]; then
    SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-/tmp/eve-ssh-ctl-${user_tag}-${host_tag}.sock}"
  else
    SSH_CONTROL_PATH="${SSH_CONTROL_PATH:-/tmp/eve-ssh-ctl-${user_tag}-${ATTEMPT_ID}.sock}"
  fi
  SSH_CONTROL_REF_FILE="${SSH_CONTROL_PATH}.refs.json"
  SSH_CONTROL_LOCK_FILE="${SSH_CONTROL_PATH}.lock"
  SSH_OPTIONS=(
    -S "${SSH_CONTROL_PATH}"
    -o ControlMaster=auto
    -o ControlPersist="${SSH_CONTROL_PERSIST}"
    -o PreferredAuthentications=publickey
    -o BatchMode=yes
    -o ConnectTimeout=10
    -o ServerAliveInterval=15
    -o ServerAliveCountMax=4
  )
  SSH_RSYNC_CMD="${SSH_BIN} -S ${SSH_CONTROL_PATH} -o ControlMaster=auto -o ControlPersist=${SSH_CONTROL_PERSIST} -o PreferredAuthentications=publickey -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=15 -o ServerAliveCountMax=4"
  RSYNC_COMMON_OPTIONS=(
    --partial
    --partial-dir=.rsync-partial
    --timeout=30
  )
  # Keep resume semantics as the primary guarantee. The local rsync client
  # rejects combining --partial-dir with --inplace.
  RSYNC_INPLACE_OPTIONS=()
  register_ssh_master_ref
}

close_ssh_master() {
  local socket_path="${SSH_CONTROL_PATH:-}"
  local should_close="1"

  if [[ -z "${socket_path}" ]]; then
    return 0
  fi

  if [[ "${SSH_SHARED_CONTROL_MASTER:-1}" == "1" ]]; then
    should_close="$(python3 - "${SSH_CONTROL_REF_FILE}" "${SSH_CONTROL_LOCK_FILE}" <<'PY'
import fcntl
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
lock_path = Path(sys.argv[2])

state = {"refs": 0}
lock_path.parent.mkdir(parents=True, exist_ok=True)

with lock_path.open("a+", encoding="utf-8") as lock_fp:
    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state.update(loaded)
        except Exception:
            pass
    refs = max(0, int(state.get("refs", 0) or 0) - 1)
    if refs == 0:
        try:
            state_path.unlink()
        except FileNotFoundError:
            pass
        print("1")
    else:
        state["refs"] = refs
        state_path.write_text(json.dumps(state), encoding="utf-8")
        print("0")
PY
)"
  fi

  if [[ "${should_close}" != "1" ]]; then
    return 0
  fi

  "${SSH_BIN}" -S "${socket_path}" -O exit "${EVE_REMOTE_HOST}" >/dev/null 2>&1 || true
  rm -f "${socket_path}" >/dev/null 2>&1 || true
  rm -f "${SSH_CONTROL_LOCK_FILE:-}" >/dev/null 2>&1 || true
}

register_ssh_master_ref() {
  if [[ "${SSH_SHARED_CONTROL_MASTER:-1}" != "1" ]]; then
    return 0
  fi
  python3 - "${SSH_CONTROL_REF_FILE}" "${SSH_CONTROL_LOCK_FILE}" "${SSH_CONTROL_PATH}" <<'PY'
import fcntl
import json
import sys
from pathlib import Path

state_path = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
socket_path = Path(sys.argv[3])

state = {"refs": 0}
lock_path.parent.mkdir(parents=True, exist_ok=True)

with lock_path.open("a+", encoding="utf-8") as lock_fp:
    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state.update(loaded)
        except Exception:
            pass
    if not socket_path.exists():
        state = {"refs": 0}
    state["refs"] = int(state.get("refs", 0) or 0) + 1
    state_path.write_text(json.dumps(state), encoding="utf-8")
PY
}

ssh_remote() {
  local attempt=1
  local max_attempts=3
  local rc=0

  while (( attempt <= max_attempts )); do
    if "${SSH_BIN}" "${SSH_OPTIONS[@]}" "${EVE_REMOTE_HOST}" "$@"; then
      return 0
    fi
    rc=$?
    if (( rc != 255 || attempt == max_attempts )); then
      return "${rc}"
    fi
    sleep $((attempt * 2))
    attempt=$((attempt + 1))
  done

  return "${rc}"
}

rsync_with_retry() {
  local attempt=1
  local max_attempts=3
  local rc=0

  while (( attempt <= max_attempts )); do
    if "${RSYNC_BIN}" "$@"; then
      return 0
    fi
    rc=$?
    if (( attempt == max_attempts )); then
      return "${rc}"
    fi
    sleep $((attempt * 2))
    attempt=$((attempt + 1))
  done

  return "${rc}"
}

compute_l2_sleep_seconds() {
  local attempt="$1"
  local sleep_seconds="${L2_BACKOFF_FLOOR_SECONDS}"
  local jitter=0

  sleep_seconds=$((L2_BACKOFF_FLOOR_SECONDS * (1 << (attempt - 1))))
  if (( sleep_seconds < L2_BACKOFF_FLOOR_SECONDS )); then
    sleep_seconds="${L2_BACKOFF_FLOOR_SECONDS}"
  fi
  if (( sleep_seconds > L2_BACKOFF_MAX_SECONDS )); then
    sleep_seconds="${L2_BACKOFF_MAX_SECONDS}"
  fi
  if (( L2_JITTER_MAX_SECONDS > 0 )); then
    jitter=$((RANDOM % (L2_JITTER_MAX_SECONDS + 1)))
  fi
  printf '%s\n' $((sleep_seconds + jitter))
}

probe_health() {
  local output=""

  output="$(
    "${SSH_BIN}" \
      "${SSH_OPTIONS[@]}" \
      -o ConnectTimeout="${PROBE_CONNECT_TIMEOUT_SECONDS}" \
      "${EVE_REMOTE_HOST}" \
      'echo ok' 2>/dev/null
  )" || return $?
  [[ "$(printf '%s' "${output}" | tr -d '\r\n')" == "ok" ]]
}

parse_breaker_result() {
  local raw="$1"
  local key="$2"

  printf '%s\n' "${raw}" | awk -F'=' -v key="${key}" '$1 == key {sub(/^[^=]+=*/, "", $0); print; exit}'
}

breaker_request_probe_slot() {
  local label="$1"
  local raw=""
  local action=""
  local sleep_seconds=0

  while true; do
    raw="$(
      python3 - \
        "${BREAKER_STATE_FILE}" \
        "${BREAKER_LOCK_FILE}" \
        "${BREAKER_COOLDOWN_SECONDS}" \
        "${BREAKER_HALF_OPEN_LEASE_SECONDS}" \
        "${BREAKER_HALT_WINDOW_SECONDS}" \
        "$$" <<'PY'
import fcntl
import json
import sys
import time
from pathlib import Path

state_path = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
cooldown = int(sys.argv[3])
lease = int(sys.argv[4])
halt_window = int(sys.argv[5])
owner = sys.argv[6]

state = {
    "state": "closed",
    "opened_at": 0,
    "cooldown_until": 0,
    "probe_owner": "",
    "probe_lease_until": 0,
}

state_path.parent.mkdir(parents=True, exist_ok=True)
lock_path.parent.mkdir(parents=True, exist_ok=True)

with lock_path.open("a+", encoding="utf-8") as lock_fp:
    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state.update(loaded)
        except Exception:
            pass

    now = int(time.time())
    opened_at = int(state.get("opened_at", 0) or 0)
    cooldown_until = int(state.get("cooldown_until", 0) or 0)
    probe_lease_until = int(state.get("probe_lease_until", 0) or 0)
    probe_owner = str(state.get("probe_owner", "") or "")
    action = "sleep"
    sleep_seconds = 0

    if opened_at and now - opened_at >= halt_window:
        action = "halt"
    elif state.get("state") not in {"open", "half_open"}:
        state.update(
            {
                "state": "open",
                "opened_at": now,
                "cooldown_until": now + cooldown,
                "probe_owner": "",
                "probe_lease_until": 0,
            }
        )
        state_path.write_text(json.dumps(state), encoding="utf-8")
        action = "sleep"
        sleep_seconds = cooldown
    elif now < cooldown_until:
        action = "sleep"
        sleep_seconds = max(1, cooldown_until - now)
    elif (not probe_owner) or probe_lease_until <= now or probe_owner == owner:
        state.update(
            {
                "state": "half_open",
                "probe_owner": owner,
                "probe_lease_until": now + lease,
            }
        )
        state_path.write_text(json.dumps(state), encoding="utf-8")
        action = "probe"
    else:
        action = "sleep"
        sleep_seconds = max(1, probe_lease_until - now)

print(f"ACTION={action}")
print(f"SLEEP_SECONDS={sleep_seconds}")
print(f"OPENED_AT={state.get('opened_at', 0) or 0}")
PY
    )"
    action="$(parse_breaker_result "${raw}" "ACTION")"
    sleep_seconds="$(parse_breaker_result "${raw}" "SLEEP_SECONDS")"
    sleep_seconds="${sleep_seconds:-0}"

    case "${action}" in
      probe)
        return 0
        ;;
      halt)
        printf '%s [REMOTE-L4] breaker open window exceeded for %s; halting wrapper\n' \
          "$(timestamp_utc)" "${label}" >&2
        return 1
        ;;
      *)
        printf '%s [REMOTE-L3] breaker open for %s; sleeping %ss before half-open probe\n' \
          "$(timestamp_utc)" "${label}" "${sleep_seconds}" >&2
        sleep "${sleep_seconds}"
        ;;
    esac
  done
}

breaker_record_probe_result() {
  local result="$1"

  python3 - \
    "${BREAKER_STATE_FILE}" \
    "${BREAKER_LOCK_FILE}" \
    "${BREAKER_COOLDOWN_SECONDS}" \
    "${result}" <<'PY'
import fcntl
import json
import sys
import time
from pathlib import Path

state_path = Path(sys.argv[1])
lock_path = Path(sys.argv[2])
cooldown = int(sys.argv[3])
result = sys.argv[4]

state = {
    "state": "closed",
    "opened_at": 0,
    "cooldown_until": 0,
    "probe_owner": "",
    "probe_lease_until": 0,
}

state_path.parent.mkdir(parents=True, exist_ok=True)
lock_path.parent.mkdir(parents=True, exist_ok=True)

with lock_path.open("a+", encoding="utf-8") as lock_fp:
    fcntl.flock(lock_fp.fileno(), fcntl.LOCK_EX)
    if state_path.exists():
        try:
            loaded = json.loads(state_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                state.update(loaded)
        except Exception:
            pass

    now = int(time.time())
    if result == "success":
        state = {
            "state": "closed",
            "opened_at": 0,
            "cooldown_until": 0,
            "probe_owner": "",
            "probe_lease_until": 0,
        }
    else:
        opened_at = int(state.get("opened_at", 0) or 0) or now
        state.update(
            {
                "state": "open",
                "opened_at": opened_at,
                "cooldown_until": now + cooldown,
                "probe_owner": "",
                "probe_lease_until": 0,
            }
        )

    state_path.write_text(json.dumps(state), encoding="utf-8")
PY
}

run_transport_operation() {
  local label="$1"
  shift

  local local_attempt=1
  local rc=0
  local sleep_seconds=0
  local probe_rc=0

  while true; do
    if "$@"; then
      return 0
    fi
    rc=$?

    while (( local_attempt <= L2_RETRY_ATTEMPTS )); do
      sleep_seconds="$(compute_l2_sleep_seconds "${local_attempt}")"
      printf '%s [REMOTE-L2] %s failed (rc=%d); retry %d/%d in %ss\n' \
        "$(timestamp_utc)" "${label}" "${rc}" "${local_attempt}" "${L2_RETRY_ATTEMPTS}" "${sleep_seconds}" >&2
      sleep "${sleep_seconds}"
      if ! probe_health; then
        probe_rc=$?
        printf '%s [REMOTE-L2] health probe failed before retrying %s (rc=%d); skipping expensive operation\n' \
          "$(timestamp_utc)" "${label}" "${probe_rc}" >&2
        local_attempt=$((local_attempt + 1))
        continue
      fi
      printf '%s [REMOTE-L2] health probe succeeded; retrying %s\n' \
        "$(timestamp_utc)" "${label}" >&2
      if "$@"; then
        return 0
      fi
      rc=$?
      local_attempt=$((local_attempt + 1))
    done

    printf '%s [REMOTE-L3] %s exhausted local retries; opening shared breaker\n' \
      "$(timestamp_utc)" "${label}" >&2
    if ! breaker_request_probe_slot "${label}"; then
      mark_transport_halt \
        "${label} exceeded the shared remote cluster transport breaker halt window after repeated failures."
      return 1
    fi

    if probe_health; then
      printf '%s [REMOTE-L3] shared half-open probe succeeded; closing breaker for %s\n' \
        "$(timestamp_utc)" "${label}" >&2
      breaker_record_probe_result "success"
      local_attempt=1
      continue
    fi

    probe_rc=$?
    printf '%s [REMOTE-L3] shared half-open probe failed for %s (rc=%d); extending breaker cooldown\n' \
      "$(timestamp_utc)" "${label}" "${probe_rc}" >&2
    breaker_record_probe_result "failure"
  done
}

ssh_remote_resilient() {
  local label="$1"
  shift
  run_transport_operation "${label}" ssh_remote "$@"
}

rsync_resilient() {
  local label="$1"
  shift
  run_transport_operation "${label}" rsync_with_retry "$@"
}

submit_pbs_with_retry() {
  local pbs_path="$1"
  local submit_cmd="$2"
  local label="$3"
  local attempt=1
  local max_attempts=3
  local raw=""
  local job_id=""
  local last_raw=""

  while (( attempt <= max_attempts )); do
    if raw="$(
      ssh_remote_resilient "submit ${label} PBS job" "
        set -euo pipefail
        cd '${REMOTE_WORKTREE}'
        ${submit_cmd} '${pbs_path}' 2>&1
      "
    )"; then
      last_raw="${raw}"
      job_id="$(printf '%s\n' "${raw}" | awk '/[0-9]+\./ {print $1; exit}')"
      if [[ -n "${job_id}" ]]; then
        printf '%s\n' "${job_id}"
        return 0
      fi
    else
      last_raw="${raw}"
      if (( OUTCOME_IS_TRANSPORT_HALT == 1 )); then
        return 1
      fi
    fi

    if (( attempt == max_attempts )); then
      mark_transport_halt \
        "Failed to capture ${label} PBS job id after ${max_attempts} attempts. Last raw qsub output: ${last_raw}"
      return 1
    fi

    sleep $((attempt * 5))
    attempt=$((attempt + 1))
  done

  return 1
}

normalize_remote_pbs_path() {
  local raw_path="$1"

  if [[ -z "${raw_path}" ]]; then
    return 1
  fi
  if [[ "${raw_path}" == *:* ]]; then
    printf '%s\n' "${raw_path#*:}"
    return 0
  fi
  printf '%s\n' "${raw_path}"
}

copy_remote_text_file() {
  local remote_path="$1"
  local local_path="$2"
  local tmp_path="${local_path}.tmp"

  [[ -n "${remote_path}" ]] || return 1

  if ! ssh_remote "test -f '${remote_path}'" >/dev/null 2>&1; then
    return 1
  fi

  mkdir -p "$(dirname "${local_path}")"
  if ! ssh_remote "cat '${remote_path}'" >"${tmp_path}"; then
    rm -f "${tmp_path}"
    return 1
  fi

  mv "${tmp_path}" "${local_path}"
}

fetch_remote_artifacts_batch() {
  local label="$1"
  local remote_dir="$2"
  local local_dir="$3"
  shift 3

  local args=()
  local include_name=""
  mkdir -p "${local_dir}"
  args+=(-a "${RSYNC_COMMON_OPTIONS[@]}" -e "${SSH_RSYNC_CMD}")
  for include_name in "$@"; do
    args+=("--include=${include_name}")
  done
  args+=("--exclude=*")
  args+=("${EVE_REMOTE_HOST}:${remote_dir}/" "${local_dir}/")
  rsync_resilient "${label}" "${args[@]}"
}

submit_watched_pbs_job() {
  local launcher_name="$1"
  local pbs_script_name="$2"
  local label="$3"
  local raw=""
  local job_id=""

  raw="$(
    ssh_remote_resilient "submit watched ${label} PBS job" "
      set -euo pipefail
      STAGING_DIR='${STAGING_DIR}' \
      PBS_SCRIPT_NAME='${pbs_script_name}' \
      WATCHER_SCRIPT_NAME='status_watcher.py' \
      WATCHER_STATUS_PATH='${STAGING_DIR}/status.json' \
      WATCHER_RESULT_PATH='${STAGING_DIR}/result.json' \
      ORCHESTRATOR_STDOUT_PATH='${STAGING_DIR}/orchestrator_stdout.log' \
      ORCHESTRATOR_STDERR_PATH='${STAGING_DIR}/orchestrator_stderr.log' \
      WATCHER_LOG_PATH='${STAGING_DIR}/watcher.log' \
      WATCHER_POLL_SECONDS='5' \
      WATCHER_MAX_SECONDS='2400' \
      bash '${STAGING_DIR}/${launcher_name}' 2>&1
    "
  )" || true
  job_id="$(printf '%s\n' "${raw}" | awk '/[0-9]+\./ {print $1; exit}')"
  if [[ -n "${job_id}" ]]; then
    printf '%s\n' "${job_id}"
    return 0
  fi
  mark_transport_halt \
    "Failed to capture watched ${label} PBS job id. Last raw launcher output: ${raw}"
  return 1
}

copy_remote_pbs_logs() {
  local job_id="$1"
  local label="$2"
  local qstat_output="${3:-}"
  local output_path=""
  local error_path=""
  local normalized_output_path=""
  local normalized_error_path=""
  local copied_paths=()
  local local_output_path=""
  local local_error_path=""
  local joined_paths=""

  if [[ -z "${qstat_output}" ]]; then
    qstat_output="$(ssh_remote "timeout 30 qstat -fx '${job_id}'" 2>&1 || true)"
  fi

  output_path="$(printf '%s\n' "${qstat_output}" | awk -F' = ' '/Output_Path = / {print $2; exit}')"
  error_path="$(printf '%s\n' "${qstat_output}" | awk -F' = ' '/Error_Path = / {print $2; exit}')"
  normalized_output_path="$(normalize_remote_pbs_path "${output_path}" || true)"
  normalized_error_path="$(normalize_remote_pbs_path "${error_path}" || true)"

  if [[ -n "${normalized_output_path}" ]]; then
    local_output_path="${LOGS_DIR}/pbs_${label}_${job_id}.o"
    if copy_remote_text_file "${normalized_output_path}" "${local_output_path}"; then
      copied_paths+=("${local_output_path}")
    fi
  fi
  if [[ -n "${normalized_error_path}" ]]; then
    local_error_path="${LOGS_DIR}/pbs_${label}_${job_id}.e"
    if copy_remote_text_file "${normalized_error_path}" "${local_error_path}"; then
      copied_paths+=("${local_error_path}")
    fi
  fi

  if (( ${#copied_paths[@]} == 0 )); then
    return 1
  fi

  joined_paths="$(IFS=', '; printf '%s' "${copied_paths[*]}")"
  printf 'Copied remote PBS logs to %s\n' "${joined_paths}"
}

emit_pbs_fetch_diagnostics() {
  local job_id="$1"
  local label="$2"
  local copied_paths=""

  if [[ -n "${LAST_QSTAT_OUTPUT:-}" ]]; then
    printf '%s\n' "${LAST_QSTAT_OUTPUT}" >&2
  fi
  if copied_paths="$(copy_remote_pbs_logs "${job_id}" "${label}" "${LAST_QSTAT_OUTPUT}" 2>/dev/null)"; then
    printf '%s\n' "${copied_paths}" >&2
  fi
}

kill_remote_job() {
  local job_id="$1"
  local label="$2"

  ssh_remote_resilient "cancel ${label} job ${job_id}" "timeout 60 qdel '${job_id}'" >/dev/null 2>&1 || true
}

reconcile_pbs_after_transport_halt() {
  local job_id="$1"
  local poll_interval="$2"
  local grace_seconds="$3"
  local label="$4"
  local deadline=0
  local output=""
  local job_state=""
  local exit_status=""

  deadline=$(( $(date +%s) + grace_seconds ))
  printf '%s [REMOTE-RECONCILE] transport halted while polling %s job %s; allowing %ss to re-check PBS state\n' \
    "$(timestamp_utc)" "${label}" "${job_id}" "${grace_seconds}" >&2

  while (( $(date +%s) < deadline )); do
    if output="$(ssh_remote "timeout 30 qstat -fx '${job_id}'" 2>&1)"; then
      LAST_QSTAT_OUTPUT="${output}"
      job_state="$(printf '%s\n' "${output}" | awk -F' = ' '/job_state = / {print $2; exit}')"
      exit_status="$(printf '%s\n' "${output}" | awk -F' = ' '/Exit_status = / {print $2; exit}')"
      case "${job_state}" in
        Q|R)
          sleep "${poll_interval}"
          ;;
        C|F)
          if [[ "${exit_status}" == "0" ]]; then
            return 0
          fi
          return "${WAIT_PBS_CANDIDATE_FAILURE}"
          ;;
        *)
          mark_transport_halt \
            "Unexpected PBS state ${job_state:-unknown} while reconciling ${label} job ${job_id}. qstat output: ${LAST_QSTAT_OUTPUT}"
          return "${WAIT_PBS_TRANSPORT_HALT}"
          ;;
      esac
    else
      sleep "${poll_interval}"
    fi
  done

  mark_transport_halt \
    "${label} job ${job_id} never reached a stable terminal PBS state during the post-halt reconciliation window."
  return "${WAIT_PBS_TRANSPORT_HALT}"
}

wait_for_status_file() {
  local job_id="$1"
  local remote_status_path="$2"
  local poll_interval="$3"
  local label="$4"
  local raw_status=""
  local parsed=""
  local job_state=""
  local exit_status=""
  local heartbeat=""
  local result_present=""
  local previous_heartbeat=""
  local stagnant_polls=0

  while true; do
    if ! raw_status="$(
      ssh_remote_resilient "read ${label} status file" "
        set -euo pipefail
        cat '${remote_status_path}'
      "
    )"; then
      return "${WAIT_PBS_TRANSPORT_HALT}"
    fi
    parsed="$(python3 - "${raw_status}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
print(payload.get("job_state") or "")
print("" if payload.get("exit_status") is None else payload.get("exit_status"))
print("" if payload.get("heartbeat") is None else payload.get("heartbeat"))
print("1" if payload.get("result_present") else "0")
PY
)"
    job_state="$(printf '%s\n' "${parsed}" | sed -n '1p')"
    exit_status="$(printf '%s\n' "${parsed}" | sed -n '2p')"
    heartbeat="$(printf '%s\n' "${parsed}" | sed -n '3p')"
    result_present="$(printf '%s\n' "${parsed}" | sed -n '4p')"

    if [[ -n "${heartbeat}" && "${heartbeat}" == "${previous_heartbeat}" ]]; then
      stagnant_polls=$((stagnant_polls + 1))
    else
      stagnant_polls=0
    fi
    previous_heartbeat="${heartbeat}"
    if (( stagnant_polls >= 2 )); then
      mark_transport_halt \
        "${label} watcher heartbeat stalled while waiting on ${job_id}"
      return "${WAIT_PBS_TRANSPORT_HALT}"
    fi

    case "${job_state}" in
      Q|R|"")
        ;;
      C|F)
        if [[ "${exit_status}" == "0" && "${result_present}" == "1" ]]; then
          return 0
        fi
        if [[ "${exit_status}" == "0" ]]; then
          :
        else
          return "${WAIT_PBS_CANDIDATE_FAILURE}"
        fi
        ;;
      *)
        mark_transport_halt \
          "Unexpected watcher state ${job_state} for ${label} job ${job_id}"
        return "${WAIT_PBS_TRANSPORT_HALT}"
        ;;
    esac

    sleep "${poll_interval}"
  done
}

wait_for_pbs() {
  local job_id="$1"
  local poll_interval="$2"
  local running_timeout_seconds="$3"
  local label="$4"
  local output=""
  local job_state=""
  local exit_status=""
  local now=0
  local running_elapsed_seconds=0
  local last_state=""
  local last_sample_at=0

  while true; do
    now="$(date +%s)"
    if (( last_sample_at > 0 )) && [[ "${last_state}" == "R" ]]; then
      running_elapsed_seconds=$((running_elapsed_seconds + now - last_sample_at))
    fi
    last_sample_at="${now}"

    if ! output="$(ssh_remote_resilient "poll ${label} job ${job_id}" "timeout 30 qstat -fx '${job_id}'")"; then
      return "${WAIT_PBS_TRANSPORT_HALT}"
    fi
    LAST_QSTAT_OUTPUT="${output}"
    job_state="$(printf '%s\n' "${output}" | awk -F' = ' '/job_state = / {print $2; exit}')"
    exit_status="$(printf '%s\n' "${output}" | awk -F' = ' '/Exit_status = / {print $2; exit}')"

    case "${job_state}" in
      Q)
        last_state="${job_state}"
        ;;
      R)
        if (( running_elapsed_seconds >= running_timeout_seconds )); then
          printf '%s [REMOTE-R2] %s job %s exceeded running cap %ss; canceling for retry\n' \
            "$(timestamp_utc)" "${label}" "${job_id}" "${running_timeout_seconds}" >&2
          kill_remote_job "${job_id}" "${label}"
          LAST_QSTAT_OUTPUT="${output}"
          return "${WAIT_PBS_RETRYABLE_TIMEOUT}"
        fi
        last_state="${job_state}"
        ;;
      C|F)
        if [[ "${exit_status}" == "0" ]]; then
          return 0
        fi
        return "${WAIT_PBS_CANDIDATE_FAILURE}"
        ;;
      *)
        mark_transport_halt \
          "Unexpected PBS state ${job_state:-unknown} for ${label} job ${job_id}. qstat output: ${LAST_QSTAT_OUTPUT}"
        return "${WAIT_PBS_TRANSPORT_HALT}"
        ;;
    esac

    sleep "${poll_interval}"
  done
}

resolve_remote_user() {
  local remote_user="${REMOTE_USER_OVERRIDE:-}"

  if [[ -z "${remote_user}" ]]; then
    if remote_user="$(ssh_remote_resilient "resolve remote user" 'printf %s "$USER"' 2>/dev/null)"; then
      remote_user="$(printf '%s' "${remote_user}" | tr -d '\r\n')"
    elif (( OUTCOME_IS_TRANSPORT_HALT == 1 )); then
      return 1
    fi
  fi
  if [[ -z "${remote_user}" ]]; then
    remote_user="$("${SSH_BIN}" -G "${EVE_REMOTE_HOST}" 2>/dev/null | awk '/^user / {print $2; exit}')"
  fi
  if [[ -z "${remote_user}" ]]; then
    mark_transport_halt "Failed to resolve remote user on ${EVE_REMOTE_HOST}"
    return 1
  fi

  printf '%s\n' "${remote_user}"
}
