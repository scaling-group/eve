#!/usr/bin/env bash

set -euo pipefail

workspace_root="${EVE_WORKSPACE_ROOT:-$PWD}"
output_root="${EVE_OUTPUT_ROOT:-$workspace_root/output}"
eval_log_root="${EVE_EVAL_LOG_ROOT:-$workspace_root/logs/evaluate}"

EVE_REMOTE_HOST="${EVE_REMOTE_HOST:-remote-cluster}"
REMOTE_REPO="${REMOTE_REPO:-~/repos/icon-core}"
REMOTE_BRANCH="${REMOTE_BRANCH:-main}"
MAX_STEPS="${MAX_STEPS:-2000}"
VAL_EVERY="${VAL_EVERY:-2000}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
TASK_NAME_PREFIX="${TASK_NAME_PREFIX:-eve_icon_evolve_iter}"
BS="${BS:-32}"
LR="${LR:-5e-4}"
NUM_WORKERS="${NUM_WORKERS:-32}"
DATA_DIR="${DATA_DIR:-${EVE_REMOTE_DATA_DIR:-}}"
TRAIN_RUNNING_TIMEOUT_SECONDS="${TRAIN_RUNNING_TIMEOUT_SECONDS:-10800}"
EVAL_RUNNING_TIMEOUT_SECONDS="${EVAL_RUNNING_TIMEOUT_SECONDS:-3600}"
DEMO_NUMS="${DEMO_NUMS:-1 2 3 4 5 6 7 8 9 10}"
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
REMOTE_WORKTREE=""
REMOTE_REPO_PATH=""
LOGS_DIR="${eval_log_root}"
SCORE_FILE="${LOGS_DIR}/score.txt"
SCORE_YAML_FILE="${LOGS_DIR}/score.yaml"
SUMMARY_FILE="${LOGS_DIR}/eval_summary.json"
ERROR_FILE="${LOGS_DIR}/error.txt"
FAILURE_SCORE="${FAILURE_SCORE:--10.0}"
OUTCOME_IS_CANDIDATE_FAILURE=0
OUTCOME_IS_TRANSPORT_HALT=0
ATTEMPT_ID="${ATTEMPT_ID_OVERRIDE:-eval_$(date -u +%Y%m%dT%H%M%SZ)_$$_${RANDOM}}"
RESULT_JSON_PATH="${LOGS_DIR}/result_${ATTEMPT_ID}.json"
ORCHESTRATOR_STDOUT_PATH="${LOGS_DIR}/orchestrator_${ATTEMPT_ID}.out.log"
ORCHESTRATOR_STDERR_PATH="${LOGS_DIR}/orchestrator_${ATTEMPT_ID}.err.log"
STATUS_JSON_PATH="${LOGS_DIR}/status_${ATTEMPT_ID}.json"
WATCHER_LOG_PATH="${LOGS_DIR}/watcher_${ATTEMPT_ID}.log"
META_JSON_PATH=""
STAGING_DIR=""
REMOTE_USER=""
STAGE_BUNDLE_DIR=""
STAGE_ATTEMPT_ROOT=""
STATUS_POLL_INTERVAL_SECONDS="${STATUS_POLL_INTERVAL_SECONDS:-60}"

mark_candidate_failure() {
  OUTCOME_IS_CANDIDATE_FAILURE=1
  OUTCOME_IS_TRANSPORT_HALT=0
  ERROR_MESSAGE="$1"
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${script_dir}/helpers.sh"
init_remote_transport

write_score_yaml() {
  local payload_json="$1"

  python3 - "${SCORE_YAML_FILE}" "${payload_json}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

score_path = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
score_path.write_text(
    yaml.safe_dump(payload, sort_keys=False),
    encoding="utf-8",
)
PY
}

write_failure_outputs() {
  local message="$1"
  local failure_payload_json=""

  mkdir -p "${LOGS_DIR}"
  printf '%s\n' "${FAILURE_SCORE}" >"${SCORE_FILE}"
  failure_payload_json="$(python3 - "${FAILURE_SCORE}" "${message}" <<'PY'
import json
import sys

print(
    json.dumps(
        {
            "score": float(sys.argv[1]),
            "summary": "evaluation failed",
            "score_metric": "evaluation_failed",
            "score_components": {
                "reason": sys.argv[2],
            },
        }
    )
)
PY
)"
  write_score_yaml "${failure_payload_json}"
  printf '%s\n' "${message}" >"${ERROR_FILE}"
}

cleanup_local_temp() {
  if [[ -n "${META_JSON_PATH:-}" ]]; then
    rm -f "${META_JSON_PATH}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${STAGE_BUNDLE_DIR:-}" ]]; then
    rm -rf "${STAGE_BUNDLE_DIR}" >/dev/null 2>&1 || true
  fi
}

cleanup() {
  local exit_code="$1"
  local final_exit_code="${exit_code}"

  set +e

  if [[ -n "${STAGING_DIR:-}" ]]; then
    ssh_remote "rm -rf '${STAGING_DIR}'" >/dev/null 2>&1 || true
  fi

  close_ssh_master
  cleanup_local_temp

  if (( exit_code != 0 )) && (( OUTCOME_IS_CANDIDATE_FAILURE == 1 )); then
    if write_failure_outputs "${ERROR_MESSAGE:-evaluate.sh failed with exit code ${exit_code}}"; then
      final_exit_code=0
    fi
  fi

  return "${final_exit_code}"
}

trap 'rc=$?; trap - EXIT; cleanup "${rc}"; final_rc=$?; exit "${final_rc}"' EXIT

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
  STAGE_BUNDLE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/icon-eval-stage-${ATTEMPT_ID}.XXXXXX")"
  STAGE_ATTEMPT_ROOT="${STAGE_BUNDLE_DIR}/eve_staging/${ATTEMPT_ID}"
  python3 - \
    "${STAGE_ATTEMPT_ROOT}" \
    "${output_root}" \
    "${META_JSON_PATH}" \
    "${script_dir}/evaluate_orchestrator.sh" \
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
  META_JSON_PATH="$(mktemp "${TMPDIR:-/tmp}/icon-eval-meta-${ATTEMPT_ID}.XXXXXX")"
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
    "${EVAL_RUNNING_TIMEOUT_SECONDS}" \
    "${DEMO_NUMS}" \
    "${editable_files_json}" \
    "${editable_folders_json}" <<'PY'
import json
import sys
from pathlib import Path

payload = {
    "attempt_id": sys.argv[2],
    "created_at": sys.argv[3],
    "task": "evaluate",
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
    "eval_timeout_seconds": int(sys.argv[17]),
    "demo_nums": sys.argv[18],
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
    "RESULT_PBS_JOB_ID": payload.get("pbs_job_id") or "",
}
for key, value in fields.items():
    print(f"{key}={shlex.quote(str(value))}")
PY
}

write_success_outputs_from_result() {
  python3 - "${RESULT_JSON_PATH}" "${SCORE_FILE}" "${SCORE_YAML_FILE}" "${SUMMARY_FILE}" <<'PY'
import json
import sys
from pathlib import Path

import yaml

result = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
score_components = dict(result.get("score_components") or {})
score_metric = result.get("score_metric") or score_components.get("score_metric") or "mean_d1_d10"

if "optimized_score" in score_components:
    score = float(score_components["optimized_score"])
else:
    mean_d1_d10 = score_components.get("mean_d1_d10")
    if mean_d1_d10 is None:
        raise SystemExit("result.json missing optimized_score and mean_d1_d10")
    score = -float(mean_d1_d10)
    score_components["optimized_score"] = score

score_components.setdefault("score_metric", score_metric)
parts = [score_metric]
mean_d1_d10 = score_components.get("mean_d1_d10")
mean_d1_d4 = score_components.get("mean_d1_d4")
d10 = score_components.get("d10")
if mean_d1_d10 is not None:
    parts.append(f"mean_d1_d10={float(mean_d1_d10):.6f}")
if mean_d1_d4 is not None:
    parts.append(f"mean_d1_d4={float(mean_d1_d4):.6f}")
if d10 is not None:
    parts.append(f"d10={float(d10):.6f}")

score_payload = {
    "score": score,
    "summary": "; ".join(parts),
    "score_metric": score_metric,
    "score_components": score_components,
}
summary_payload = {
    "ckpt": result.get("checkpoint_path"),
    "score_metric": score_metric,
    "score_components": score_components,
    "score": score,
    "quest_qoi_v_by_demo": score_components.get("d1_to_d10"),
    "demo_nums": score_components.get("score_window_demo_nums"),
    "mean_quest_qoi_v": score_components.get("mean_d1_d10"),
    "results": [],
}

Path(sys.argv[2]).write_text(f"{score}\n", encoding="utf-8")
Path(sys.argv[3]).write_text(yaml.safe_dump(score_payload, sort_keys=False), encoding="utf-8")
Path(sys.argv[4]).write_text(json.dumps(summary_payload, indent=2, sort_keys=False) + "\n", encoding="utf-8")
PY
}

mkdir -p "${LOGS_DIR}"
rm -f "${SUMMARY_FILE}" "${ERROR_FILE}"

for relative_path in "${EDITABLE_FOLDERS[@]:-}"; do
  [[ -n "${relative_path}" ]] || continue
  if [[ ! -d "${output_root}/${relative_path}" ]]; then
    mark_transport_halt "Missing editable folder: ${output_root}/${relative_path}"
    exit 1
  fi
done

for relative_path in "${EDITABLE_FILES[@]}"; do
  if [[ ! -f "${output_root}/${relative_path}" ]]; then
    mark_transport_halt "Missing editable file: ${output_root}/${relative_path}"
    exit 1
  fi
done

if ! REMOTE_USER="$(resolve_remote_user)"; then
  mark_transport_halt "Failed to resolve remote user on ${EVE_REMOTE_HOST}"
  exit 1
fi
REMOTE_USER="$(printf '%s' "${REMOTE_USER}" | tr -d '\r\n')"
DATA_DIR="${DATA_DIR:-/scratch/${REMOTE_USER}/data}"
STAGING_DIR="/scratch/${REMOTE_USER}/eve_staging/${ATTEMPT_ID}"
REMOTE_REPO_PATH="${REMOTE_REPO/#\~/\$HOME}"
REMOTE_WORKTREE="${STAGING_DIR}"

write_meta_json
prepare_stage_bundle

if ! rsync_resilient \
  "stage evaluate bundle" \
  -a "${RSYNC_COMMON_OPTIONS[@]}" -e "${SSH_RSYNC_CMD}" \
  "${STAGE_BUNDLE_DIR}/" \
  "${EVE_REMOTE_HOST}:/scratch/${REMOTE_USER}/"; then
  mark_transport_halt "Failed to stage evaluate bundle into /scratch/${REMOTE_USER}"
  exit 1
fi

JOB_ID=""
if ! JOB_ID="$(submit_watched_pbs_job "launch_watched_job.sh" "evaluate_orchestrator.sh" "evaluate orchestrator")"; then
  mark_transport_halt "Failed to submit evaluate orchestrator PBS job"
  exit 1
fi

pbs_wait_rc=0
if wait_for_status_file "${JOB_ID}" "${STAGING_DIR}/status.json" "${STATUS_POLL_INTERVAL_SECONDS}" "evaluate orchestrator"; then
  pbs_wait_rc=0
else
  pbs_wait_rc=$?
fi

if [[ "${pbs_wait_rc}" -eq "${WAIT_PBS_TRANSPORT_HALT}" ]]; then
  if reconcile_pbs_after_transport_halt \
    "${JOB_ID}" \
    "${TRANSPORT_RECONCILE_POLL_INTERVAL_SECONDS}" \
    "${TRANSPORT_RECONCILE_GRACE_SECONDS}" \
    "evaluate orchestrator"; then
    pbs_wait_rc=0
  else
    pbs_wait_rc=$?
  fi
fi

FETCH_DIR="$(mktemp -d "${TMPDIR:-/tmp}/icon-eval-fetch-${ATTEMPT_ID}.XXXXXX")"
if fetch_remote_artifacts_batch \
  "fetch eval artifacts batch" \
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
  emit_pbs_fetch_diagnostics "${JOB_ID}" "evaluate_orchestrator" || true
  case "${pbs_wait_rc}" in
    "${WAIT_PBS_CANDIDATE_FAILURE}")
      mark_transport_halt "Orchestrator PBS job ${JOB_ID} exited non-zero before result.json could be fetched"
      ;;
    "${WAIT_PBS_RETRYABLE_TIMEOUT}")
      mark_transport_halt "Timed out while polling orchestrator PBS job ${JOB_ID} before result.json was available"
      ;;
    "${WAIT_PBS_TRANSPORT_HALT}")
      mark_transport_halt "Transport halted while polling or fetching result.json for orchestrator PBS job ${JOB_ID}"
      ;;
    *)
      mark_transport_halt "Failed to fetch ${STAGING_DIR}/result.json"
      ;;
  esac
  exit 1
fi

result_env=""
if ! result_env="$(parse_result_env)"; then
  mark_transport_halt "Failed to parse ${RESULT_JSON_PATH}"
  exit 1
fi
eval "${result_env}"

case "${RESULT_STATUS}" in
  pass)
    if ! write_success_outputs_from_result; then
      mark_transport_halt "Failed to materialize local score artifacts from ${RESULT_JSON_PATH}"
      exit 1
    fi
    rm -f "${ERROR_FILE}"
    ;;
  candidate_failure)
    # Post-v2 policy: treat every evaluate-stage candidate_failure as an
    # infra halt rather than a -10 score. Check-runner already validated
    # the candidate code; an evaluate-stage failure after that point is
    # overwhelmingly a server-side tail event, and writing -10 into the
    # population corrupts downstream Phase 2 sampling. The outer Codex
    # supervisor decides whether to resume from the last committed
    # iteration. Genuine code-side failures are still caught by
    # check.sh and do write -10 there.
    ERROR_MESSAGE="${RESULT_ERROR_MESSAGE}; result_json=${RESULT_JSON_PATH}"
    if [[ -f "${ORCHESTRATOR_STDOUT_PATH}" ]]; then
      ERROR_MESSAGE="${ERROR_MESSAGE}; orchestrator_stdout=${ORCHESTRATOR_STDOUT_PATH}"
    fi
    if [[ -f "${ORCHESTRATOR_STDERR_PATH}" ]]; then
      ERROR_MESSAGE="${ERROR_MESSAGE}; orchestrator_stderr=${ORCHESTRATOR_STDERR_PATH}"
    fi
    mark_transport_halt "${ERROR_MESSAGE}"
    exit 1
    ;;
  infra_failure|unclear|"")
    mark_transport_halt "${RESULT_ERROR_MESSAGE:-Orchestrator did not report a classified result}; result_json=${RESULT_JSON_PATH}"
    exit 1
    ;;
  *)
    mark_transport_halt "Unknown result status ${RESULT_STATUS}; result_json=${RESULT_JSON_PATH}"
    exit 1
    ;;
esac
