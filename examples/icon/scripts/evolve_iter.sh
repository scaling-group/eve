#!/bin/bash
# ADAPT: your PBS project allocation
#PBS -P REPLACE_WITH_YOUR_PROJECT_CODE
#PBS -j oe
#PBS -k oed
#PBS -N eve_icon_evolve
#PBS -l walltime=01:00:00
#PBS -l select=1:ngpus=2

# Vanilla ICON evolve iter — 10000 steps, 2x GPU DDP.

set -eo pipefail

REPO_ROOT="${PBS_O_WORKDIR}"
cd "${REPO_ROOT}"
set +u
source ~/.bashrc
set -euo pipefail

export PROJECT_ROOT="${REPO_ROOT}"
if [[ -f "${REPO_ROOT}/scripts/wandb_env.sh" ]]; then
  source "${REPO_ROOT}/scripts/wandb_env.sh"
fi
export WANDB_PROJECT="${WANDB_PROJECT:-icon-pe}"
export WANDB_GROUP="${WANDB_GROUP:-icon-pe-evolve-iter}"
export WANDB_JOB_TYPE="${WANDB_JOB_TYPE:-iter}"
export WANDB_DIR="${WANDB_DIR:-/scratch/${USER}/cache/wandb}"
mkdir -p "${WANDB_DIR}"

PYTHON_BIN="${EVE_REMOTE_PYTHON:-/scratch/${USER}/envs/venvs/icon-core/bin/python}"
if [[ -x "${PYTHON_BIN}" ]]; then
  RUNNER=("${PYTHON_BIN}")
else
  RUNNER=(uv run python)
fi

BS="${BS:-32}"
LR="${LR:-5e-4}"
MAX_STEPS="${MAX_STEPS:-10000}"
VAL_EVERY="${VAL_EVERY:-2000}"
SAVE_EVERY="${SAVE_EVERY:-2000}"
TASK_TAG="${TASK_TAG:-$(date -u +%Y%m%dT%H%M%SZ)_$$}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DATA_DIR="${DATA_DIR:-${EVE_REMOTE_DATA_DIR:-/scratch/${USER}/data}}"
LOG_DIR="${LOG_DIR:-/scratch/${USER}/logs}"

TASK_NAME="eve_icon_evolve_iter_bs${BS}x2_lr${LR}_s${MAX_STEPS}_${TASK_TAG}"

echo "== host =="; hostname
echo "== task =="; echo "${TASK_NAME}"
echo "== gpu =="; nvidia-smi -L || true

"${RUNNER[@]}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

"${RUNNER[@]}" src/train.py \
  --config-name=train_icon \
  task_name="${TASK_NAME}" \
  trainer=ddp trainer.devices=2 \
  +trainer.accumulate_grad_batches=1 \
  trainer.max_steps="${MAX_STEPS}" \
  trainer.val_check_interval="${VAL_EVERY}" \
  callbacks.save_ckpt.every_n_train_steps="${SAVE_EVERY}" \
  opt.optimizer.lr="${LR}" \
  opt.scheduler.max_iters="${MAX_STEPS}" \
  data.batch_size_per_device="${BS}" \
  data.num_workers="${NUM_WORKERS}" \
  paths.data_dir="${DATA_DIR}" \
  paths.log_dir="${LOG_DIR}" \
  logger=wandb \
  log_project="${WANDB_PROJECT}" \
  +logger.wandb.entity="${WANDB_ENTITY}" \
  logger.wandb.group="${WANDB_GROUP}" \
  logger.wandb.job_type="${WANDB_JOB_TYPE}" \
  tags="[icon,weno,evolve,pe]" \
  experiment=evolve_base

echo "Done: ${TASK_NAME} (rc=$?)"
