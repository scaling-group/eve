#!/bin/bash
# ADAPT: your PBS project allocation
#PBS -P REPLACE_WITH_YOUR_PROJECT_CODE
#PBS -j oe
#PBS -k oed
#PBS -N eval_ctx
#PBS -l walltime=01:00:00
#PBS -l select=1:ngpus=1

# Context length generalization eval — offline, checkpoint-based.
# Single GPU, no training, no W&B logging.
#
# Usage:
#   qsub -v CKPT=/path/to/last.ckpt,NUM_EXAMPLES=20 scripts/eval_context_length.sh
#   qsub -v CKPT=/path/to/last.ckpt,NUM_EXAMPLE_LIST=1,2,3,4,5,6,7,8,9,10 scripts/eval_context_length.sh
#   qsub -v CKPT=/path/to/last.ckpt,NUM_EXAMPLE_LIST=1,2,3,4,5,6,7,8,9,10,BS=4 scripts/eval_context_length.sh
#   qsub -v CKPT=/path/to/last.ckpt,NUM_EXAMPLES=99,BS=2,MODEL=icon scripts/eval_context_length.sh

set -eo pipefail

REPO_ROOT="${PBS_O_WORKDIR}"
cd "${REPO_ROOT}"
set +u
source ~/.bashrc
set -euo pipefail

PYTHON_BIN="${EVE_REMOTE_PYTHON:-/scratch/${USER}/envs/venvs/icon-core/bin/python}"

CKPT="${CKPT:?Must set CKPT}"
NUM_EXAMPLES="${NUM_EXAMPLES:-${DEMO_NUM:-20}}"
NUM_EXAMPLE_LIST="${NUM_EXAMPLE_LIST:-${DEMO_NUMS:-}}"
BS="${BS:-32}"
MODEL="${MODEL:-}"
DATA_DIR="${DATA_DIR:-${EVE_REMOTE_DATA_DIR:-/scratch/${USER}/data}}"
OUT_DIR="${OUT_DIR:-/scratch/${USER}/logs/eval_context}"
SUMMARY_OUT="${SUMMARY_OUT:-}"

mkdir -p "${OUT_DIR}"

echo "== host =="; hostname
echo "== ckpt =="; echo "${CKPT}"
echo "== model =="; echo "${MODEL}"
echo "== num_examples =="; echo "${NUM_EXAMPLES}"
echo "== num_example_list =="; echo "${NUM_EXAMPLE_LIST}"
echo "== bs =="; echo "${BS}"
echo "== gpu =="; nvidia-smi -L || true

"${PYTHON_BIN}" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

if [[ -n "${NUM_EXAMPLE_LIST}" ]]; then
  cmd=(
    "${PYTHON_BIN}" -m src.eval_context_length_batch
    --ckpt "${CKPT}"
    --demo-nums "${NUM_EXAMPLE_LIST}"
    --batch-size "${BS}"
    --model "${MODEL}"
    --data-dir "${DATA_DIR}"
    --out-dir "${OUT_DIR}"
  )
  if [[ -n "${SUMMARY_OUT}" ]]; then
    cmd+=(--summary-out "${SUMMARY_OUT}")
  fi
else
  cmd=(
    "${PYTHON_BIN}" -m src.eval_context_length
    --ckpt "${CKPT}"
    --demo-num "${NUM_EXAMPLES}"
    --batch-size "${BS}"
    --model "${MODEL}"
    --data-dir "${DATA_DIR}"
    --out-dir "${OUT_DIR}"
  )
fi

"${cmd[@]}"

echo "Done (rc=$?)"
