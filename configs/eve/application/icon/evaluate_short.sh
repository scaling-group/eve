#!/usr/bin/env bash

set -euo pipefail

export MAX_STEPS="${MAX_STEPS:-50}"
export VAL_EVERY="${VAL_EVERY:-50}"
export SAVE_EVERY="${SAVE_EVERY:-50}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/evaluate.sh"
