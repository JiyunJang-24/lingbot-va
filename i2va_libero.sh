#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lingbot-va" ]]; then
  if command -v conda >/dev/null 2>&1; then
    exec conda run -n lingbot-va bash "$PWD/i2va_libero.sh" "$@"
  fi

  echo "Please activate the lingbot-va conda environment first." >&2
  exit 1
fi

MODEL_PATH=${MODEL_PATH:-"$PWD/checkpoints/lingbot-va-base"}
SAVE_ROOT=${SAVE_ROOT:-"train_out_libero_base"}

has_model_path=0
has_save_root=0
for arg in "$@"; do
  if [[ "$arg" == "--model_path" || "$arg" == --model_path=* ]]; then
    has_model_path=1
  fi
  if [[ "$arg" == "--save_root" || "$arg" == --save_root=* ]]; then
    has_save_root=1
  fi
done

extra_args=("$@")
if [[ $has_model_path -eq 0 ]]; then
  extra_args=(--model_path "$MODEL_PATH" "${extra_args[@]}")
fi
if [[ $has_save_root -eq 0 ]]; then
  extra_args=(--save_root "$SAVE_ROOT" "${extra_args[@]}")
fi

NGPU=1 CONFIG_NAME='libero_i2av' bash script/run_launch_va_server_sync.sh \
  "${extra_args[@]}"
