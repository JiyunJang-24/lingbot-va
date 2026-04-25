#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ "${CONDA_DEFAULT_ENV:-}" != "lingbot-va" ]]; then
  if command -v conda >/dev/null 2>&1; then
    exec conda run -n lingbot-va bash "$PWD/eval.sh" "$@"
  fi

  echo "Please activate the lingbot-va conda environment first." >&2
  exit 1
fi

# export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
MODEL_PATH=${MODEL_PATH:-"$PWD/checkpoints/lingbot-va-posttrain-robotwin"}

NGPU=1 CONFIG_NAME='robotwin_i2av' bash script/run_launch_va_server_sync.sh \
  --model_path "$MODEL_PATH" \
  "$@"
