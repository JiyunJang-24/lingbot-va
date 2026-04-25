#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data1/local/lingbot-va}"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-lingbot-va}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/libero_custom_lingbot}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints/lingbot-va-base}"
DEVICE="${DEVICE:-cuda}"
DTYPE="${DTYPE:-bfloat16}"
OVERWRITE="${OVERWRITE:-0}"

args=(
  --dataset-dir "$DATASET_DIR"
  --model-path "$MODEL_PATH"
  --device "$DEVICE"
  --dtype "$DTYPE"
)

if [[ "$OVERWRITE" == "1" ]]; then
  args+=(--overwrite)
fi

conda run -n "$CONDA_ENV" python tools/libero_dataset/extract_wan_latents_for_lingbot.py "${args[@]}"

conda run -n "$CONDA_ENV" python tools/libero_dataset/prepare_lingbot_dataset.py \
  --dataset-dir "$DATASET_DIR" \
  --require-latents

echo "[DONE] Dataset is ready for LingBot-VA training: $DATASET_DIR"
