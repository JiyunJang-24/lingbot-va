#!/usr/bin/env bash
# Fine-tuning launchers for Run A (regular only) and Run B (regular + rewind w/ latent_loss mask).
# Bypasses script/train.sh preflight (which expects single-dataset DATASET_DIR with data/ dir);
# we use multi-dataset symlink roots so we invoke run_va_posttrain.sh directly.
#
# Usage:
#   bash script/train_runs.sh a    # Run A
#   bash script/train_runs.sh b    # Run B
#
# Override knobs:
#   NUM_STEPS=2000 GRAD_ACCUM_STEPS=4 BATCH_SIZE=1 NGPU=2 \
#     bash script/train_runs.sh a

set -uo pipefail

run=${1:-}
if [[ "$run" != "a" && "$run" != "b" ]]; then
  echo "Usage: bash script/train_runs.sh [a|b]" >&2
  exit 1
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

case "$run" in
  a)
    DATASET_DIR_DEFAULT="$REPO_ROOT/datasets/_runs/run_a_regular"
    SAVE_ROOT_DEFAULT="$REPO_ROOT/train_out/run_a_regular"
    ;;
  b)
    DATASET_DIR_DEFAULT="$REPO_ROOT/datasets/_runs/run_b_combined"
    SAVE_ROOT_DEFAULT="$REPO_ROOT/train_out/run_b_combined"
    ;;
esac

export DATASET_DIR="${DATASET_DIR:-$DATASET_DIR_DEFAULT}"
export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints/lingbot-va-base}"
export OBS_CAM_KEYS="${OBS_CAM_KEYS:-observation.images.agentview_rgb,observation.images.eye_in_hand_rgb}"
export USED_ACTION_CHANNEL_IDS="${USED_ACTION_CHANNEL_IDS:-0,1,2,3,4,5,6}"
export SAVE_ROOT="${SAVE_ROOT:-$SAVE_ROOT_DEFAULT}"
export NGPU="${NGPU:-2}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export NUM_STEPS="${NUM_STEPS:-2000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
export LOAD_WORKER="${LOAD_WORKER:-0}"
export NUM_INIT_WORKER="${NUM_INIT_WORKER:-1}"
export ENABLE_WANDB="${ENABLE_WANDB:-1}"
export CONFIG_NAME="${CONFIG_NAME:-libero_train}"
export MASTER_PORT="${MASTER_PORT:-29501}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export PATH="$REPO_ROOT/.venv/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"

echo "[INFO] Run: $run"
echo "[INFO] DATASET_DIR=$DATASET_DIR"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] SAVE_ROOT=$SAVE_ROOT"
echo "[INFO] NGPU=$NGPU  BATCH_SIZE=$BATCH_SIZE  GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS  NUM_STEPS=$NUM_STEPS"
echo "[INFO] CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES  MASTER_PORT=$MASTER_PORT"
echo "[INFO] ENABLE_WANDB=$ENABLE_WANDB"

bash script/run_va_posttrain.sh
