#!/usr/bin/env bash
# Run B: Fine-tune lingbot-va-base on regular + rewind LIBERO spatial-object-location datasets.
# Rewind samples use ONLY action_loss (latent_loss masked to 0 via per-sample use_latent_loss flag).
# See wan_va/dataset/lerobot_latent_dataset.py — is_rewind detected from path containing "rewind".
#
# Default GPU set: 2,3   (override via CUDA_VISIBLE_DEVICES env)
# Total compute: ~28h on 2x H100 at 5000 steps × GA=4 (effective batch 8)
#
# Usage:
#   bash libero_posttrain_regular_rewind.sh
#
# Override knobs:
#   NUM_STEPS=2000 GRAD_ACCUM_STEPS=8 BATCH_SIZE=1 CUDA_VISIBLE_DEVICES=2,3 \
#     bash libero_posttrain_regular_rewind.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

# --- Preflight: attn_mode must be "flex" for training ---
ATTN_MODE=$(python -c "import json; print(json.load(open('checkpoints/lingbot-va-base/transformer/config.json'))['attn_mode'])" 2>/dev/null || echo "?")
if [[ "$ATTN_MODE" != "flex" ]]; then
  echo "[ERROR] checkpoints/lingbot-va-base/transformer/config.json attn_mode=$ATTN_MODE, must be 'flex' for training." >&2
  echo "        Fix with: python -c \"import json; p='checkpoints/lingbot-va-base/transformer/config.json'; c=json.load(open(p)); c['attn_mode']='flex'; json.dump(c, open(p,'w'), indent=2)\"" >&2
  exit 1
fi

# --- Training config (override via env) ---
export DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/_runs/run_b_combined}"
export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints/lingbot-va-base}"
export OBS_CAM_KEYS="${OBS_CAM_KEYS:-observation.images.agentview_rgb,observation.images.eye_in_hand_rgb}"
export USED_ACTION_CHANNEL_IDS="${USED_ACTION_CHANNEL_IDS:-0,1,2,3,4,5,6}"
export SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/train_out/run_b_combined}"
export NGPU="${NGPU:-2}"
export BATCH_SIZE="${BATCH_SIZE:-1}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-4}"
export NUM_STEPS="${NUM_STEPS:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
export LOAD_WORKER="${LOAD_WORKER:-0}"
export NUM_INIT_WORKER="${NUM_INIT_WORKER:-1}"
export ENABLE_WANDB="${ENABLE_WANDB:-1}"
export WANDB_NAME="${WANDB_NAME:-lingbot-va-libero-spatial-rewind-actionloss}"
export CONFIG_NAME="${CONFIG_NAME:-libero_train}"
export MASTER_PORT="${MASTER_PORT:-29502}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
export PATH="$REPO_ROOT/.venv/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-$REPO_ROOT}"

mkdir -p "$SAVE_ROOT"
LOG_FILE="${LOG_FILE:-$SAVE_ROOT/train.log}"

echo "=============================================================="
echo "  Run B (regular + rewind, rewind uses action_loss only)"
echo "=============================================================="
echo "  DATASET_DIR       : $DATASET_DIR"
echo "  MODEL_PATH        : $MODEL_PATH"
echo "  SAVE_ROOT         : $SAVE_ROOT"
echo "  NGPU              : $NGPU   (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  BATCH_SIZE        : $BATCH_SIZE"
echo "  GRAD_ACCUM_STEPS  : $GRAD_ACCUM_STEPS   (effective batch = $((BATCH_SIZE * NGPU * GRAD_ACCUM_STEPS)))"
echo "  NUM_STEPS         : $NUM_STEPS"
echo "  SAVE_INTERVAL     : $SAVE_INTERVAL"
echo "  ENABLE_WANDB      : $ENABLE_WANDB   (run name: $WANDB_NAME)"
echo "  MASTER_PORT       : $MASTER_PORT"
echo "  LOG_FILE          : $LOG_FILE"
echo "  NOTE              : Rewind samples (dataset path contains 'rewind') → latent_loss=0, action_loss only"
echo "=============================================================="

bash script/run_va_posttrain.sh 2>&1 | tee "$LOG_FILE"
