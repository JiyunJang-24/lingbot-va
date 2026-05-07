#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$REPO_ROOT/checkpoints/lingbot-va-base}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-$REPO_ROOT/train_out/libero_manual_action_only_w_lora/checkpoints/checkpoint_step_4600}"
INFER_MODEL_PATH="${INFER_MODEL_PATH:-$REPO_ROOT/train_out/libero_manual_action_only_w_lora/inference_model_step_4600}"

cd "$REPO_ROOT"

if [[ ! -x "$REPO_ROOT/.venv/bin/python" ]]; then
  echo "[ERROR] venv python not found at $REPO_ROOT/.venv/bin/python" >&2
  exit 1
fi

if [[ ! -d "$BASE_MODEL_PATH/vae" || ! -d "$BASE_MODEL_PATH/text_encoder" || ! -d "$BASE_MODEL_PATH/tokenizer" ]]; then
  echo "[ERROR] BASE_MODEL_PATH does not contain vae/, text_encoder/, and tokenizer/: $BASE_MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH/transformer/diffusion_pytorch_model.safetensors" ]]; then
  echo "[ERROR] checkpoint transformer weights not found under: $CHECKPOINT_PATH" >&2
  exit 1
fi

mkdir -p "$INFER_MODEL_PATH"
ln -sfn "$BASE_MODEL_PATH/vae" "$INFER_MODEL_PATH/vae"
ln -sfn "$BASE_MODEL_PATH/text_encoder" "$INFER_MODEL_PATH/text_encoder"
ln -sfn "$BASE_MODEL_PATH/tokenizer" "$INFER_MODEL_PATH/tokenizer"
ln -sfn "$CHECKPOINT_PATH/transformer" "$INFER_MODEL_PATH/transformer"

export PATH="$REPO_ROOT/.venv/bin:$PATH"
export PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
export MODEL_PATH="$INFER_MODEL_PATH"
export MASTER_PORT="${MASTER_PORT:-29601}"

echo "[INFO] Running i2va_libero_overall.sh with venv python: $PYTHON_BIN"
echo "[INFO] Using trained checkpoint transformer: $CHECKPOINT_PATH/transformer"
echo "[INFO] Inference model root: $MODEL_PATH"
echo "[INFO] MASTER_PORT=$MASTER_PORT"

exec bash "$REPO_ROOT/i2va_libero_overall.sh" "$@"
