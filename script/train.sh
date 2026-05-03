#!/usr/bin/env bash
set -euo pipefail

# Fine-tune LingBot-VA base on the local LIBERO-long LeRobot dataset.
#
# Default goal:
#   model   : /root/Desktop/workspace/lingbot-va/checkpoints/lingbot-va-base
#   dataset : /root/Desktop/workspace/lingbot-va/datasets/libero-long-lerobot
#   batch   : 1 per GPU
#
# Run:
#   bash script/train.sh
#
# Try batch size 2:
#   BATCH_SIZE=2 bash script/train.sh
#
# Useful short smoke test before a long run:
#   NUM_STEPS=1 GRAD_ACCUM_STEPS=1 SAVE_INTERVAL=999999 bash script/train.sh
#
# Common knobs:
#   NGPU=1 or NGPU=2              number of GPUs/processes
#   BATCH_SIZE=1 or 2             per-GPU batch size
#   GRAD_ACCUM_STEPS=10           effective batch multiplier
#   NUM_STEPS=5000                optimizer steps
#   SAVE_INTERVAL=200             checkpoint interval
#   SAVE_ROOT=./train_out/libero  output root; checkpoints go under $SAVE_ROOT/checkpoints
#
# If BATCH_SIZE=2 runs out of memory, go back to BATCH_SIZE=1 and increase
# GRAD_ACCUM_STEPS instead. For example, BATCH_SIZE=1 GRAD_ACCUM_STEPS=20 gives
# the same per-GPU effective batch as BATCH_SIZE=2 GRAD_ACCUM_STEPS=10.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints/lingbot-va-base}"
export DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/libero-long-lerobot}"
export CONFIG_NAME="${CONFIG_NAME:-libero_train}"
export NGPU="${NGPU:-2}"
export BATCH_SIZE="${BATCH_SIZE:-2}"
export GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-5}"
export NUM_STEPS="${NUM_STEPS:-5000}"
export SAVE_INTERVAL="${SAVE_INTERVAL:-200}"
export SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/train_out/libero-long}"
export LOAD_WORKER="${LOAD_WORKER:-0}"
export NUM_INIT_WORKER="${NUM_INIT_WORKER:-1}"
export ENABLE_WANDB="${ENABLE_WANDB:-0}"
export MASTER_PORT="${MASTER_PORT:-29501}"

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
    export PATH="$REPO_ROOT/.venv/bin:$PATH"
  else
    PYTHON_BIN="python"
  fi
fi
echo DATASET_DIR="$DATASET_DIR"
if [[ ! -d "$MODEL_PATH/transformer" ]]; then
  echo "[ERROR] MODEL_PATH does not look like a LingBot-VA checkpoint: $MODEL_PATH" >&2
  exit 1
fi

if [[ ! -f "$DATASET_DIR/meta/info.json" ]]; then
  echo "[ERROR] DATASET_DIR does not look like a LeRobot dataset: $DATASET_DIR" >&2
  exit 1
fi

# Make sure meta/action_norm_quantiles.json exists and latent files match
# meta/episodes.jsonl. The train config reads these quantiles automatically.
"$PYTHON_BIN" tools/libero_dataset/prepare_lingbot_dataset.py \
  --dataset-dir "$DATASET_DIR" \
  --require-latents

# The downloaded dataset already contains per-segment text_emb inside latents,
# but training also needs one dataset-level empty_emb.pt for CFG dropout.
# Generate the exact empty-prompt embedding from the base text encoder when it
# is missing. This is much cheaper than re-extracting all video latents.
if [[ ! -f "$DATASET_DIR/empty_emb.pt" ]]; then
  EMPTY_EMB_DEVICE="${EMPTY_EMB_DEVICE:-cuda}"
  if ! "$PYTHON_BIN" - "$MODEL_PATH" "$DATASET_DIR" "$EMPTY_EMB_DEVICE" <<'PY'
import sys
from pathlib import Path

import torch

model_path = Path(sys.argv[1]).resolve()
dataset_dir = Path(sys.argv[2]).resolve()
device = sys.argv[3]

if device == "cuda" and not torch.cuda.is_available():
    device = "cpu"

from tools.libero_dataset.extract_wan_latents_for_lingbot import encode_text
from wan_va.modules import load_text_encoder, load_tokenizer

dtype = torch.bfloat16 if device == "cuda" else torch.float32
tokenizer = load_tokenizer(model_path / "tokenizer")
text_encoder = load_text_encoder(model_path / "text_encoder", torch_dtype=dtype, torch_device=device)
empty_emb = encode_text(
    tokenizer,
    text_encoder,
    "",
    device="cpu",
    dtype=torch.bfloat16,
    max_sequence_length=512,
)
torch.save(empty_emb, dataset_dir / "empty_emb.pt")
print(f"[DONE] Wrote {dataset_dir / 'empty_emb.pt'} with shape {tuple(empty_emb.shape)}")
PY
  then
    echo "[ERROR] Failed to create $DATASET_DIR/empty_emb.pt" >&2
    echo "        Try: EMPTY_EMB_DEVICE=cpu bash script/train.sh" >&2
    exit 1
  fi
fi

echo "[INFO] Starting fine-tuning"
echo "[INFO] MODEL_PATH=$MODEL_PATH"
echo "[INFO] DATASET_DIR=$DATASET_DIR"
echo "[INFO] NGPU=$NGPU BATCH_SIZE=$BATCH_SIZE GRAD_ACCUM_STEPS=$GRAD_ACCUM_STEPS NUM_STEPS=$NUM_STEPS"
echo "[INFO] SAVE_ROOT=$SAVE_ROOT"

bash script/run_va_posttrain.sh
