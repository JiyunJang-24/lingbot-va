#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data1/local/lingbot-va}"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-lingbot-va}"
LIBERO_ROOT="${LIBERO_ROOT:-/data1/local/LIBERO}"
MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints/lingbot-va-base}"
DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/libero_table_cover_action_lang_smoke}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/libero_table_cover_action_lang_smoke_train}"
PREVIEW_VIDEO="${PREVIEW_VIDEO:-$REPO_ROOT/outputs/libero_table_cover_action_lang_smoke_preview.mp4}"

export PYTHONPATH="$LIBERO_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"
TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:-1}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-1}"
MAX_STEPS="${MAX_STEPS:-24}"
FPS="${FPS:-60}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"
SEGMENT_FRAMES="${SEGMENT_FRAMES:-5}"
SEGMENT_STRIDE="${SEGMENT_STRIDE:-1}"

rm -rf "$DATASET_DIR"

conda run -n "$CONDA_ENV" python tools/libero_dataset/collect_libero_lingbot_dataset.py \
  --libero-benchmark "$LIBERO_BENCHMARK" \
  --output-dir "$DATASET_DIR" \
  --task-start "$TASK_START" \
  --task-end "$TASK_END" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --max-steps "$MAX_STEPS" \
  --fps "$FPS" \
  --image-size "$IMAGE_SIZE" \
  --policy python \
  --policy-file "$REPO_ROOT/tools/libero_dataset/random_table_cover_policy.py"

conda run -n "$CONDA_ENV" python tools/libero_dataset/action_language_postprocess.py \
  --dataset-dir "$DATASET_DIR" \
  --segment-frames "$SEGMENT_FRAMES" \
  --stride "$SEGMENT_STRIDE"

conda run -n "$CONDA_ENV" python tools/libero_dataset/prepare_lingbot_dataset.py \
  --dataset-dir "$DATASET_DIR"

conda run -n "$CONDA_ENV" python tools/libero_dataset/extract_wan_latents_for_lingbot.py \
  --dataset-dir "$DATASET_DIR" \
  --model-path "$MODEL_PATH" \
  --device "${DEVICE:-cuda}" \
  --dtype "${DTYPE:-bfloat16}" \
  --overwrite

conda run -n "$CONDA_ENV" python tools/libero_dataset/prepare_lingbot_dataset.py \
  --dataset-dir "$DATASET_DIR" \
  --require-latents

conda run -n "$CONDA_ENV" python tools/libero_dataset/make_episode_preview.py \
  --dataset-dir "$DATASET_DIR" \
  --episode-index 0 \
  --output "$PREVIEW_VIDEO" \
  --fps 30

env \
  DATASET_DIR="$DATASET_DIR" \
  MODEL_PATH="$MODEL_PATH" \
  SAVE_ROOT="$SAVE_ROOT" \
  BATCH_SIZE="${BATCH_SIZE:-2}" \
  NUM_STEPS="${NUM_STEPS:-1}" \
  GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}" \
  SAVE_INTERVAL="${SAVE_INTERVAL:-1}" \
  NGPU="${NGPU:-1}" \
  CONFIG_NAME="libero_action_lang_smoke_train" \
  bash script/train_libero_lingbot_custom.sh

echo "[DONE] Dataset: $DATASET_DIR"
echo "[DONE] Preview: $PREVIEW_VIDEO"
echo "[DONE] Finetune output: $SAVE_ROOT"
