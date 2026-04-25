#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data1/local/lingbot-va}"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-lingbot-va}"
LIBERO_ROOT="${LIBERO_ROOT:-/data1/local/LIBERO}"
export PYTHONPATH="$LIBERO_ROOT:$REPO_ROOT:${PYTHONPATH:-}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/libero_custom_lingbot}"
LIBERO_BENCHMARK="${LIBERO_BENCHMARK:-libero_10}"
TASK_START="${TASK_START:-0}"
TASK_END="${TASK_END:-1}"
EPISODES_PER_TASK="${EPISODES_PER_TASK:-5}"
MAX_STEPS="${MAX_STEPS:-800}"
FPS="${FPS:-60}"
IMAGE_SIZE="${IMAGE_SIZE:-128}"

# POLICY can be: zero, random, python, websocket.
# For POLICY=python, set POLICY_FILE to a module that defines make_policy(), policy, or act(context).
POLICY="${POLICY:-python}"
POLICY_FILE="${POLICY_FILE:-$REPO_ROOT/tools/libero_dataset/example_policy.py}"
PORT="${PORT:-23908}"

conda run -n "$CONDA_ENV" python tools/libero_dataset/collect_libero_lingbot_dataset.py \
  --libero-benchmark "$LIBERO_BENCHMARK" \
  --output-dir "$DATASET_DIR" \
  --task-start "$TASK_START" \
  --task-end "$TASK_END" \
  --episodes-per-task "$EPISODES_PER_TASK" \
  --max-steps "$MAX_STEPS" \
  --fps "$FPS" \
  --image-size "$IMAGE_SIZE" \
  --policy "$POLICY" \
  --policy-file "$POLICY_FILE" \
  --port "$PORT"

conda run -n "$CONDA_ENV" python tools/libero_dataset/prepare_lingbot_dataset.py \
  --dataset-dir "$DATASET_DIR"

echo "[DONE] Collected dataset at: $DATASET_DIR"
echo "[NEXT] Extract latents before training:"
echo "MODEL_PATH=/path/to/lingbot-va-base bash script/extract_libero_lingbot_latents.sh"
