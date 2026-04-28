#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
if [[ -z "${LIBERO_ROOT:-}" && -d "$REPO_ROOT/third_party/LIBERO" ]]; then
  LIBERO_ROOT="$REPO_ROOT/third_party/LIBERO"
else
  LIBERO_ROOT="${LIBERO_ROOT:-/data1/local/LIBERO}"
fi
CONDA_ENV="${CONDA_ENV:-lingbot-va}"
MUJOCO_GL_BACKEND="${MUJOCO_GL:-egl}"
PYOPENGL_BACKEND="${PYOPENGL_PLATFORM:-$MUJOCO_GL_BACKEND}"

cd "$REPO_ROOT"

if [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="conda run -n $CONDA_ENV python"
fi

pkill -f 'evaluation/libero/client.py' || true
pkill -f 'wan_va.wan_va_server --config-name libero' || true


save_folder_name="libero_10_base_time_200"
libero_benchmark="libero_10"
model_path="${MODEL_PATH:-$REPO_ROOT/checkpoints/lingbot-va-base}"
mkdir -p outputs/$save_folder_name

# Start the server
(
  PYTHONPATH="$REPO_ROOT:$LIBERO_ROOT:${PYTHONPATH:-}" \
  $PYTHON_BIN -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29061 \
    -m wan_va.wan_va_server \
    --config-name libero \
    --port 29056 \
    --save_root outputs/$save_folder_name/data \
    --model_path $model_path
) &
(
    PYTHONPATH="$REPO_ROOT:$LIBERO_ROOT:${PYTHONPATH:-}" \
    MUJOCO_GL="$MUJOCO_GL_BACKEND" \
    PYOPENGL_PLATFORM="$PYOPENGL_BACKEND" \
    $PYTHON_BIN evaluation/libero/client.py \
    --libero-benchmark $libero_benchmark \
    --port 29056 \
    --test-num 1 \
    --task-range 0 1 \
    --max-timesteps 200 \
    --out-dir outputs/$save_folder_name
)
#time sleep 5

actual=$(find outputs/$save_folder_name -name '*.mp4' | sort | tail -1)
pred_dir=$(find outputs/$save_folder_name/data/real -mindepth 1 -maxdepth 1 -type d | sort | tail -1)

echo $actual
echo $pred_dir

# Make the comparison video between the actual and predicted videos
$PYTHON_BIN evaluation/libero/make_prediction_comparison.py \
  --actual-video "$actual" \
  --latents-dir "$pred_dir" \
  --model-path $model_path \
  --predicted-video outputs/$save_folder_name/predicted_from_latents.mp4 \
  --comparison-video outputs/$save_folder_name/actual_vs_predicted.mp4 \
  --fps 10

# Make the video only from the predicted latents

pkill -f 'evaluation/libero/client.py' || true
pkill -f 'wan_va.wan_va_server --config-name libero' || true


PYTHONPATH="$REPO_ROOT:$LIBERO_ROOT:${PYTHONPATH:-}" \
MUJOCO_GL="$MUJOCO_GL_BACKEND" \
PYOPENGL_PLATFORM="$PYOPENGL_BACKEND" \
$PYTHON_BIN evaluation/libero/run_i2va_from_first_scene.py \
  --benchmark $libero_benchmark \
  --test-num 1 \
  --task-idx 0 \
  --model-path $model_path \
  --output-root outputs/$save_folder_name
