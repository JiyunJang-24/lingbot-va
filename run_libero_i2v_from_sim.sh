#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

PYTHONPATH=/root/Desktop/workspace/lingbot-va/third_party/LIBERO \
MUJOCO_GL=egl \
conda run -n lingbot-va python evaluation/libero/run_i2va_from_first_scene.py \
  --benchmark libero_spatial \
  --test-num 1 \
  --task-idx 0 \
  --model-path /root/Desktop/workspace/lingbot-va/checkpoints/lingbot-va-base \
  --output-root /root/Desktop/workspace/lingbot-va/outputs/libero_i2va