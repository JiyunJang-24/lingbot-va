#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$REPO_ROOT"

export UV_LINK_MODE="${UV_LINK_MODE:-copy}"

# 1) Python environment
uv venv .venv --python 3.10.16 --prompt "$(basename "$PWD")"
source .venv/bin/activate
python -m ensurepip --upgrade

# 2) Core runtime and training dependencies.
# Let lerobot resolve the compatible torch/torchvision stack. On this machine it
# resolves to torch 2.7.1+cu126, which was verified with CUDA on RTX A6000.
uv pip install \
  "numpy==1.26.4" \
  "diffusers==0.36.0" \
  "transformers==4.55.2" \
  accelerate \
  einops \
  easydict \
  "imageio[ffmpeg]" \
  websockets \
  msgpack \
  opencv-python \
  matplotlib \
  ftfy \
  safetensors \
  scipy \
  wandb \
  setuptools \
  ninja \
  packaging \
  wheel \
  "lerobot==0.3.3"

# The local LeRobot parquet metadata uses the datasets "List" feature. The
# version pulled by lerobot (<=3.6.0) cannot read it, so pin a newer datasets.
uv pip install "datasets==4.8.5"

# Do not install flash-attn by default here. The prebuilt flash-attn wheel needs
# a newer glibc than this host has, while a local source build is very slow.
# The model/server paths used below run with attn_mode="torch".

# 3) LIBERO simulator dependency for i2va_libero_overall.sh.
mkdir -p third_party
if [[ ! -d third_party/LIBERO/.git ]]; then
  git clone --depth 1 https://github.com/Lifelong-Robot-Learning/LIBERO.git third_party/LIBERO
fi

# PyTorch 2.6+ defaults torch.load(weights_only=True), but LIBERO init states
# are trusted benchmark pickles and need the old behavior.
python - <<'PY'
from pathlib import Path
path = Path("third_party/LIBERO/libero/libero/benchmark/__init__.py")
text = path.read_text()
text = text.replace("torch.load(init_states_path)", "torch.load(init_states_path, weights_only=False)")
path.write_text(text)
PY

uv pip install \
  -e third_party/LIBERO \
  "robosuite==1.4.0" \
  "bddl==1.0.1" \
  "robomimic==0.2.0" \
  "gym==0.25.2" \
  "hydra-core==1.2.0" \
  "thop==0.1.1.post2209072238" \
  "future==0.18.2"

# Avoid LIBERO's first-import interactive prompt.
mkdir -p "${LIBERO_CONFIG_PATH:-$HOME/.libero}"
cat > "${LIBERO_CONFIG_PATH:-$HOME/.libero}/config.yaml" <<EOF
assets: $REPO_ROOT/third_party/LIBERO/libero/libero/assets
bddl_files: $REPO_ROOT/third_party/LIBERO/libero/libero/bddl_files
benchmark_root: $REPO_ROOT/third_party/LIBERO/libero/libero
datasets: $REPO_ROOT/third_party/LIBERO/libero/datasets
init_states: $REPO_ROOT/third_party/LIBERO/libero/libero/init_files
EOF

# 4) Common environment for the requested custom training dataset.
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/third_party/LIBERO:${PYTHONPATH:-}"
export DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/pick-n-place-sq-lerobot-v21}"
export MODEL_PATH="${MODEL_PATH:-$REPO_ROOT/checkpoints/lingbot-va-base}"
export OBS_CAM_KEYS="${OBS_CAM_KEYS:-observation.images.top,observation.images.wrist}"
export USED_ACTION_CHANNEL_IDS="${USED_ACTION_CHANNEL_IDS:-0,1,2,3,4,5}"
export LOAD_WORKER="${LOAD_WORKER:-0}"
export NUM_INIT_WORKER="${NUM_INIT_WORKER:-1}"
export ENABLE_WANDB="${ENABLE_WANDB:-0}"

python - <<'PY'
import torch
from wan_va.configs import VA_CONFIGS
from wan_va.dataset import MultiLatentLeRobotDataset
from libero.libero import benchmark

print("python", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
cfg = VA_CONFIGS["libero_train"]
dset = MultiLatentLeRobotDataset(cfg, num_init_worker=1)
sample = dset[0]
print("dataset_len", len(dset), "sample_shapes", {k: tuple(v.shape) for k, v in sample.items()})
print("libero_10_available", "libero_10" in benchmark.get_benchmark_dict())
PY

cat <<'EOF'

Environment ready.

Training smoke command:
  DATASET_DIR="$PWD/datasets/pick-n-place-sq-lerobot-v21" \
  MODEL_PATH="$PWD/checkpoints/lingbot-va-base" \
  OBS_CAM_KEYS=observation.images.top,observation.images.wrist \
  USED_ACTION_CHANNEL_IDS=0,1,2,3,4,5 \
  NUM_STEPS=1 BATCH_SIZE=1 GRAD_ACCUM_STEPS=1 LOAD_WORKER=0 NUM_INIT_WORKER=1 NGPU=1 \
  bash script/train_libero_lingbot_custom.sh

LIBERO i2va smoke command:
  MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa bash i2va_libero_overall.sh
EOF
