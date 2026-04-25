#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data1/local/lingbot-va}"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-lingbot-va}"
export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"

DATASET_DIR="${DATASET_DIR:-$REPO_ROOT/datasets/libero_custom_lingbot}"
NGPU="${NGPU:-1}"
CONFIG_NAME="${CONFIG_NAME:-libero_train}"

echo "[INFO] Set wan_va/configs/va_libero_train_cfg.py dataset_path to:"
echo "       $DATASET_DIR"
echo "[INFO] Copy q01/q99 from $DATASET_DIR/meta/action_norm_quantiles.json into va_libero_cfg.norm_stat if your action scale differs."

conda run -n "$CONDA_ENV" env NGPU="$NGPU" CONFIG_NAME="$CONFIG_NAME" bash script/run_va_posttrain.sh
