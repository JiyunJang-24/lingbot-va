#!/usr/bin/env bash
# Create symlink-based run directories that MultiLatentLeRobotDataset will discover via
# recursive info.json walk. Excludes num3 and num5 per user request.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

REGULAR_SRC="datasets/libero_spatial_object_location"
REWIND_SRC="datasets/libero_spatial_object_location_rewind_gripper"
SUBS=(num1 num2 num4 num6 num7 num8 num9 num10)

RUN_A="datasets/_runs/run_a_regular"
RUN_B="datasets/_runs/run_b_combined"

# --- Run A: regular only ---
rm -rf "$RUN_A"
mkdir -p "$RUN_A"
for sub in "${SUBS[@]}"; do
  ln -s "$REPO_ROOT/$REGULAR_SRC/v-1.000-1.000_$sub" "$RUN_A/$sub"
done

# meta/action_norm_quantiles.json at the run root: copy from regular num1
mkdir -p "$RUN_A/meta"
cp "$REGULAR_SRC/v-1.000-1.000_num1/meta/action_norm_quantiles.json" "$RUN_A/meta/action_norm_quantiles.json"
# empty_emb.pt: copy from regular num1 (training auto-creates if missing, but provide it)
cp "$REGULAR_SRC/v-1.000-1.000_num1/empty_emb.pt" "$RUN_A/empty_emb.pt"

# --- Run B: regular + rewind, marked by prefix ---
rm -rf "$RUN_B"
mkdir -p "$RUN_B"
for sub in "${SUBS[@]}"; do
  ln -s "$REPO_ROOT/$REGULAR_SRC/v-1.000-1.000_$sub" "$RUN_B/regular_$sub"
  ln -s "$REPO_ROOT/$REWIND_SRC/v-1.000-1.000_$sub" "$RUN_B/rewind_$sub"
done
mkdir -p "$RUN_B/meta"
cp "$REGULAR_SRC/v-1.000-1.000_num1/meta/action_norm_quantiles.json" "$RUN_B/meta/action_norm_quantiles.json"
cp "$REGULAR_SRC/v-1.000-1.000_num1/empty_emb.pt" "$RUN_B/empty_emb.pt"

echo "=== Run A ($RUN_A) ==="
ls -la "$RUN_A"
echo ""
echo "=== Run B ($RUN_B) ==="
ls -la "$RUN_B"
