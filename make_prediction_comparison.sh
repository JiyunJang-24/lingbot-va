

folder_name="libero_goal_base_time_200"

actual=$(find outputs/$folder_name -name '*.mp4' | sort | tail -1)
pred_dir=$(find visualization_libero_base_50_exact/real -mindepth 1 -maxdepth 1 -type d | sort | tail -1)

echo $actual
echo $pred_dir

conda run -n lingbot-va python evaluation/libero/make_prediction_comparison.py \
  --actual-video "$actual" \
  --latents-dir "$pred_dir" \
  --model-path /data1/local/lingbot-va/checkpoints/lingbot-va-base \
  --predicted-video outputs/$folder_name/predicted_from_latents.mp4 \
  --comparison-video outputs/$folder_name/actual_vs_predicted.mp4 \
  --fps 15