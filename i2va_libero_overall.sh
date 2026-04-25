pkill -f 'evaluation/libero/client.py' || true
pkill -f 'wan_va.wan_va_server --config-name libero' || true


save_folder_name="libero_10_base_time_200"
libero_benchmark="libero_10"
model_path="/data1/local/lingbot-va/checkpoints/lingbot-va-base"
mkdir -p outputs/$save_folder_name

# Start the server
(
  PYTHONPATH=/data1/local/LIBERO \
  conda run -n lingbot-va python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29061 \
    -m wan_va.wan_va_server \
    --config-name libero \
    --port 29056 \
    --save_root outputs/$save_folder_name/data \
    --model_path $model_path
) &
(
    PYTHONPATH=/data1/local/lingbot-va:/data1/local/LIBERO \
    MUJOCO_GL=egl \
    conda run -n lingbot-va python evaluation/libero/client.py \
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
conda run -n lingbot-va python evaluation/libero/make_prediction_comparison.py \
  --actual-video "$actual" \
  --latents-dir "$pred_dir" \
  --model-path $model_path \
  --predicted-video outputs/$save_folder_name/predicted_from_latents.mp4 \
  --comparison-video outputs/$save_folder_name/actual_vs_predicted.mp4 \
  --fps 10

# Make the video only from the predicted latents

pkill -f 'evaluation/libero/client.py' || true
pkill -f 'wan_va.wan_va_server --config-name libero' || true


PYTHONPATH=/data1/local/lingbot-va:/data1/local/LIBERO \
MUJOCO_GL=egl \
conda run -n lingbot-va python evaluation/libero/run_i2va_from_first_scene.py \
  --benchmark $libero_benchmark \
  --test-num 1 \
  --task-idx 0 \
  --model-path $model_path \
  --output-root outputs/$save_folder_name