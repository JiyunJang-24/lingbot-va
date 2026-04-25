PYTHONPATH=/data1/local/lingbot-va:/data1/local/LIBERO \
MUJOCO_GL=egl \
conda run -n lingbot-va python evaluation/libero/client.py \
  --libero-benchmark libero_goal \
  --port 29056 \
  --test-num 1 \
  --task-range 0 1 \
  --max-timesteps 200 \
  --out-dir outputs/libero_goal_base_time_200