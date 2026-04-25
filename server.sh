pkill -f 'evaluation/libero/client.py' || true
pkill -f 'wan_va.wan_va_server --config-name libero' || true

PYTHONPATH=/data1/local/LIBERO \
conda run -n lingbot-va python -m torch.distributed.run \
  --nproc_per_node 1 \
  --master_port 29061 \
  -m wan_va.wan_va_server \
  --config-name libero \
  --port 29056 \
  --save_root visualization_libero_base_50_exact \
  --model_path /data1/local/lingbot-va/checkpoints/lingbot-va-base