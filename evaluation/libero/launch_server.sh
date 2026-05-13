
MODEL_PATH=${MODEL_PATH:-checkpoints/lingbot-va-posttrain-libero-long}
export PYTHONPATH="$PWD:$PWD/third_party/LIBERO:${PYTHONPATH:-}"
save_root='visualization/'
mkdir -p $save_root

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29061 \
    wan_va/wan_va_server.py \
    --config-name libero \
    --port 29056 \
    --save_root $save_root \
    --model_path $MODEL_PATH