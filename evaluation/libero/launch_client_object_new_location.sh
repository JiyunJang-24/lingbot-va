START=0
END=1
export PYTHONPATH="$PWD:$PWD/third_party/LIBERO:${PYTHONPATH:-}"

python evaluation/libero/client_object_new_location.py \
    --libero-benchmark libero_10 \
    --port 29056 \
    --test-num 11 \
    --task-range $START $END \
    --out-dir outputs/libero
