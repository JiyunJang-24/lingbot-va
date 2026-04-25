# LIBERO to LingBot-VA Dataset Pipeline

This folder contains the local LIBERO dataset tools. They do not modify the
LIBERO source tree.

## 1. Collect rollouts

```bash
DATASET_DIR=/data1/local/lingbot-va/datasets/libero_custom_lingbot \
LIBERO_BENCHMARK=libero_10 \
TASK_START=0 \
TASK_END=1 \
EPISODES_PER_TASK=5 \
POLICY=python \
POLICY_FILE=/data1/local/lingbot-va/tools/libero_dataset/example_policy.py \
bash script/collect_libero_lingbot_dataset.sh
```

`POLICY` can be `zero`, `random`, `python`, or `websocket`. For a custom Python
policy, define one of:

- `make_policy(**kwargs)` returning an object with `act(context)`
- `policy` object with `act(context)` or callable behavior
- `act(context)` function

The context includes `raw_obs`, extracted RGB `obs`, `prompt`, `task_idx`,
`episode_index`, `frame_index`, and `env`.

## 2. Extract Wan latents

LingBot-VA training requires `latents/` and `empty_emb.pt` in addition to
LeRobot parquet/video metadata.

```bash
DATASET_DIR=/data1/local/lingbot-va/datasets/libero_custom_lingbot \
MODEL_PATH=/data1/local/lingbot-va/checkpoints/lingbot-va-base \
bash script/extract_libero_lingbot_latents.sh
```

## 3. Train

Set `va_libero_train_cfg.dataset_path` to your dataset directory, and if your
action scale differs from LIBERO-long, copy values from
`meta/action_norm_quantiles.json` into `va_libero_cfg.norm_stat`.

```bash
DATASET_DIR=/data1/local/lingbot-va/datasets/libero_custom_lingbot \
NGPU=8 \
bash script/train_libero_lingbot_custom.sh
```
