# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

LingBot-VA is an autoregressive video-action world model for robotic manipulation, built on top of the Wan2.2 video diffusion transformer (Alibaba). The transformer jointly produces (a) future video latents and (b) robot actions, in an interleaved sequence, using a dual-stream Mixture-of-Transformers (MoT) backbone. Models are loaded via `from_pretrained` from a Wan2.2-style folder containing `transformer/`, `vae/`, `text_encoder/`, and `tokenizer/` subdirs.

## Environment

- Python 3.10.16, CUDA 12.6, torch 2.9.0 (per `pyproject.toml`/`requirements.txt`).
- Preferred setup for this workstation: `bash setting_codex.sh` — creates `.venv/` via `uv`, installs the runtime stack, clones LIBERO into `third_party/LIBERO`, patches `torch.load(... weights_only=False)` for LIBERO benchmark pickles, and writes `~/.libero/config.yaml`. `setting.sh` is a more minimal `uv` setup with no LIBERO bits.
- A `lingbot-va` **conda env** is still referenced by some legacy launch scripts ([server.sh](server.sh), [libero_client.sh](libero_client.sh), [eval.sh](eval.sh), [script/collect_libero_lingbot_dataset.sh](script/collect_libero_lingbot_dataset.sh), [script/extract_libero_lingbot_latents.sh](script/extract_libero_lingbot_latents.sh)). When `.venv/bin/python` exists, the newer wrappers ([i2va_libero.sh](i2va_libero.sh), [i2va_libero_overall.sh](i2va_libero_overall.sh), [script/train.sh](script/train.sh), [script/train_libero_lingbot_custom.sh](script/train_libero_lingbot_custom.sh)) prefer it and skip conda.
- LIBERO simulation requires `MUJOCO_GL=egl` (or `osmesa` on headless hosts) and matching `PYOPENGL_PLATFORM`. `PYTHONPATH` must include both the repo root and `third_party/LIBERO`.

## Critical: `attn_mode` in `transformer/config.json`

The transformer's attention kernel is read from `<MODEL_PATH>/transformer/config.json`, **not** from a Python flag. You must hand-edit this file when switching between training and inference:

| Mode | `attn_mode` |
| --- | --- |
| Training | `"flex"` |
| Inference / evaluation | `"torch"` or `"flashattn"` |

Mixing them silently fails (flex graphs error at eval; torch is too slow at train). The inference servers in [wan_va/wan_va_server.py](wan_va/wan_va_server.py) hard-code `attn_mode="torch"` when calling `load_transformer`, so the file value matters mostly for training.

## Common commands

```bash
# Fine-tune on LIBERO-Long (default: 2 GPUs, BATCH_SIZE=1, GRAD_ACCUM_STEPS=8, 5000 steps)
bash script/train.sh

# Action-modules-only fine-tune (freezes shared backbone)
bash script/train.sh --finetune_only_action

# Action modules + transformer LoRA adapters
bash script/train.sh --finetune_only_action --lora

# Smoke test (one optimizer step, no checkpointing)
NUM_STEPS=1 GRAD_ACCUM_STEPS=1 SAVE_INTERVAL=999999 bash script/train.sh

# Generic training launcher; pick a config from VA_CONFIGS in wan_va/configs/__init__.py
NGPU=8 CONFIG_NAME='robotwin_train' bash script/run_va_posttrain.sh
NGPU=8 CONFIG_NAME='libero_train'   bash script/run_va_posttrain.sh

# Inference server (one config; pass --model_path/--save_root/--port as overrides)
NGPU=1 CONFIG_NAME='libero'         bash script/run_launch_va_server_sync.sh --model_path /path/to/ckpt
NGPU=1 CONFIG_NAME='robotwin_i2av'  bash script/run_launch_va_server_sync.sh

# LIBERO end-to-end (server + sim client + comparison video), all in one script
bash i2va_libero_overall.sh

# LIBERO eval client only (server must already be running on port 29056)
bash evaluation/libero/launch_client.sh

# RoboTwin eval (server-client; must be on the same machine)
bash evaluation/robotwin/launch_server.sh
bash evaluation/robotwin/launch_client.sh results/ adjust_bottle

# Build a custom LIBERO dataset for training: collect → extract Wan latents → train
bash script/collect_libero_lingbot_dataset.sh
MODEL_PATH=$PWD/checkpoints/lingbot-va-base bash script/extract_libero_lingbot_latents.sh
DATASET_DIR=$PWD/datasets/libero_custom_lingbot bash script/train_libero_lingbot_custom.sh

# Code formatting (only acts on wan_va/)
make format
```

There is no test suite. The `tests/test.sh` referenced in [INSTALL.md](INSTALL.md) does not exist in-tree.

## Environment variables that flow into configs

Training/inference configs (e.g. [wan_va/configs/va_libero_train_cfg.py](wan_va/configs/va_libero_train_cfg.py)) read most knobs at import time from env vars, so they are how you override behavior — there is no CLI parser for them. Common ones:

- Paths: `DATASET_DIR`, `MODEL_PATH`, `SAVE_ROOT`, `REPO_ROOT`
- Data: `OBS_CAM_KEYS` (comma list), `USED_ACTION_CHANNEL_IDS` (comma list of ints into the 30-dim action), `LOAD_WORKER`, `NUM_INIT_WORKER`
- Training: `NGPU`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `NUM_STEPS`, `SAVE_INTERVAL`, `LEARNING_RATE`, `WARMUP_STEPS`, `CFG_PROB`, `GC_INTERVAL`, `MASTER_PORT`
- Logging: `ENABLE_WANDB` (`0`/`1`). [script/run_va_posttrain.sh](script/run_va_posttrain.sh) currently exports a hard-coded `WANDB_API_KEY`; treat that as a value not to be propagated beyond this repo.

Hydra-style positional overrides (e.g. `key=value`) passed to `script/run_va_posttrain.sh` / `script/run_launch_va_server_sync.sh` are forwarded to the python entry point and override config fields after env-var resolution.

## Code architecture

### `wan_va/` — model + training/serving entry points

- [wan_va/configs/__init__.py](wan_va/configs/__init__.py) defines `VA_CONFIGS`, the dict keyed by names passed to `--config-name` (`libero`, `libero_train`, `libero_i2av`, `robotwin`, `robotwin_train`, `robotwin_i2av`, `franka`, `franka_i2av`, `demo*`). Each config is an `easydict.EasyDict` that `update()`s from `va_shared_cfg` and the per-task base. Training configs inherit from inference configs and overlay env-driven training knobs (`*_train_cfg.py`).
- [wan_va/modules/model.py](wan_va/modules/model.py) defines `WanTransformer3DModel`, the dual-stream MoT transformer registered as a diffusers `ModelMixin`/`ConfigMixin`. It supports three attention paths: `flex` (`torch.nn.attention.flex_attention`, compiled), `flashattn` (`flash_attn_interface` or `flash_attn`), and `torch` (`F.scaled_dot_product_attention`).
- [wan_va/modules/utils.py](wan_va/modules/utils.py) holds `load_vae` / `load_text_encoder` / `load_tokenizer` / `load_transformer` (all `from_pretrained` against the Wan2.2-layout folder), plus `WanVAEStreamingWrapper` for chunked encode in inference.
- [wan_va/train.py](wan_va/train.py) is the fine-tuning entry (`python -m wan_va.train --config-name <name>`). It supports three regimes selected by CLI flags: full FT (default), `--finetune_only_action` (freeze backbone, train `action_embedder.`, `condition_embedder_action.`, `action_proj_out.` only), and adding `--lora` to additionally insert `LoRALinear` adapters on every `nn.Linear` inside `blocks.*`. Uses `MultiLatentLeRobotDataset`, `collate_latent_lerobot_batch` (trims to the shortest sequence in a batch), and writes a `trainable_parameter_report` to `SAVE_ROOT/checkpoints/`.
- [wan_va/wan_va_server.py](wan_va/wan_va_server.py) is the inference websocket server. It loads VAE + text encoder (optionally offloaded to CPU when `enable_offload=True`) and the FSDP-sharded transformer, then serves prediction requests for both the simulation eval clients and the i2va generation scripts.
- [wan_va/dataset/lerobot_latent_dataset.py](wan_va/dataset/lerobot_latent_dataset.py) implements `MultiLatentLeRobotDataset`. It reads the LeRobot dataset under `DATASET_DIR`, indexes the per-segment latent `.pth` files under `latents/`, and yields dicts of `latents`, `actions`, `actions_mask`, `text_emb`, etc.
- [wan_va/distributed/fsdp.py](wan_va/distributed/fsdp.py) shards per transformer block (`attn1`, `attn2`, `ffn`, then the block) using `fully_shard`, and provides `apply_ac` for activation checkpointing.

### `tools/libero_dataset/` — custom dataset pipeline

Three-stage pipeline driven by the wrappers in `script/`:

1. **Collect** raw LIBERO rollouts with a chosen policy (`zero`/`random`/`python`/`websocket`) into LeRobot format → [tools/libero_dataset/collect_libero_lingbot_dataset.py](tools/libero_dataset/collect_libero_lingbot_dataset.py).
2. **Extract Wan VAE latents** per `action_config` segment → [tools/libero_dataset/extract_wan_latents_for_lingbot.py](tools/libero_dataset/extract_wan_latents_for_lingbot.py). Output goes to `latents/chunk-XXX/<cam>/episode_{idx}_{start}_{end}.pth` mirroring `videos/`.
3. **Prepare** the dataset for training → [tools/libero_dataset/prepare_lingbot_dataset.py](tools/libero_dataset/prepare_lingbot_dataset.py), which writes `meta/action_norm_quantiles.json`. The training config (`va_libero_train_cfg.py`) auto-loads these quantiles into `cfg.norm_stat`.

[script/train.sh](script/train.sh) also auto-generates `DATASET_DIR/empty_emb.pt` (the empty-prompt T5 embedding required for CFG dropout) on first run.

### `evaluation/` — sim-side evaluation harnesses

Inference server and sim client communicate via websockets on a configurable port (default 29056 for LIBERO). `evaluation/libero/{launch_server,launch_client}.sh` and `evaluation/robotwin/{launch_server,launch_client}*.sh` are thin wrappers; the actual server logic lives in [wan_va/wan_va_server.py](wan_va/wan_va_server.py). RoboTwin multi-GPU launch pads 50 tasks to 56 and groups them into 7 groups of 8 — see [evaluation/robotwin/launch_client_multigpus.sh](evaluation/robotwin/launch_client_multigpus.sh).

## Data contract

- LeRobot format under `DATASET_DIR/{meta,videos,latents}/`. `meta/episodes.jsonl` must include `action_config: [{start_frame, end_frame, action_text}, ...]` for every episode.
- Latent files are `.pth` dicts with keys: `latent` (flat `[N, C]` bfloat16), `latent_num_frames`/`latent_height`/`latent_width`, `video_num_frames`/`video_height`/`video_width`, `text_emb` (`[L, D]` bfloat16), `text`, `frame_ids`, `start_frame`, `end_frame`, `fps`, `ori_fps`. Naming: `episode_{index}_{start_frame}_{end_frame}.pth`.
- Action vector is **always 30-dim** in this codebase, structured `[L-EEF(7), R-EEF(7), L-joints(7), R-joints(7), L-gripper(1), R-gripper(1)]`. Robots with fewer DOF zero-pad the unused slots and set `USED_ACTION_CHANNEL_IDS` to the indices that actually carry signal (LIBERO uses `0..6`, RoboTwin uses all 30). `inverse_used_action_channel_ids` is derived inside the config — do not set it directly.

## Submodules and external state

- `third_party/LIBERO` is a git submodule (https://github.com/Lifelong-Robot-Learning/LIBERO). [setting_codex.sh](setting_codex.sh) patches its `benchmark/__init__.py` to pass `weights_only=False` to `torch.load` so PyTorch 2.6+ can load the benchmark init-state pickles.
- Checkpoints live under `checkpoints/` (e.g. `lingbot-va-base`, `lingbot-va-posttrain-libero-long`). Downloaded from HuggingFace `robbyant/*` or ModelScope `Robbyant/*` — see [README.md](README.md).
- Local datasets under `datasets/` (e.g. `libero-long-lerobot`). The default `DATASET_DIR` in `va_libero_train_cfg.py` still points at an absolute path from another machine; always set `DATASET_DIR` explicitly when invoking training to avoid that fallback.
