# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict
from .va_libero_cfg import va_libero_cfg
import os
import json


def _load_norm_stat(dataset_path):
    norm_path = os.path.join(dataset_path, "meta", "action_norm_quantiles.json")
    if not os.path.exists(norm_path):
        return va_libero_cfg.norm_stat
    with open(norm_path) as f:
        stat = json.load(f)
    pad = 30 - len(stat["q01"])
    return {
        "q01": stat["q01"] + [0.0] * pad,
        "q99": stat["q99"] + [0.0] * pad,
    }


def _split_env_list(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def _split_env_int_list(name, default):
    value = os.environ.get(name)
    if not value:
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]

va_libero_train_cfg = EasyDict(__name__='Config: VA libero train')
va_libero_train_cfg.update(va_libero_cfg)

va_libero_train_cfg.dataset_path = os.environ.get(
    "DATASET_DIR",
    "/path/to/your/dataset",
)
va_libero_train_cfg.empty_emb_path = os.path.join(va_libero_train_cfg.dataset_path, 'empty_emb.pt')
va_libero_train_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    "MODEL_PATH",
    va_libero_train_cfg.wan22_pretrained_model_name_or_path,
)
va_libero_train_cfg.obs_cam_keys = _split_env_list(
    "OBS_CAM_KEYS",
    va_libero_train_cfg.obs_cam_keys,
)
va_libero_train_cfg.used_action_channel_ids = _split_env_int_list(
    "USED_ACTION_CHANNEL_IDS",
    va_libero_train_cfg.used_action_channel_ids,
)
inverse_used_action_channel_ids = [len(va_libero_train_cfg.used_action_channel_ids)] * va_libero_train_cfg.action_dim
for i, j in enumerate(va_libero_train_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_libero_train_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids
va_libero_train_cfg.enable_wandb = os.environ.get("ENABLE_WANDB", "0") == "1"
va_libero_train_cfg.load_worker = int(os.environ.get("LOAD_WORKER", "0"))
va_libero_train_cfg.num_init_worker = int(os.environ.get("NUM_INIT_WORKER", "1"))
va_libero_train_cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", "200"))
va_libero_train_cfg.gc_interval = int(os.environ.get("GC_INTERVAL", "50"))
va_libero_train_cfg.cfg_prob = float(os.environ.get("CFG_PROB", "0.1"))
va_libero_train_cfg.save_root = os.environ.get("SAVE_ROOT", va_libero_train_cfg.save_root)
va_libero_train_cfg.norm_stat = _load_norm_stat(va_libero_train_cfg.dataset_path)

# Training parameters
va_libero_train_cfg.learning_rate = float(os.environ.get("LEARNING_RATE", "1e-5"))
va_libero_train_cfg.beta1 = 0.9
va_libero_train_cfg.beta2 = 0.95
va_libero_train_cfg.weight_decay = 1e-1
va_libero_train_cfg.warmup_steps = int(os.environ.get("WARMUP_STEPS", "10"))
va_libero_train_cfg.batch_size = int(os.environ.get("BATCH_SIZE", "1"))
va_libero_train_cfg.gradient_accumulation_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "10"))
va_libero_train_cfg.num_steps = int(os.environ.get("NUM_STEPS", "5000"))
