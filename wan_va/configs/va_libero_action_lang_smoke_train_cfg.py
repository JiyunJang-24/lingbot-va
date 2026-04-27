# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
import os

from easydict import EasyDict

from .va_libero_cfg import va_libero_cfg


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


va_libero_action_lang_smoke_train_cfg = EasyDict(__name__="Config: VA libero action-language smoke train")
va_libero_action_lang_smoke_train_cfg.update(va_libero_cfg)

_dataset_path = os.environ.get(
    "DATASET_DIR",
    "/data1/local/lingbot-va/datasets/libero_table_cover_action_lang_smoke",
)
_model_path = os.environ.get(
    "MODEL_PATH",
    "/data1/local/lingbot-va/checkpoints/lingbot-va-base",
)

va_libero_action_lang_smoke_train_cfg.dataset_path = _dataset_path
va_libero_action_lang_smoke_train_cfg.empty_emb_path = os.path.join(_dataset_path, "empty_emb.pt")
va_libero_action_lang_smoke_train_cfg.wan22_pretrained_model_name_or_path = _model_path
va_libero_action_lang_smoke_train_cfg.enable_wandb = False
va_libero_action_lang_smoke_train_cfg.load_worker = int(os.environ.get("LOAD_WORKER", "0"))
va_libero_action_lang_smoke_train_cfg.num_init_worker = int(os.environ.get("NUM_INIT_WORKER", "1"))
va_libero_action_lang_smoke_train_cfg.save_interval = int(os.environ.get("SAVE_INTERVAL", "1"))
va_libero_action_lang_smoke_train_cfg.gc_interval = 1
va_libero_action_lang_smoke_train_cfg.cfg_prob = 0.0

va_libero_action_lang_smoke_train_cfg.learning_rate = float(os.environ.get("LEARNING_RATE", "1e-5"))
va_libero_action_lang_smoke_train_cfg.beta1 = 0.9
va_libero_action_lang_smoke_train_cfg.beta2 = 0.95
va_libero_action_lang_smoke_train_cfg.weight_decay = 1e-1
va_libero_action_lang_smoke_train_cfg.warmup_steps = 1
va_libero_action_lang_smoke_train_cfg.batch_size = int(os.environ.get("BATCH_SIZE", "2"))
va_libero_action_lang_smoke_train_cfg.gradient_accumulation_steps = int(os.environ.get("GRAD_ACCUM_STEPS", "1"))
va_libero_action_lang_smoke_train_cfg.num_steps = int(os.environ.get("NUM_STEPS", "1"))
va_libero_action_lang_smoke_train_cfg.save_root = os.environ.get(
    "SAVE_ROOT",
    "/data1/local/lingbot-va/outputs/libero_table_cover_action_lang_smoke_train",
)
va_libero_action_lang_smoke_train_cfg.norm_stat = _load_norm_stat(_dataset_path)
