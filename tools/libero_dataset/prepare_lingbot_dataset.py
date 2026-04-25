#!/usr/bin/env python3
"""Validate and patch a collected LeRobot dataset for LingBot-VA training."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def patch_action_config(dataset_dir: Path) -> None:
    episodes_path = dataset_dir / "meta/episodes.jsonl"
    episodes = read_jsonl(episodes_path)
    changed = False
    for ep in episodes:
        if "action_config" not in ep or not ep["action_config"]:
            task_text = ep.get("tasks", [""])[0]
            ep["action_config"] = [
                {
                    "start_frame": 0,
                    "end_frame": int(ep["length"]),
                    "action_text": task_text,
                    "skill": "",
                }
            ]
            changed = True
    if changed:
        write_jsonl(episodes_path, episodes)
        write_jsonl(dataset_dir / "meta/episodes_ori.jsonl", episodes)


def compute_action_quantiles(dataset_dir: Path, q_low: float, q_high: float) -> dict[str, list[float]]:
    actions = []
    for parquet_path in sorted((dataset_dir / "data").glob("chunk-*/episode_*.parquet")):
        df = pd.read_parquet(parquet_path, columns=["action"])
        actions.append(np.stack(df["action"].to_numpy()).astype(np.float32))
    if not actions:
        raise FileNotFoundError(f"No parquet episodes found under {dataset_dir / 'data'}")
    action = np.concatenate(actions, axis=0)
    return {
        "q01": np.quantile(action, q_low, axis=0).astype(float).tolist(),
        "q99": np.quantile(action, q_high, axis=0).astype(float).tolist(),
    }


def validate_latents(dataset_dir: Path) -> tuple[int, list[str]]:
    episodes = read_jsonl(dataset_dir / "meta/episodes.jsonl")
    with (dataset_dir / "meta/info.json").open() as f:
        info = json.load(f)
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    missing = []
    checked = 0
    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        chunk = ep_idx // int(info.get("chunks_size", 1000))
        for acfg in ep["action_config"]:
            start = int(acfg["start_frame"])
            end = int(acfg["end_frame"])
            for key in video_keys:
                path = dataset_dir / "latents" / f"chunk-{chunk:03d}" / key / f"episode_{ep_idx:06d}_{start}_{end}.pth"
                checked += 1
                if not path.is_file():
                    missing.append(str(path))
    return checked, missing


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--copy-empty-emb-from", type=Path, default=None)
    parser.add_argument("--require-latents", action="store_true")
    parser.add_argument("--q-low", type=float, default=0.01)
    parser.add_argument("--q-high", type=float, default=0.99)
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    patch_action_config(dataset_dir)

    norm_stat = compute_action_quantiles(dataset_dir, args.q_low, args.q_high)
    norm_path = dataset_dir / "meta/action_norm_quantiles.json"
    norm_path.write_text(json.dumps(norm_stat, indent=2) + "\n")
    print(f"[DONE] Wrote action quantiles: {norm_path}")

    if args.copy_empty_emb_from:
        src = args.copy_empty_emb_from.resolve()
        dst = dataset_dir / "empty_emb.pt"
        shutil.copy2(src, dst)
        print(f"[DONE] Copied empty_emb.pt from {src}")

    checked, missing = validate_latents(dataset_dir)
    if missing:
        print(f"[WARN] Missing {len(missing)}/{checked} latent files. First missing: {missing[0]}")
        if args.require_latents:
            raise FileNotFoundError("Latent validation failed.")
    else:
        print(f"[DONE] All {checked} latent files are present.")


if __name__ == "__main__":
    main()
