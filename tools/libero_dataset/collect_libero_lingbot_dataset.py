#!/usr/bin/env python3
"""Collect LIBERO rollouts and write a LingBot-VA compatible LeRobot dataset."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import cv2
import imageio.v2 as imageio
import numpy as np
import pandas as pd
from tqdm import tqdm

from action_providers import make_action_provider


VIDEO_KEYS = [
    "observation.images.agentview_rgb",
    "observation.images.eye_in_hand_rgb",
]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def as_rgb_uint8(frame: np.ndarray, size: int) -> np.ndarray:
    frame = np.asarray(frame)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    if frame.shape[0] != size or frame.shape[1] != size:
        frame = cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame)


def extract_obs(raw_obs: dict[str, Any], image_size: int) -> dict[str, np.ndarray]:
    return {
        "observation.images.agentview_rgb": as_rgb_uint8(
            raw_obs["agentview_image"][::-1], image_size
        ),
        "observation.images.eye_in_hand_rgb": as_rgb_uint8(
            raw_obs["robot0_eye_in_hand_image"][::-1], image_size
        ),
    }


def extract_state(raw_obs: dict[str, Any]) -> np.ndarray:
    pos = np.asarray(raw_obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32)
    quat = np.asarray(raw_obs.get("robot0_eef_quat", np.array([0, 0, 0, 1])), dtype=np.float32)
    gripper = np.asarray(raw_obs.get("robot0_gripper_qpos", np.zeros(2)), dtype=np.float32)
    return np.concatenate([pos[:3], quat[:4], gripper[:1]]).astype(np.float32)


def construct_single_env(env_args: dict[str, Any], offscreen_env_cls: Any):
    last_exc = None
    for attempt in range(5):
        try:
            return offscreen_env_cls(**env_args)
        except Exception as exc:
            last_exc = exc
            print(f"[WARN] Env creation failed ({attempt + 1}/5): {exc}")
            time.sleep(2)
    raise RuntimeError(f"Failed to create LIBERO env: {last_exc}")


def init_env(env: Any, init_state: np.ndarray, image_size: int) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    env.reset()
    env.set_init_state(init_state)
    raw_obs = None
    for _ in range(5):
        raw_obs, _, _, _ = env.step([0.0] * 7)
    assert raw_obs is not None
    return raw_obs, extract_obs(raw_obs, image_size)


def save_videos(video_dir: Path, episode_index: int, frames_by_key: dict[str, list[np.ndarray]], fps: int) -> None:
    for key, frames in frames_by_key.items():
        out_dir = video_dir / "chunk-000" / key
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"episode_{episode_index:06d}.mp4"
        imageio.mimsave(out_path, frames, fps=fps, macro_block_size=1)


def numeric_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).astype(float).tolist(),
        "max": values.max(axis=0).astype(float).tolist(),
        "mean": values.mean(axis=0).astype(float).tolist(),
        "std": values.std(axis=0).astype(float).tolist(),
        "count": [int(values.shape[0])],
    }


def image_stats(frames: list[np.ndarray]) -> dict[str, Any]:
    arr = np.stack(frames).astype(np.float32) / 255.0
    channel_axes = (0, 1, 2)
    return {
        "min": arr.min(axis=channel_axes).reshape(3, 1, 1).astype(float).tolist(),
        "max": arr.max(axis=channel_axes).reshape(3, 1, 1).astype(float).tolist(),
        "mean": arr.mean(axis=channel_axes).reshape(3, 1, 1).astype(float).tolist(),
        "std": arr.std(axis=channel_axes).reshape(3, 1, 1).astype(float).tolist(),
        "count": [int(arr.shape[0])],
    }


def write_metadata(
    out_dir: Path,
    *,
    episodes: list[dict[str, Any]],
    episode_stats: list[dict[str, Any]],
    tasks: list[str],
    fps: int,
    image_size: int,
    action_dim: int,
    state_dim: int,
) -> None:
    task_rows = [{"task_index": i, "task": task} for i, task in enumerate(tasks)]
    total_frames = sum(ep["length"] for ep in episodes)
    info = {
        "codebase_version": "v2.1",
        "robot_type": "Franka",
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episodes) * len(VIDEO_KEYS),
        "total_chunks": 1,
        "chunks_size": 1000,
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [state_dim],
                "names": {"motors": [f"state_{i}" for i in range(state_dim)]},
            },
            "action": {
                "dtype": "float32",
                "shape": [action_dim],
                "names": {"motors": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"][:action_dim]},
            },
            **{
                key: {
                    "dtype": "video",
                    "shape": [image_size, image_size, 3],
                    "names": ["height", "width", "rgb"],
                    "info": {
                        "video.height": image_size,
                        "video.width": image_size,
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "video.is_depth_map": False,
                        "video.fps": fps,
                        "video.channels": 3,
                        "has_audio": False,
                    },
                }
                for key in VIDEO_KEYS
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
    }
    write_json(out_dir / "meta/info.json", info)
    append_jsonl(out_dir / "meta/tasks.jsonl", task_rows)
    append_jsonl(out_dir / "meta/episodes.jsonl", episodes)
    append_jsonl(out_dir / "meta/episodes_ori.jsonl", episodes)
    append_jsonl(out_dir / "meta/episodes_stats.jsonl", episode_stats)


def collect(args: argparse.Namespace) -> None:
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data/chunk-000").mkdir(parents=True, exist_ok=True)
    provider = make_action_provider(
        args.policy,
        policy_file=args.policy_file,
        port=args.port,
        action_dim=args.action_dim,
        random_scale=args.random_scale,
        seed=args.seed,
    )

    benchmark_instance = benchmark.get_benchmark_dict()[args.libero_benchmark]()
    task_indices = list(range(args.task_start, args.task_end or benchmark_instance.get_num_tasks()))
    episodes: list[dict[str, Any]] = []
    episode_stats: list[dict[str, Any]] = []
    task_names: list[str] = []
    global_index = 0
    episode_index = 0

    try:
        for task_idx in task_indices:
            task = benchmark_instance.get_task(task_idx)
            prompt = task.language
            if prompt not in task_names:
                task_names.append(prompt)
            task_index = task_names.index(prompt)
            env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": args.image_size,
                "camera_widths": args.image_size,
            }
            init_states = benchmark_instance.get_task_init_states(task_idx)
            env = construct_single_env(env_args, OffScreenRenderEnv)
            try:
                iterator = range(args.episodes_per_task)
                for local_ep in tqdm(iterator, desc=f"task {task_idx}: {prompt[:48]}"):
                    init_state = init_states[(args.init_state_offset + local_ep) % init_states.shape[0]]
                    raw_obs, obs = init_env(env, init_state, args.image_size)
                    provider.reset(prompt=prompt, task_idx=task_idx, episode_idx=episode_index)

                    rows = []
                    frames_by_key = {key: [] for key in VIDEO_KEYS}
                    done = False

                    for frame_index in range(args.max_steps):
                        context = {
                            "raw_obs": raw_obs,
                            "obs": obs,
                            "prompt": prompt,
                            "task_idx": task_idx,
                            "episode_index": episode_index,
                            "frame_index": frame_index,
                            "env": env,
                        }
                        action = provider.act(context).reshape(-1).astype(np.float32)
                        if action.shape[0] != args.action_dim:
                            raise ValueError(f"Policy returned action shape {action.shape}, expected {args.action_dim}")

                        state = extract_state(raw_obs)
                        rows.append(
                            {
                                "episode_index": episode_index,
                                "index": global_index,
                                "frame_index": frame_index,
                                "task_index": task_index,
                                "timestamp": np.float32(frame_index / args.fps),
                                "action": action,
                                "observation.state": state,
                            }
                        )
                        for key in VIDEO_KEYS:
                            frames_by_key[key].append(obs[key])

                        raw_obs, _, done, _ = env.step(action)
                        obs = extract_obs(raw_obs, args.image_size)
                        global_index += 1
                        if done and not args.keep_going_after_success:
                            break

                    if len(rows) < args.min_steps:
                        print(f"[WARN] Skip short episode {episode_index} with {len(rows)} frames")
                        continue

                    df = pd.DataFrame(rows)
                    parquet_path = args.output_dir / "data/chunk-000" / f"episode_{episode_index:06d}.parquet"
                    df.to_parquet(parquet_path, index=False)
                    save_videos(args.output_dir / "videos", episode_index, frames_by_key, args.fps)

                    length = len(rows)
                    episodes.append(
                        {
                            "episode_index": episode_index,
                            "tasks": [prompt],
                            "length": length,
                            "action_config": [
                                {
                                    "start_frame": 0,
                                    "end_frame": length,
                                    "action_text": prompt,
                                    "skill": "",
                                }
                            ],
                        }
                    )
                    stats = {
                        "episode_index": numeric_stats(df["episode_index"].to_numpy()),
                        "index": numeric_stats(df["index"].to_numpy()),
                        "frame_index": numeric_stats(df["frame_index"].to_numpy()),
                        "task_index": numeric_stats(df["task_index"].to_numpy()),
                        "timestamp": numeric_stats(df["timestamp"].to_numpy()),
                        "action": numeric_stats(np.stack(df["action"].to_numpy())),
                        "observation.state": numeric_stats(np.stack(df["observation.state"].to_numpy())),
                    }
                    for key, frames in frames_by_key.items():
                        stats[key] = image_stats(frames)
                    episode_stats.append({"episode_index": episode_index, "stats": stats})
                    episode_index += 1
            finally:
                env.close()
    finally:
        provider.close()

    if not episodes:
        raise RuntimeError("No episodes were collected.")

    write_metadata(
        args.output_dir,
        episodes=episodes,
        episode_stats=episode_stats,
        tasks=task_names,
        fps=args.fps,
        image_size=args.image_size,
        action_dim=args.action_dim,
        state_dim=len(rows[0]["observation.state"]),
    )
    print(f"[DONE] Wrote {len(episodes)} episodes to {args.output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-benchmark", default="libero_10", choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--task-start", type=int, default=0)
    parser.add_argument("--task-end", type=int, default=None)
    parser.add_argument("--episodes-per-task", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=800)
    parser.add_argument("--min-steps", type=int, default=8)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--init-state-offset", type=int, default=0)
    parser.add_argument("--keep-going-after-success", action="store_true")
    parser.add_argument("--policy", default="zero", choices=["zero", "random", "python", "websocket"])
    parser.add_argument("--policy-file", type=Path, default=None)
    parser.add_argument("--port", type=int, default=23908)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--random-scale", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    collect(parse_args())
