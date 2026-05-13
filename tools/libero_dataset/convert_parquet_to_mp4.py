#!/usr/bin/env python3
"""Convert parquet-embedded LIBERO image frames to LingBot-VA compatible mp4 files.

Source format (LeRobot v2.1 with inline images):
  <sub_version>/
    data/chunk-XXX/episode_NNNNNN.parquet  -- contains observation.image.bytes (PNG)
                                              and observation.wrist_image.bytes (PNG)
    meta/info.json                          -- declares observation.image as dtype=image

Output added in-place (per sub-version):
  videos/chunk-XXX/observation.images.agentview_rgb/episode_NNNNNN.mp4
  videos/chunk-XXX/observation.images.eye_in_hand_rgb/episode_NNNNNN.mp4
  meta/info.json                            -- patched with new dtype=video features

Encoding matches tools/libero_dataset/collect_libero_lingbot_dataset.py to keep
the input distribution consistent with lingbot-va-base pretraining.
"""

from __future__ import annotations

import argparse
import io
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
from tqdm import tqdm


IMAGE_KEY_OUT = "observation.images.agentview_rgb"
WRIST_KEY_OUT = "observation.images.eye_in_hand_rgb"


def decode_frame(struct_value: dict, target_size: int) -> np.ndarray:
    raw_bytes = struct_value["bytes"]
    img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
    frame = np.asarray(img, dtype=np.uint8)
    if frame.shape[0] != target_size or frame.shape[1] != target_size:
        frame = cv2.resize(frame, (target_size, target_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(frame)


def decode_column(table: pq.Table, column: str, target_size: int) -> np.ndarray:
    pylist = table[column].to_pylist()
    frames = [decode_frame(item, target_size) for item in pylist]
    return np.stack(frames, axis=0)


def write_video(out_path: Path, frames: np.ndarray, fps: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(out_path, list(frames), fps=fps, macro_block_size=1)


def patch_info(info_path: Path, target_size: int, fps: int, total_episodes: int) -> None:
    with info_path.open() as f:
        info = json.load(f)

    video_feature = lambda: {
        "dtype": "video",
        "shape": [target_size, target_size, 3],
        "names": ["height", "width", "rgb"],
        "info": {
            "video.height": target_size,
            "video.width": target_size,
            "video.codec": "h264",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": 3,
            "has_audio": False,
        },
    }

    info["features"][IMAGE_KEY_OUT] = video_feature()
    info["features"][WRIST_KEY_OUT] = video_feature()
    info["total_videos"] = total_episodes * 2
    if "video_path" not in info:
        info["video_path"] = "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"

    info_path.write_text(json.dumps(info, indent=2) + "\n")


def _process_one_episode(table: pq.Table, agent_path: Path, wrist_path: Path,
                          image_col: str, wrist_col: str, target_size: int,
                          fps: int, overwrite: bool) -> None:
    if not overwrite and agent_path.is_file() and wrist_path.is_file():
        return
    if overwrite or not agent_path.is_file():
        agent_frames = decode_column(table, image_col, target_size)
        write_video(agent_path, agent_frames, fps)
    if overwrite or not wrist_path.is_file():
        wrist_frames = decode_column(table, wrist_col, target_size)
        write_video(wrist_path, wrist_frames, fps)


def convert_sub_version(sub_dir: Path, target_size: int, fps_override: int | None, overwrite: bool) -> None:
    info_path = sub_dir / "meta" / "info.json"
    if not info_path.is_file():
        print(f"[SKIP] no meta/info.json under {sub_dir}")
        return

    with info_path.open() as f:
        info = json.load(f)

    fps = fps_override if fps_override is not None else int(info.get("fps", 10))
    chunks_size = int(info.get("chunks_size", 1000))

    image_col = "observation.image" if "observation.image" in info["features"] else None
    wrist_col = "observation.wrist_image" if "observation.wrist_image" in info["features"] else None
    if image_col is None or wrist_col is None:
        raise RuntimeError(f"{sub_dir}: missing observation.image / observation.wrist_image in info.json features")

    # Detect format:
    #   (A) regular: data/chunk-XXX/episode_NNNNNN.parquet (1 file per episode)
    #   (B) sharded: data/chunk-XXX/file-NNN.parquet (multiple episodes per file, grouped by episode_index)
    episode_parquet_files = sorted((sub_dir / "data").glob("chunk-*/episode_*.parquet"))
    sharded_parquet_files = sorted((sub_dir / "data").glob("chunk-*/file-*.parquet"))

    if episode_parquet_files:
        format_name = "per-episode"
        total_episodes = int(info.get("total_episodes", len(episode_parquet_files)))
        print(f"[INFO] {sub_dir.name} [{format_name}]: {total_episodes} episodes, fps={fps}, target_size={target_size}")

        for parquet_path in tqdm(episode_parquet_files, desc=f"{sub_dir.name}"):
            ep_idx = int(parquet_path.stem.split("_")[-1])
            chunk_idx = ep_idx // chunks_size
            agent_path = sub_dir / "videos" / f"chunk-{chunk_idx:03d}" / IMAGE_KEY_OUT / f"episode_{ep_idx:06d}.mp4"
            wrist_path = sub_dir / "videos" / f"chunk-{chunk_idx:03d}" / WRIST_KEY_OUT / f"episode_{ep_idx:06d}.mp4"
            if not overwrite and agent_path.is_file() and wrist_path.is_file():
                continue
            table = pq.read_table(parquet_path, columns=[image_col, wrist_col])
            _process_one_episode(table, agent_path, wrist_path, image_col, wrist_col, target_size, fps, overwrite)

    elif sharded_parquet_files:
        format_name = "sharded-by-episode_index"
        # Each file may contain multiple episodes; rows are episode-ordered within a file.
        # Episodes can span shard boundaries — accumulate slices per episode across ALL shards
        # before writing mp4 (previous version dropped rows from the second shard for spanning eps).
        from collections import defaultdict
        import pyarrow as pa

        total_episodes = int(info.get("total_episodes", 0))
        print(f"[INFO] {sub_dir.name} [{format_name}]: {total_episodes} episodes across {len(sharded_parquet_files)} shards, fps={fps}")

        ep_slices = defaultdict(list)  # ep_idx -> [pa.Table slice, ...]
        for shard_path in tqdm(sharded_parquet_files, desc=f"{sub_dir.name} shards"):
            full_table = pq.read_table(shard_path, columns=["episode_index", image_col, wrist_col])
            ep_indices = full_table["episode_index"].to_pylist()
            i, N = 0, len(ep_indices)
            while i < N:
                ep = int(ep_indices[i])
                j = i
                while j < N and int(ep_indices[j]) == ep:
                    j += 1
                ep_slices[ep].append(full_table.slice(i, j - i))
                i = j

        for ep_idx in tqdm(sorted(ep_slices), desc=f"{sub_dir.name} episodes"):
            chunk_idx = ep_idx // chunks_size
            agent_path = sub_dir / "videos" / f"chunk-{chunk_idx:03d}" / IMAGE_KEY_OUT / f"episode_{ep_idx:06d}.mp4"
            wrist_path = sub_dir / "videos" / f"chunk-{chunk_idx:03d}" / WRIST_KEY_OUT / f"episode_{ep_idx:06d}.mp4"
            if not overwrite and agent_path.is_file() and wrist_path.is_file():
                continue
            slices = ep_slices[ep_idx]
            ep_table = pa.concat_tables(slices) if len(slices) > 1 else slices[0]
            _process_one_episode(ep_table, agent_path, wrist_path, image_col, wrist_col, target_size, fps, overwrite)
    else:
        print(f"[SKIP] no parquet under {sub_dir / 'data'} (checked episode_*.parquet and file-*.parquet)")
        return

    patch_info(info_path, target_size, fps, total_episodes)
    print(f"[DONE] {sub_dir.name}: videos/ written, info.json patched")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True,
                        help="Parent directory containing v-*_num* sub-version directories.")
    parser.add_argument("--include-subversions", type=str, default=None,
                        help="Comma-separated sub-version names (e.g. num1,num2,num4). If omitted, all v-*_num* are processed.")
    parser.add_argument("--target-size", type=int, default=128,
                        help="Output mp4 spatial size. Matches va_libero_cfg.height/width=128.")
    parser.add_argument("--fps", type=int, default=None,
                        help="Override fps. Default uses source info.json fps (typically 10).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Re-encode mp4 even if it exists.")
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.resolve()
    if not dataset_dir.is_dir():
        raise RuntimeError(f"--dataset-dir does not exist: {dataset_dir}")

    sub_dirs = sorted(p for p in dataset_dir.glob("v-*_num*") if p.is_dir())
    if args.include_subversions:
        wanted = {f"num{name[3:]}" if name.startswith("num") else name
                  for name in args.include_subversions.split(",")}
        filtered = []
        for p in sub_dirs:
            suffix = p.name.rsplit("_", 1)[-1]  # e.g. "num1"
            if suffix in wanted:
                filtered.append(p)
        sub_dirs = filtered
    if not sub_dirs:
        raise RuntimeError(f"No matching sub-version directories under {dataset_dir}")

    print(f"[PLAN] {len(sub_dirs)} sub-versions: {[p.name for p in sub_dirs]}")
    for sub_dir in sub_dirs:
        convert_sub_version(sub_dir, args.target_size, args.fps, args.overwrite)


if __name__ == "__main__":
    main()
