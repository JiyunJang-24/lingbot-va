#!/usr/bin/env python3
"""Create timestep-level action language segments for a LingBot-VA dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ACTION_NAMES = ["x", "y", "z", "roll", "pitch", "yaw", "gripper"]


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def quantize_abs(value: float, unit: float) -> int:
    return int(round(abs(value) * unit))


def action_to_language(
    action: np.ndarray,
    *,
    translation_unit_cm: float,
    rotation_unit_deg: float,
    min_translation_cm: int,
    min_rotation_deg: int,
) -> str:
    parts: list[str] = []
    x, y, z, roll, pitch, yaw, gripper = [float(v) for v in action[:7]]

    x_cm = quantize_abs(x, translation_unit_cm)
    if x_cm >= min_translation_cm:
        direction = "forward" if x > 0 else "backward"
        parts.append(f"move {direction} {x_cm}cm")

    y_cm = quantize_abs(y, translation_unit_cm)
    if y_cm >= min_translation_cm:
        direction = "right" if y > 0 else "left"
        parts.append(f"move {direction} {y_cm}cm")

    z_cm = quantize_abs(z, translation_unit_cm)
    if z_cm >= min_translation_cm:
        direction = "up" if z > 0 else "down"
        parts.append(f"move {direction} {z_cm}cm")

    roll_deg = quantize_abs(roll, rotation_unit_deg)
    if roll_deg >= min_rotation_deg:
        direction = "backward" if roll > 0 else "forward"
        parts.append(f"tilt {direction} {roll_deg} degrees")

    pitch_deg = quantize_abs(pitch, rotation_unit_deg)
    if pitch_deg >= min_rotation_deg:
        direction = "right" if pitch > 0 else "left"
        parts.append(f"tilt {direction} {pitch_deg} degrees")

    yaw_deg = quantize_abs(yaw, rotation_unit_deg)
    if yaw_deg >= min_rotation_deg:
        direction = "clockwise" if yaw < 0 else "counterclockwise"
        parts.append(f"rotate {direction} {yaw_deg} degrees")

    if gripper > 0:
        parts.append("close gripper")
    elif gripper < 0:
        parts.append("open gripper")

    if not parts:
        parts.append("hold position")
    return ", ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--segment-frames", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--translation-unit-cm", type=float, default=100.0)
    parser.add_argument("--rotation-unit-deg", type=float, default=100.0)
    parser.add_argument("--min-translation-cm", type=int, default=1)
    parser.add_argument("--min-rotation-deg", type=int, default=1)
    parser.add_argument("--preview-lines", type=int, default=12)
    args = parser.parse_args()

    if args.segment_frames < 2:
        raise ValueError("--segment-frames must be >= 2 for LingBot-VA action alignment")

    dataset_dir = args.dataset_dir.resolve()
    episodes = read_jsonl(dataset_dir / "meta/episodes.jsonl")
    preview_rows = []

    for ep in episodes:
        ep_idx = int(ep["episode_index"])
        parquet = dataset_dir / "data" / f"chunk-{ep_idx // 1000:03d}" / f"episode_{ep_idx:06d}.parquet"
        df = pd.read_parquet(parquet, columns=["action"])
        actions = np.stack(df["action"].to_numpy()).astype(np.float32)
        length = int(ep["length"])
        segments = []
        last_start = max(0, length - args.segment_frames)

        for start in range(0, max(1, length - args.segment_frames + 1), args.stride):
            end = min(length, start + args.segment_frames)
            if end - start < 2:
                continue
            text = action_to_language(
                actions[start],
                translation_unit_cm=args.translation_unit_cm,
                rotation_unit_deg=args.rotation_unit_deg,
                min_translation_cm=args.min_translation_cm,
                min_rotation_deg=args.min_rotation_deg,
            )
            segments.append(
                {
                    "start_frame": int(start),
                    "end_frame": int(end),
                    "action_text": text,
                    "skill": "action_language",
                    "source_action_frame": int(start),
                    "source_action": actions[start].astype(float).tolist(),
                }
            )
            if len(preview_rows) < args.preview_lines:
                preview_rows.append({"episode_index": ep_idx, "frame": start, "action_text": text})

        if not segments and length >= 2:
            text = action_to_language(
                actions[0],
                translation_unit_cm=args.translation_unit_cm,
                rotation_unit_deg=args.rotation_unit_deg,
                min_translation_cm=args.min_translation_cm,
                min_rotation_deg=args.min_rotation_deg,
            )
            segments.append(
                {
                    "start_frame": 0,
                    "end_frame": length,
                    "action_text": text,
                    "skill": "action_language",
                    "source_action_frame": 0,
                    "source_action": actions[0].astype(float).tolist(),
                }
            )

        ep["action_config"] = segments

    write_jsonl(dataset_dir / "meta/episodes.jsonl", episodes)
    (dataset_dir / "meta/action_language_preview.json").write_text(
        json.dumps(preview_rows, indent=2) + "\n"
    )

    print(f"[DONE] Wrote timestep action-language segments to {dataset_dir / 'meta/episodes.jsonl'}")
    print(f"[DONE] Preview: {dataset_dir / 'meta/action_language_preview.json'}")


if __name__ == "__main__":
    main()
