#!/usr/bin/env python3
"""Make a side-by-side preview video for one collected episode."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--video-keys",
        nargs="+",
        default=[
            "observation.images.agentview_rgb",
            "observation.images.eye_in_hand_rgb",
        ],
    )
    args = parser.parse_args()

    readers = []
    try:
        for key in args.video_keys:
            path = (
                args.dataset_dir
                / "videos"
                / "chunk-000"
                / key
                / f"episode_{args.episode_index:06d}.mp4"
            )
            readers.append(imageio.get_reader(path))

        args.output.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(args.output, fps=args.fps, macro_block_size=1)
        try:
            for frames in zip(*readers):
                h = min(frame.shape[0] for frame in frames)
                w = min(frame.shape[1] for frame in frames)
                resized = [
                    cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
                    for frame in frames
                ]
                writer.append_data(np.concatenate(resized, axis=1))
        finally:
            writer.close()
    finally:
        for reader in readers:
            reader.close()

    print(args.output)


if __name__ == "__main__":
    main()
