import argparse
import os
import re
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from diffusers import AutoencoderKLWan
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video


def _latent_index(path):
    match = re.search(r"latents_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def decode_latents(latents_dir, model_path, device):
    latent_paths = sorted(Path(latents_dir).glob("latents_*.pt"), key=_latent_index)
    if not latent_paths:
        raise FileNotFoundError(f"No latents_*.pt files found under {latents_dir}")

    latents = [torch.load(path, map_location="cpu", weights_only=False) for path in latent_paths]
    latents = torch.cat(latents, dim=2)

    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    vae = AutoencoderKLWan.from_pretrained(os.path.join(model_path, "vae"), torch_dtype=dtype).to(device)
    video_processor = VideoProcessor(vae_scale_factor=1)

    latents = latents.to(device=device, dtype=vae.dtype)
    latents_mean = torch.tensor(vae.config.latents_mean, device=device, dtype=vae.dtype).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    latents_std = 1.0 / torch.tensor(vae.config.latents_std, device=device, dtype=vae.dtype).view(
        1, vae.config.z_dim, 1, 1, 1
    )
    latents = latents / latents_std + latents_mean

    with torch.no_grad():
        decoded = vae.decode(latents, return_dict=False)[0]
    video = video_processor.postprocess_video(decoded, output_type="np")[0]
    print(f"Decoded video shape: {video.shape}, dtype: {video.dtype}")
    return video


# def read_video(path):
#     return [np.asarray(frame).astype(np.uint8) for frame in imageio.mimread(path)]
def to_uint8_frame(frame):
    frame = np.asarray(frame)

    if frame.dtype == np.uint8:
        return frame

    frame = np.nan_to_num(frame)

    # float [0, 1] case
    if frame.max() <= 1.0:
        frame = frame * 255.0

    return np.clip(frame, 0, 255).astype(np.uint8)


def read_video(path):
    return [to_uint8_frame(frame) for frame in imageio.mimread(path)]

def resize_to_height(frame, height):
    h, w = frame.shape[:2]
    width = max(1, int(round(w * height / h)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def write_side_by_side(actual_frames, predicted_frames, output_path, fps):
    n = min(len(actual_frames), len(predicted_frames))
    if n == 0:
        raise ValueError("No frames available for comparison")

    frames = []
    for actual, predicted in zip(actual_frames[:n], predicted_frames[:n]):
        height = min(actual.shape[0], predicted.shape[0])
        actual = resize_to_height(actual, height)
        predicted = resize_to_height(predicted, height)
        frames.append(np.hstack([actual, predicted]).astype(np.uint8))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--actual-video", required=True)
    parser.add_argument("--latents-dir", required=True)
    parser.add_argument("--model-path", default="checkpoints/lingbot-va-base")
    parser.add_argument("--predicted-video", required=True)
    parser.add_argument("--comparison-video", required=True)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    predicted_frames = decode_latents(args.latents_dir, args.model_path, args.device)
    predicted_frames_uint8 = [to_uint8_frame(frame) for frame in predicted_frames]
    # export_to_video(predicted_frames, os.path.join('/data1/local/lingbot-va/temp', "demo.mp4"), fps=10)
    # TODO: 여기 아래부터 수정해야함
    Path(args.predicted_video).parent.mkdir(parents=True, exist_ok=True)
    # imageio.mimsave(args.predicted_video, [frame.astype(np.uint8) for frame in predicted_frames], fps=args.fps)
    imageio.mimsave(args.predicted_video, predicted_frames_uint8, fps=args.fps)
    actual_frames = read_video(args.actual_video)
    write_side_by_side(actual_frames, predicted_frames_uint8, args.comparison_video, args.fps)

    print(f"predicted_video={args.predicted_video}")
    print(f"comparison_video={args.comparison_video}")


if __name__ == "__main__":
    main()
