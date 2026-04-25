#!/usr/bin/env python3
"""Extract Wan2.2 VAE latents and text embeddings for LingBot-VA datasets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch
from tqdm import tqdm


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def normalize_latents(latents: torch.Tensor, latents_mean: torch.Tensor, latents_std: torch.Tensor) -> torch.Tensor:
    latents_mean = latents_mean.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
    latents_std = latents_std.view(1, -1, 1, 1, 1).to(device=latents.device, dtype=latents.dtype)
    return ((latents.float() - latents_mean) * latents_std).to(latents)


@torch.no_grad()
def encode_text(tokenizer, text_encoder, text: str, *, device: str, dtype: torch.dtype, max_sequence_length: int) -> torch.Tensor:
    try:
        from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    except ImportError:
        prompt_clean = lambda value: value

    cleaned = prompt_clean(text)
    inputs = tokenizer(
        [cleaned],
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        add_special_tokens=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    text_device = next(text_encoder.parameters()).device
    prompt_embeds = text_encoder(
        inputs.input_ids.to(text_device),
        inputs.attention_mask.to(text_device),
    ).last_hidden_state
    seq_len = inputs.attention_mask.gt(0).sum(dim=1).long()[0].item()
    prompt_embeds = prompt_embeds[:, :seq_len].to(dtype=dtype, device=device)
    if seq_len < max_sequence_length:
        pad = prompt_embeds.new_zeros(1, max_sequence_length - seq_len, prompt_embeds.shape[-1])
        prompt_embeds = torch.cat([prompt_embeds, pad], dim=1)
    return prompt_embeds[0].cpu()


def read_video_segment(video_path: Path, start: int, end: int, height: int, width: int) -> tuple[np.ndarray, list[int], tuple[int, int]]:
    frames = []
    frame_ids = []
    reader = imageio.get_reader(video_path)
    try:
        for idx, frame in enumerate(reader):
            if idx < start:
                continue
            if idx >= end:
                break
            ori_h, ori_w = frame.shape[:2]
            if frame.shape[0] != height or frame.shape[1] != width:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            frames.append(frame)
            frame_ids.append(idx)
    finally:
        reader.close()
    if not frames:
        raise ValueError(f"No frames read from {video_path} for [{start}, {end})")
    valid_len = ((len(frames) - 1) // 4) * 4 + 1
    frames = frames[:valid_len]
    frame_ids = frame_ids[:valid_len]
    return np.stack(frames).astype(np.uint8), frame_ids, (ori_h, ori_w)


@torch.no_grad()
def encode_video(vae, frames: np.ndarray, *, device: str, dtype: torch.dtype) -> torch.Tensor:
    video = torch.from_numpy(frames).float().permute(3, 0, 1, 2).unsqueeze(0)
    video = video / 255.0 * 2.0 - 1.0
    video = video.to(device=device, dtype=dtype)
    enc_out = vae._encode(video)
    mu, _ = torch.chunk(enc_out, 2, dim=1)
    latents_mean = torch.tensor(vae.config.latents_mean, device=mu.device, dtype=mu.dtype)
    latents_std = torch.tensor(vae.config.latents_std, device=mu.device, dtype=mu.dtype)
    mu_norm = normalize_latents(mu, latents_mean, 1.0 / latents_std)
    return mu_norm[0].permute(1, 2, 3, 0).contiguous().cpu()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True, help="Directory containing vae/, text_encoder/, tokenizer/")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--max-sequence-length", type=int, default=512)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    dataset_dir = args.dataset_dir.resolve()
    model_path = args.model_path.resolve()
    from wan_va.modules import load_text_encoder, load_tokenizer, load_vae

    with (dataset_dir / "meta/info.json").open() as f:
        info = json.load(f)
    episodes = read_jsonl(dataset_dir / "meta/episodes.jsonl")
    video_keys = [k for k, v in info["features"].items() if v.get("dtype") == "video"]
    fps = int(info["fps"])

    tokenizer = load_tokenizer(model_path / "tokenizer")
    text_encoder = load_text_encoder(model_path / "text_encoder", torch_dtype=dtype, torch_device=args.device)
    vae = load_vae(model_path / "vae", torch_dtype=dtype, torch_device=args.device)

    empty_emb = encode_text(tokenizer, text_encoder, "", device="cpu", dtype=dtype, max_sequence_length=args.max_sequence_length)
    torch.save(empty_emb, dataset_dir / "empty_emb.pt")

    for ep in tqdm(episodes, desc="episodes"):
        ep_idx = int(ep["episode_index"])
        chunk = ep_idx // int(info.get("chunks_size", 1000))
        for acfg in ep["action_config"]:
            start = int(acfg["start_frame"])
            end = int(acfg["end_frame"])
            text = acfg.get("action_text") or ep.get("tasks", [""])[0]
            text_emb = encode_text(tokenizer, text_encoder, text, device="cpu", dtype=dtype, max_sequence_length=args.max_sequence_length)
            for key in video_keys:
                out_dir = dataset_dir / "latents" / f"chunk-{chunk:03d}" / key
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"episode_{ep_idx:06d}_{start}_{end}.pth"
                if out_path.exists() and not args.overwrite:
                    continue
                video_path = dataset_dir / "videos" / f"chunk-{chunk:03d}" / key / f"episode_{ep_idx:06d}.mp4"
                feature = info["features"][key]
                height = int(feature["info"]["video.height"])
                width = int(feature["info"]["video.width"])
                frames, frame_ids, (ori_h, ori_w) = read_video_segment(video_path, start, end, height, width)
                latent = encode_video(vae, frames, device=args.device, dtype=dtype)
                latent_num_frames, latent_height, latent_width, channels = latent.shape
                payload = {
                    "latent": latent.reshape(-1, channels).to(dtype),
                    "latent_num_frames": int(latent_num_frames),
                    "latent_height": int(latent_height),
                    "latent_width": int(latent_width),
                    "video_num_frames": int(frames.shape[0]),
                    "video_height": int(height),
                    "video_width": int(width),
                    "text_emb": text_emb.to(dtype),
                    "text": text,
                    "frame_ids": np.asarray(frame_ids, dtype=np.int64),
                    "start_frame": start,
                    "end_frame": end,
                    "fps": fps,
                    "ori_fps": fps,
                }
                torch.save(payload, out_path)

    print(f"[DONE] Latents written under {dataset_dir / 'latents'}")


if __name__ == "__main__":
    main()
