#!/usr/bin/env python3
import argparse
import json
import subprocess
import time
from pathlib import Path

import cv2


def construct_single_env(env_args, offscreen_env_cls):
    retries = 5
    for attempt in range(retries):
        try:
            return offscreen_env_cls(**env_args)
        except Exception as exc:
            if attempt == retries - 1:
                raise
            print(f"[WARN] Env creation failed ({attempt + 1}/{retries}): {exc}")
            time.sleep(2)


def extract_obs(obs):
    agentview = obs["agentview_image"][::-1].copy()
    eye_in_hand = obs["robot0_eye_in_hand_image"][::-1].copy()
    return {
        "observation.images.agentview_rgb": agentview,
        "observation.images.eye_in_hand_rgb": eye_in_hand,
    }


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    obs = None
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.0] * 7)
    return extract_obs(obs)


def capture_first_scene(libero_benchmark, task_idx, test_num):
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    if task_idx >= num_tasks:
        raise ValueError(f"task_idx={task_idx} is out of range for {libero_benchmark} (num_tasks={num_tasks})")

    prompt = benchmark_instance.get_task(task_idx).language
    env_args = {
        "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
        "camera_heights": 128,
        "camera_widths": 128,
    }
    init_states = benchmark_instance.get_task_init_states(task_idx)
    episode_idx = test_num % init_states.shape[0]

    env = construct_single_env(env_args, OffScreenRenderEnv)
    try:
        first_obs = init_single_env(env, init_states[episode_idx])
    finally:
        env.close()

    return first_obs, prompt, episode_idx


def save_first_scene(obs_dict, example_dir):
    example_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for key, frame in obs_dict.items():
        out = example_dir / f"{key}.png"
        # cv2 expects BGR
        cv2.imwrite(str(out), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        saved.append(str(out))
    return saved


def run_i2va(repo_root, model_path, save_root, prompt):
    cmd = [
        "bash",
        "i2va_libero.sh",
        "--model_path",
        str(model_path),
        "--save_root",
        str(save_root),
        "--prompt",
        prompt,
    ]
    print("[INFO] Running:", " ".join(cmd))
    subprocess.run(cmd, cwd=repo_root, check=True)


def main():
    parser = argparse.ArgumentParser(description="Capture LIBERO first scene and run i2va generation")
    parser.add_argument("--benchmark", required=True, choices=["libero_goal", "libero_10", "libero_spatial", "libero_object"])
    parser.add_argument("--test-num", required=True, type=int, help="Episode index used to select init state")
    parser.add_argument("--task-idx", type=int, default=0, help="Task index inside the benchmark")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--example-dir", type=Path, default=Path(__file__).resolve().parents[2] / "example/libero")
    parser.add_argument("--model-path", type=Path, default=Path(__file__).resolve().parents[2] / "checkpoints/lingbot-va-base")
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parents[2] / "outputs/libero_i2va")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    first_obs, prompt, episode_idx = capture_first_scene(args.benchmark, args.task_idx, args.test_num)
    saved_images = save_first_scene(first_obs, args.example_dir.resolve())

    run_name = f"{args.benchmark}_task{args.task_idx}_test{args.test_num}_ep{episode_idx}_{time.strftime('%Y%m%d_%H%M%S')}"
    save_root = (args.output_root.resolve() / run_name)
    save_root.mkdir(parents=True, exist_ok=True)

    # keep a copy of first-scene images under this run folder for traceability
    first_scene_dir = save_root / "first_scene"
    first_scene_dir.mkdir(parents=True, exist_ok=True)
    for src in saved_images:
        src_path = Path(src)
        dst = first_scene_dir / src_path.name
        dst.write_bytes(src_path.read_bytes())

    run_i2va(repo_root=repo_root, model_path=args.model_path.resolve(), save_root=save_root, prompt=prompt)

    meta = {
        "benchmark": args.benchmark,
        "task_idx": args.task_idx,
        "test_num": args.test_num,
        "episode_idx_used": int(episode_idx),
        "prompt": prompt,
        "example_dir": str(args.example_dir.resolve()),
        "saved_images": saved_images,
        "save_root": str(save_root),
        "generated_video": str(save_root / "demo.mp4"),
    }
    meta_path = save_root / "run_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"[DONE] First scene captured and i2va generation completed.")
    print(f"[DONE] Video: {save_root / 'demo.mp4'}")
    print(f"[DONE] Meta:  {meta_path}")


if __name__ == "__main__":
    main()
