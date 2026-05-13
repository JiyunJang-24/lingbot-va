import numpy as np
from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import WebsocketClientPolicy
import argparse
from libero.libero import benchmark
import time
from libero.libero.envs import OffScreenRenderEnv
from pathlib import Path
from tqdm import tqdm
import os
import imageio
import cv2
import json

# Each entry is (dx, dy, dz) in world frame (metres).
# Episodes cycle through this list: episode i → DELTA_XYZ_LIST[i % len(...)].
DELTA_XYZ_LIST = [
    (0.00,  0.00, 0.0),   # original position
    (0.10,  0.00, 0.0),   # +x
    (-0.10, 0.00, 0.0),   # -x
    # (0.00,  0.10, 0.0),   # +y
    (0.00, -0.10, 0.0),   # -y
    # (0.10,  0.10, 0.0),   # +x+y
    # (-0.10, 0.10, 0.0),   # -x+y
    (0.10, -0.10, 0.0),   # +x-y
    (-0.10,-0.10, 0.0),   # -x-y
    (0.15,  0.00, 0.0),   # far +x
    (0.15, -0.10, 0.0),   # far +x -y
    (0.15, -0.15, 0.0),   # far +x -y more
    (0.25, 0.0, 0.0),     # far +x
    (0.25, -0.15, 0.0),   # far +x -y
]


# ---------------------------------------------------------------------------
# Inlined from LIBERO/xyg_scripts/rotate_recolor_dataset.py so we avoid
# importing the full module (which pulls in lerobot-VAI internals).
# ---------------------------------------------------------------------------

def _get_mujoco_sim_holder(env):
    if hasattr(env, "sim"):
        return env
    if hasattr(env, "envs") and len(env.envs) > 0:
        e0 = env.envs[0]
        if hasattr(e0, "sim"):
            return e0
        if hasattr(e0, "_env") and hasattr(e0._env, "sim"):
            return e0._env
    raise AttributeError("Cannot find mujoco sim. Expected env.sim or env.envs[0].sim")


def _body_depth(model, body_id):
    depth, cur = 0, body_id
    while cur != 0:
        cur = int(model.body_parentid[cur])
        depth += 1
        if depth > 10_000:
            break
    return depth


def reposition_object(env, object_name, delta_xyz):
    """Move object by (dx, dy, dz) in world frame. Returns abs_xyz after move."""
    sim_env = _get_mujoco_sim_holder(env)
    sim = sim_env.sim
    model, data = sim.model, sim.data

    delta_w = np.asarray(delta_xyz, dtype=np.float64).reshape(3)
    sim.forward()

    body_ids = [i for i, name in enumerate(model.body_names) if name and object_name in name]
    if not body_ids:
        raise ValueError(f"No MuJoCo body containing '{object_name}' found")

    exact = [i for i in body_ids if model.body_names[i] == object_name]
    rep_ids = [exact[0]] if exact else [min(body_ids, key=lambda b: _body_depth(model, b))]

    for body_id in rep_ids:
        parent_id = int(model.body_parentid[body_id])
        if parent_id == 0:
            delta_parent = delta_w
        else:
            parent_R_w = data.xmat[parent_id].reshape(3, 3).copy()
            delta_parent = parent_R_w.T @ delta_w
        if model.body_jntnum[body_id] > 0:
            jnt_adr = int(model.body_jntadr[body_id])
            if int(model.jnt_type[jnt_adr]) == 0:  # mjJNT_FREE
                qpos_adr = int(model.jnt_qposadr[jnt_adr])
                data.qpos[qpos_adr:qpos_adr + 3] += delta_w
                model.body_pos[body_id] += delta_parent
            else:
                model.body_pos[body_id] += delta_parent
        else:
            model.body_pos[body_id] += delta_parent

    sim.forward()
    return tuple(data.xpos[rep_ids[0]].copy())


def write_json(data, path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def save_video(real_obs_list, save_path, fps=15, video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]):
    if not real_obs_list:
        print("❌ No real observation frames")
        return

    first_obs = real_obs_list[0]
    base_h, width_base = first_obs[video_names[0]].shape[:2]
    target_size = (width_base, base_h)
    
    print(f"Saving video: {len(real_obs_list)} frames...")

    final_frames = [
        np.hstack([cv2.resize(obs[name], target_size) for name in video_names]).astype(np.uint8)
        for obs in real_obs_list
    ]

    imageio.mimsave(save_path, final_frames, fps=fps)
    print(f"✅ Video saved to: {save_path}")


def construct_single_env(env_args):
    count = 0
    env = None
    env_creation = False
    while not env_creation and count < 5:
        try:
            env = OffScreenRenderEnv(**env_args)
            env_creation = True
        except Exception as e:
            print(f"Error!!!  construct env failed: {e}")
            time.sleep(5)
            count += 1
    if count >= 5:
        return None
    return env


def _extract_obs(obs):
    """
    Extract agentview and eye_in_hand images from raw env obs dict.

    Avoids torch round-trip: the env already returns uint8 numpy arrays [H, W, C].
    We just flip the vertical axis ([::-1]) and make a contiguous copy once.
    """
    agentview = np.ascontiguousarray(obs["agentview_image"][::-1])
    eye_in_hand = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1])
    return {"observation.images.agentview_rgb": agentview, "observation.images.eye_in_hand_rgb": eye_in_hand}


def init_single_env(env_in, init_state):
    env_in.reset()
    env_in.set_init_state(init_state)
    for _ in range(5):
        obs, _, _, _ = env_in.step([0.] * 7)
    return _extract_obs(obs)


def env_one_step(env_in, action):
    obs, _, done, _ = env_in.step(action)
    return _extract_obs(obs), done


def run_one(model, libero_benchmark, task_idx, out_dir, episode_idx, max_timesteps,
            task_description=None, object_name="alphabet_soup_1_main", delta_xyz_list=None):
    if delta_xyz_list is None:
        delta_xyz_list = DELTA_XYZ_LIST

    benchmark_dict = benchmark.get_benchmark_dict()
    benchmark_instance = benchmark_dict[libero_benchmark]()
    num_tasks = benchmark_instance.get_num_tasks()
    assert task_idx < num_tasks, f"Error: error id must smaller than {num_tasks}"
    original_prompt = benchmark_instance.get_task(task_idx).language
    prompt = task_description if task_description else original_prompt
    env_args = {
                "bddl_file_name": benchmark_instance.get_task_bddl_file_path(task_idx),
                "camera_heights": 128,
                "camera_widths": 128,
            }
    init_states = benchmark_instance.get_task_init_states(task_idx)

    cur_env = construct_single_env(env_args)
    first_obs = init_single_env(cur_env, init_states[episode_idx % init_states.shape[0]])

    # Reposition the target object for this episode.
    delta_xyz = delta_xyz_list[episode_idx % len(delta_xyz_list)]
    abs_xyz = reposition_object(cur_env, object_name, delta_xyz=delta_xyz)
    print(f"[episode {episode_idx}] {object_name}: delta={delta_xyz}, abs={abs_xyz}")
    # Warmup steps so physics settles after repositioning.
    for _ in range(10):
        obs, _, _, _ = cur_env.step([0.] * 7)
    first_obs = _extract_obs(obs)

    ret = model.infer(dict(reset=True, prompt=prompt))

    full_obs_list = []
    done = False
    first = True
    while cur_env.env.timestep < max_timesteps:
        ret = model.infer(dict(obs=first_obs, prompt=prompt))
        action = ret['action']

        key_frame_list = []
        assert action.shape[2] % 4 == 0
        action_per_frame = action.shape[2] // 4
        start_idx = 1 if first else 0
        for i in range(start_idx, action.shape[1]):
            for j in range(action.shape[2]):
                ee_action = action[:, i, j]
                observes, done = env_one_step(cur_env, ee_action)
                full_obs_list.append(observes)
                if done:
                    break
                if (j+1) % action_per_frame == 0:
                    key_frame_list.append(observes)
                if cur_env.env.timestep >= max_timesteps:
                    break

            if done:
                break
            if cur_env.env.timestep >= max_timesteps:
                break

        first = False

        if done or cur_env.env.timestep >= max_timesteps:
            break
        else:
            model.infer(dict(obs=key_frame_list, compute_kv_cache=True, imagine=False, state=action))

    dx, dy, dz = delta_xyz
    delta_tag = f"dx{dx:.3f}_dy{dy:.3f}_dz{dz:.3f}"
    out_dir_ep = Path(out_dir) / libero_benchmark / f"{task_idx}_{prompt.replace(' ', '_')}"
    out_dir_ep.mkdir(exist_ok=True, parents=True)

    save_video(
        real_obs_list=full_obs_list,
        save_path=out_dir_ep / f"{episode_idx}_{done}_{delta_tag}.mp4",
        fps=60,
        video_names=["observation.images.agentview_rgb", "observation.images.eye_in_hand_rgb"]
    )

    cur_env.close()
    return done


def run(libero_benchmark, port, out_dir, test_num, task_range=None, max_timesteps=800,
        task_description=None, object_name="alphabet_soup_1_main"):
    '''
        task_range: [start, end) for splitting tasks
    '''
    if task_range is None:
        benchmark_dict = benchmark.get_benchmark_dict()
        benchmark_instance = benchmark_dict[libero_benchmark]()
        num_tasks = benchmark_instance.get_num_tasks()
        progress_bar = tqdm(range(num_tasks), total=num_tasks)
    else:
        assert len(task_range) == 2, f'task_range: [start, end) for splitting tasks, however, task_range: {task_range}'
        num_tasks = task_range[1] - task_range[0]
        progress_bar = tqdm(range(task_range[0], task_range[1]), total=num_tasks)

    print(f"#################### Use benchmark: {libero_benchmark}, num_tasks: {num_tasks} #############")
    model = WebsocketClientPolicy(port=port)

    video_save_root_dict = None

    episode_list = range(test_num)
    for task_idx in progress_bar:
        if video_save_root_dict is not None and task_idx in video_save_root_dict:
            video_save_list = os.listdir(os.path.join(out_dir, libero_benchmark, video_save_root_dict[task_idx]))
            video_states = [1 for file in video_save_list if file.split('_')[1].split('.')[0] == 'True']
            succ_num = float(len(video_states))
            episode_list = range(len(video_save_list), test_num)
        else:
            succ_num = 0.

        for episode_idx in tqdm(episode_list, total=len(episode_list)):
            res_i = run_one(
                model,
                libero_benchmark,
                task_idx,
                out_dir,
                episode_idx,
                max_timesteps,
                task_description=task_description,
                object_name=object_name,
            )
            succ_num += res_i
            succ_rate = succ_num / (episode_idx + 1)
            print(f"Success rate: {succ_rate}, success num: {succ_num}, total num: {episode_idx + 1}")
            out_file = Path(out_dir) / f"{libero_benchmark}_{task_idx}.json"
            out_file.parent.mkdir(exist_ok=True, parents=True)
            write_json({
                "succ_num": succ_num,
                "total_num": episode_idx + 1.,
                "succ_rate": succ_rate,
                "task_description": task_description,
                }, out_file
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--libero-benchmark",
        type=str,
        default="libero_10",
        choices=["libero_10", "libero_goal", "libero_spatial", "libero_object"],
        help="Benchmark name",
    )
    parser.add_argument(
        "--task-range",
        type=int,
        nargs="+",
        default=[0, 10],
        help="Task range [start, end) for splitting tasks",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=23908,
        help="WebSocket port",
    )
    parser.add_argument(
        "--test-num",
        type=int,
        default=50,
        help="Number of test episodes",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="outputs/libero",
        help="Output directory for results",
    )
    parser.add_argument(
        "--max-timesteps",
        type=int,
        default=800,
        help="Maximum environment timesteps per episode",
    )
    parser.add_argument(
        "--task-description",
        type=str,
        default=None,
        help="Override the LIBERO task language prompt sent to the model",
    )
    parser.add_argument(
        "--object-name",
        type=str,
        default="alphabet_soup_1_main",
        help="MuJoCo body name of the object to reposition (default: alphabet_soup_1_main)",
    )
    args = parser.parse_args()
    run(**vars(args))
    print("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    main()
