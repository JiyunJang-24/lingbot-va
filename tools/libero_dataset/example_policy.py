"""Tiny example policy for collect_libero_lingbot_dataset.py.

Replace the body of `act` with your own controller. The collector passes raw
LIBERO observations, extracted RGB observations, prompt, step index, and env.
"""

from __future__ import annotations

import numpy as np


class ExamplePolicy:
    def reset(self, *, prompt: str, task_idx: int, episode_idx: int) -> None:
        self.prompt = prompt

    def act(self, context: dict) -> np.ndarray:
        # LIBERO's default Franka action is [dx, dy, dz, droll, dpitch, dyaw, gripper].
        # This placeholder holds position with an open gripper.
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], dtype=np.float32)


def make_policy(**kwargs) -> ExamplePolicy:
    return ExamplePolicy()
