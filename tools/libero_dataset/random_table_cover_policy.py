"""Random desk-covering policy for LIBERO collection.

The policy stays simple on purpose: it samples smooth random end-effector
delta actions with stronger x/y motion than rotation so the camera/arm sweeps
over the table instead of only jittering in place.
"""

from __future__ import annotations

import numpy as np


class RandomTableCoverPolicy:
    def __init__(
        self,
        seed: int = 0,
        xy_scale: float = 0.08,
        z_scale: float = 0.025,
        rot_scale: float = 0.05,
        hold_steps: int = 8,
    ) -> None:
        self.seed = seed
        self.xy_scale = xy_scale
        self.z_scale = z_scale
        self.rot_scale = rot_scale
        self.hold_steps = hold_steps
        self.rng = np.random.default_rng(seed)
        self.step = 0
        self.current = np.zeros(7, dtype=np.float32)

    def reset(self, *, prompt: str, task_idx: int, episode_idx: int) -> None:
        self.rng = np.random.default_rng(self.seed + 1009 * episode_idx + 37 * task_idx)
        self.step = 0
        self.current = np.zeros(7, dtype=np.float32)

    def _sample_action(self) -> np.ndarray:
        action = np.zeros(7, dtype=np.float32)
        action[0] = self.rng.uniform(-self.xy_scale, self.xy_scale)
        action[1] = self.rng.uniform(-self.xy_scale, self.xy_scale)
        action[2] = self.rng.uniform(-self.z_scale, self.z_scale)
        action[3:6] = self.rng.uniform(-self.rot_scale, self.rot_scale, size=3)
        action[6] = self.rng.choice([-1.0, 1.0])
        return action

    def act(self, context: dict) -> np.ndarray:
        if self.step % self.hold_steps == 0:
            self.current = self._sample_action()

        jitter = np.array(
            [
                self.rng.normal(0.0, self.xy_scale * 0.15),
                self.rng.normal(0.0, self.xy_scale * 0.15),
                self.rng.normal(0.0, self.z_scale * 0.10),
                self.rng.normal(0.0, self.rot_scale * 0.10),
                self.rng.normal(0.0, self.rot_scale * 0.10),
                self.rng.normal(0.0, self.rot_scale * 0.10),
                0.0,
            ],
            dtype=np.float32,
        )
        self.step += 1
        return (self.current + jitter).astype(np.float32)


def make_policy(**kwargs) -> RandomTableCoverPolicy:
    return RandomTableCoverPolicy(**kwargs)
