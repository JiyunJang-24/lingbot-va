#!/usr/bin/env python3
"""Action provider helpers for LIBERO dataset collection."""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np


class ActionProvider:
    def reset(self, *, prompt: str, task_idx: int, episode_idx: int) -> None:
        return None

    def act(self, context: dict[str, Any]) -> np.ndarray:
        raise NotImplementedError

    def close(self) -> None:
        return None


@dataclass
class ZeroActionProvider(ActionProvider):
    action_dim: int = 7

    def act(self, context: dict[str, Any]) -> np.ndarray:
        return np.zeros(self.action_dim, dtype=np.float32)


@dataclass
class RandomActionProvider(ActionProvider):
    action_dim: int = 7
    scale: float = 0.05
    seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    def act(self, context: dict[str, Any]) -> np.ndarray:
        action = self.rng.uniform(-self.scale, self.scale, size=self.action_dim)
        return action.astype(np.float32)


class PythonActionProvider(ActionProvider):
    """Wrap a user Python policy module.

    Supported module shapes:
    - make_policy(**kwargs) -> object with act(context) and optional reset/close
    - policy object with act(context) or callable(context)
    - act(context) function
    """

    def __init__(self, policy_file: Path, policy_kwargs: dict[str, Any] | None = None):
        self.policy_file = Path(policy_file)
        self.policy_kwargs = policy_kwargs or {}
        spec = importlib.util.spec_from_file_location("libero_user_policy", self.policy_file)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import policy file: {self.policy_file}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        if hasattr(module, "make_policy"):
            self.policy = module.make_policy(**self.policy_kwargs)
        elif hasattr(module, "policy"):
            self.policy = module.policy
        elif hasattr(module, "act"):
            self.policy = module.act
        else:
            raise AttributeError(
                f"{self.policy_file} must define make_policy, policy, or act(context)."
            )

    def reset(self, *, prompt: str, task_idx: int, episode_idx: int) -> None:
        reset = getattr(self.policy, "reset", None)
        if callable(reset):
            reset(prompt=prompt, task_idx=task_idx, episode_idx=episode_idx)

    def act(self, context: dict[str, Any]) -> np.ndarray:
        if callable(self.policy) and not hasattr(self.policy, "act"):
            action = self.policy(context)
        else:
            action = self.policy.act(context)
        return np.asarray(action, dtype=np.float32)

    def close(self) -> None:
        close = getattr(self.policy, "close", None)
        if callable(close):
            close()


class WebsocketActionProvider(ActionProvider):
    def __init__(self, port: int):
        from wan_va.utils.Simple_Remote_Infer.deploy.websocket_client_policy import (
            WebsocketClientPolicy,
        )

        self.client = WebsocketClientPolicy(port=port)
        self.prompt = ""

    def reset(self, *, prompt: str, task_idx: int, episode_idx: int) -> None:
        self.prompt = prompt
        self.client.infer(dict(reset=True, prompt=prompt))

    def act(self, context: dict[str, Any]) -> np.ndarray:
        ret = self.client.infer(dict(obs=context["obs"], prompt=self.prompt))
        action = np.asarray(ret["action"], dtype=np.float32)
        return action.reshape(-1, action.shape[-1])[0]


def make_action_provider(
    name: str,
    *,
    policy_file: Path | None = None,
    policy_kwargs: dict[str, Any] | None = None,
    port: int = 23908,
    action_dim: int = 7,
    random_scale: float = 0.05,
    seed: int = 0,
) -> ActionProvider:
    if name == "zero":
        return ZeroActionProvider(action_dim=action_dim)
    if name == "random":
        return RandomActionProvider(action_dim=action_dim, scale=random_scale, seed=seed)
    if name == "python":
        if policy_file is None:
            raise ValueError("--policy-file is required when --policy python")
        return PythonActionProvider(policy_file=policy_file, policy_kwargs=policy_kwargs)
    if name == "websocket":
        return WebsocketActionProvider(port=port)
    raise ValueError(f"Unknown policy provider: {name}")
