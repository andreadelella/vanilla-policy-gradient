"""Deterministic rollout collection for frozen-policy Fisher experiments."""

from __future__ import annotations

import multiprocessing
from dataclasses import dataclass
from typing import Sequence

import gymnasium as gym
import numpy as np
import torch


@dataclass(frozen=True)
class RolloutBatch:
    states: np.ndarray
    actions: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    invalid_sample_count: int


def policy_seed(base_seed: int, width: int) -> int:
    return int((base_seed + 10_007 * width) % (2**63 - 1))


def environment_seed(
    base_seed: int,
    iteration: int,
    trajectory_index: int,
    env_index: int,
    trajectories_per_env: int,
    n_envs: int,
) -> int:
    episode_index = (
        (iteration * trajectories_per_env + trajectory_index) * n_envs
        + env_index
    )
    return int((base_seed + 1_000_003 + episode_index) % (2**32))


def action_seed(base_seed: int, width: int, iteration: int) -> int:
    return int(
        (base_seed + 2_000_003 + 100_003 * width + iteration) % (2**63 - 1)
    )


def make_env_factory(env_id: str):
    """Return a picklable environment factory for ``AsyncVectorEnv``."""

    def make_env():
        return gym.make(env_id)

    return make_env


def multiprocessing_context() -> str | None:
    """Prefer memory-efficient fork workers where the platform supports them."""

    return "fork" if "fork" in multiprocessing.get_all_start_methods() else None


def _sample_policy_actions(
    policy: torch.nn.Module,
    observations: Sequence[np.ndarray],
    generator: torch.Generator,
) -> np.ndarray:
    parameter = next(policy.parameters())
    states = torch.as_tensor(
        np.asarray(observations),
        dtype=parameter.dtype,
        device=parameter.device,
    )
    with torch.no_grad():
        output = policy(states)
        if isinstance(output, tuple):
            mean, std = output
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            actions = mean + std * noise
        else:
            probabilities = torch.softmax(output, dim=-1)
            actions = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            ).squeeze(-1)
    return actions.detach().cpu().numpy()


def collect_fixed_policy_batch(
    policy: torch.nn.Module,
    envs: gym.vector.VectorEnv,
    *,
    iteration: int,
    trajectories_per_env: int,
    horizon: int,
    base_seed: int,
    width: int,
) -> RolloutBatch:
    """Collect seeded trajectories without modifying the fixed policy.

    Raw policy samples are retained for score calculation. Only the actions sent
    to bounded continuous environments are clipped.
    """

    action_generator = torch.Generator(device="cpu")
    action_generator.manual_seed(action_seed(base_seed, width, iteration))

    collected_states: list[np.ndarray] = []
    collected_actions: list[np.ndarray] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    invalid_sample_count = 0
    n_envs = envs.num_envs
    action_space = envs.single_action_space
    continuous_actions = isinstance(action_space, gym.spaces.Box)

    for trajectory_index in range(trajectories_per_env):
        env_seeds = [
            environment_seed(
                base_seed,
                iteration,
                trajectory_index,
                env_index,
                trajectories_per_env,
                n_envs,
            )
            for env_index in range(n_envs)
        ]
        observations, _ = envs.reset(seed=env_seeds)
        active = np.ones(n_envs, dtype=bool)
        running_returns = np.zeros(n_envs, dtype=np.float64)
        running_lengths = np.zeros(n_envs, dtype=np.int64)

        if continuous_actions:
            full_actions = np.zeros(
                (n_envs, *action_space.shape),
                dtype=action_space.dtype,
            )
        else:
            full_actions = np.zeros(n_envs, dtype=np.int64)

        while np.any(active):
            active_indices = np.flatnonzero(active)
            active_observations = observations[active_indices]
            raw_actions = _sample_policy_actions(
                policy,
                active_observations,
                action_generator,
            )

            finite_states = np.all(
                np.isfinite(active_observations),
                axis=tuple(range(1, active_observations.ndim)),
            )
            finite_actions = (
                np.isfinite(raw_actions)
                if raw_actions.ndim == 1
                else np.all(
                    np.isfinite(raw_actions),
                    axis=tuple(range(1, raw_actions.ndim)),
                )
            )
            valid_samples = finite_states & finite_actions
            if np.any(valid_samples):
                collected_states.append(
                    np.asarray(
                        active_observations[valid_samples],
                        dtype=np.float64,
                    ).copy()
                )
                collected_actions.append(
                    np.asarray(
                        raw_actions[valid_samples],
                        dtype=np.float64 if continuous_actions else np.int64,
                    ).copy()
                )
            invalid_sample_count += int(np.count_nonzero(~valid_samples))

            full_actions[...] = 0
            if continuous_actions:
                full_actions[active_indices] = np.clip(
                    raw_actions,
                    action_space.low,
                    action_space.high,
                ).astype(action_space.dtype, copy=False)
            else:
                full_actions[active_indices] = raw_actions.astype(
                    np.int64,
                    copy=False,
                )

            next_observations, rewards, terminated, truncated, _ = envs.step(
                full_actions
            )
            running_returns[active_indices] += rewards[active_indices]
            running_lengths[active_indices] += 1

            reached_horizon = np.zeros(n_envs, dtype=bool)
            if horizon > 0:
                reached_horizon = running_lengths >= horizon
            finished = active & (terminated | truncated | reached_horizon)
            if np.any(finished):
                episode_returns.extend(running_returns[finished].tolist())
                episode_lengths.extend(running_lengths[finished].tolist())
                active[finished] = False
            observations = next_observations

    if not collected_states:
        raise RuntimeError("rollout collection produced no finite samples")

    action_dtype = np.float64 if continuous_actions else np.int64
    return RolloutBatch(
        states=np.concatenate(collected_states, axis=0, dtype=np.float64),
        actions=np.concatenate(collected_actions, axis=0, dtype=action_dtype),
        episode_returns=np.asarray(episode_returns, dtype=np.float64),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int64),
        invalid_sample_count=invalid_sample_count,
    )
