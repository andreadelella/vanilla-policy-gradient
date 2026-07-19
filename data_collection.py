from dataclasses import dataclass, field
from typing import List

import gymnasium as gym
import numpy as np
import torch


@dataclass
class Trajectory:
    states: List[np.ndarray]
    # Samples drawn from the policy distribution. GPOMDP and the empirical
    # Fisher must evaluate log pi(a|s) at these actions.
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]
    # Actions actually passed to the environment. For bounded continuous
    # spaces these may differ from `actions` because of clipping.
    executed_actions: List[np.ndarray] = field(default_factory=list)


def collect_parallel_trajectories(
    envs,
    policy,
    n_trajectories_per_env: int = 1,
    clip_actions: bool = True,
) -> List[Trajectory]:
    """
    Collect n_trajectories_per_env complete episodes from each environment.

    Total collected trajectories:
        n_envs * n_trajectories_per_env
    """

    all_trajectories = []

    for _ in range(n_trajectories_per_env):
        states, _ = envs.reset()

        n_envs = envs.num_envs

        trajectories = [
            Trajectory(states=[], actions=[], rewards=[], dones=[])
            for _ in range(n_envs)
        ]

        finished = np.zeros(n_envs, dtype=bool)

        if isinstance(envs.single_action_space, gym.spaces.Box):
            full_actions = np.zeros(
                (n_envs, *envs.single_action_space.shape),
                dtype=np.float32,
            )
        else:
            full_actions = np.zeros(n_envs, dtype=np.int64)

        while not np.all(finished):
            active_indices = np.where(~finished)[0]

            active_states = states[active_indices]
            state_tensor = torch.tensor(active_states, dtype=torch.float32)  # all active states in one batch

            with torch.no_grad():  # no autograd graph needed during rollouts
                raw_actions = policy.sample_action(state_tensor)

            if clip_actions and isinstance(envs.single_action_space, gym.spaces.Box):
                env_actions = np.clip(
                    raw_actions,
                    envs.single_action_space.low,
                    envs.single_action_space.high,
                )
            else:
                env_actions = raw_actions

            if isinstance(envs.single_action_space, gym.spaces.Box):
                full_actions[:] = 0.0
            else:
                full_actions[:] = 0

            # AsyncVectorEnv requires stepping all envs simultaneously.
            # Finished envs receive a dummy action (zero); their transitions are discarded below.
            full_actions[active_indices] = env_actions

            next_states, rewards, terminated, truncated, _ = envs.step(full_actions)
            dones = np.logical_or(terminated, truncated)

            for local_idx, env_idx in enumerate(active_indices):
                trajectories[env_idx].states.append(states[env_idx].copy())
                # The environment sees the bounded action, but the likelihood-ratio
                # estimator is defined over the random variable sampled from the Gaussian.
                # Using Normal.log_prob(clipped_action) would incorrectly treat boundary
                # probability mass as an ordinary Gaussian density value.
                trajectories[env_idx].actions.append(raw_actions[local_idx].copy())
                trajectories[env_idx].executed_actions.append(
                    env_actions[local_idx].copy()
                )
                trajectories[env_idx].rewards.append(float(rewards[env_idx]))
                trajectories[env_idx].dones.append(bool(dones[env_idx]))

                if dones[env_idx]:
                    finished[env_idx] = True

            # For envs that just finished, `next_states` holds the vector env's auto-reset
            # observation for a *new* episode, not a terminal state. That's fine here: those
            # indices are excluded from `active_indices` on every future iteration, so this
            # stray reset observation is written into `states` but never read again.
            states = next_states

        all_trajectories.extend(trajectories)

    return all_trajectories
