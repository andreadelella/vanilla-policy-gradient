from dataclasses import dataclass, field
from typing import List

import gymnasium as gym
import numpy as np
import torch


@dataclass
class Trajectory:
    states: List[np.ndarray]
    # Original actions sampled by the policy.
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]
    # Actions sent to the environment, possibly after clipping.
    executed_actions: List[np.ndarray] = field(default_factory=list)


def collect_parallel_trajectories(
    envs,
    policy,
    n_trajectories_per_env: int = 1,
    clip_actions: bool = True,
    device=None,
) -> List[Trajectory]:
    """Collect complete episodes from each worker in a vector environment."""

    all_trajectories = []

    # Each pass collects one episode from every environment.
    for _ in range(n_trajectories_per_env):
        states, _ = envs.reset()

        n_envs = envs.num_envs

        trajectories = [
            Trajectory(states=[], actions=[], rewards=[], dones=[])
            for _ in range(n_envs)
        ]

        # True means that worker's current episode has ended.
        finished = np.zeros(n_envs, dtype=bool)

        # AsyncVectorEnv needs one action slot for every worker.
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
            state_tensor = torch.tensor(active_states, dtype=torch.float32, device=device)

            # Rollout collection does not need gradients.
            with torch.no_grad():
                raw_actions = policy.sample_action(state_tensor)

            # Clip continuous actions only before sending them to the environment.
            if clip_actions and isinstance(envs.single_action_space, gym.spaces.Box):
                env_actions = np.clip(
                    raw_actions,
                    envs.single_action_space.low,
                    envs.single_action_space.high,
                )
            else:
                env_actions = raw_actions

            # Finished workers receive zero; their transitions are ignored.
            if isinstance(envs.single_action_space, gym.spaces.Box):
                full_actions[:] = 0.0
            else:
                full_actions[:] = 0
            full_actions[active_indices] = env_actions

            # Step every worker together.
            next_states, rewards, terminated, truncated, _ = envs.step(full_actions)
            dones = np.logical_or(terminated, truncated)

            for local_idx, env_idx in enumerate(active_indices):
                trajectories[env_idx].states.append(states[env_idx].copy())
                # Learning uses the original sampled action, not the clipped action.
                trajectories[env_idx].actions.append(raw_actions[local_idx].copy())
                trajectories[env_idx].executed_actions.append(
                    env_actions[local_idx].copy()
                )
                trajectories[env_idx].rewards.append(float(rewards[env_idx]))
                trajectories[env_idx].dones.append(bool(dones[env_idx]))

                if dones[env_idx]:
                    finished[env_idx] = True

            # Auto-reset states from finished workers are ignored on the next loop.
            states = next_states

        all_trajectories.extend(trajectories)

    return all_trajectories
