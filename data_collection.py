# collect_trajectories.py

from dataclasses import dataclass
from typing import List
import numpy as np
import torch


@dataclass
class Trajectory:
    states: List[np.ndarray]
    actions: List[int]
    rewards: List[float]
    dones: List[bool]


def collect_trajectory(env, policy, seed=None) -> Trajectory:
    states = []
    actions = []
    rewards = []
    dones = []

    state, _ = env.reset(seed=seed)
    done = False

    while not done:
        states.append(state)

        state_tensor = torch.tensor(state, dtype=torch.float32)

        with torch.no_grad():
            action = policy.sample_action(state_tensor)

        next_state, reward, terminated, truncated, _ = env.step(action)

        done = terminated or truncated

        actions.append(action)
        rewards.append(float(reward))
        dones.append(done)

        state = next_state

    return Trajectory(
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
    )


def collect_trajectories(env, policy, n_trajectories: int, seed=None) -> List[Trajectory]:
    trajectories = []

    for i in range(n_trajectories):
        trajectory_seed = None if seed is None else seed + i
        traj = collect_trajectory(env, policy, seed=trajectory_seed)
        trajectories.append(traj)

    return trajectories