# collect_trajectories.py

from dataclasses import dataclass
from typing import List
import numpy as np
import torch
import gymnasium as gym


@dataclass
class Trajectory:
    states: List[np.ndarray]
    actions: List[np.ndarray]
    rewards: List[float]
    dones: List[bool]


def make_env(env_id: str, seed: int):
    def thunk():
        env = gym.make(env_id)
        env.reset(seed=seed)
        return env

    return thunk


def collect_parallel_trajectories(
    env_id: str,
    policy,
    n_envs: int,
    seed: int | None = None,
) -> List[Trajectory]:
    env_fns = [
        make_env(env_id, None if seed is None else seed + i)
        for i in range(n_envs)
    ]

    envs = gym.vector.AsyncVectorEnv(env_fns)

    states, _ = envs.reset()

    trajectories = [
        Trajectory(states=[], actions=[], rewards=[], dones=[])
        for _ in range(n_envs)
    ]

    finished = np.zeros(n_envs, dtype=bool)

    while not np.all(finished):
        active_indices = np.where(~finished)[0]

        state_tensor = torch.tensor(states, dtype=torch.float32)

        with torch.no_grad():
            raw_actions = policy.sample_action(state_tensor)
            
        env_actions = np.clip(
            raw_actions,
            envs.single_action_space.low,
            envs.single_action_space.high,
        )

        next_states, rewards, terminated, truncated, _ = envs.step(env_actions)

        dones = np.logical_or(terminated, truncated)

        for i in active_indices:
            trajectories[i].states.append(states[i].copy())
            trajectories[i].actions.append(raw_actions[i].copy())
            trajectories[i].rewards.append(float(rewards[i]))
            trajectories[i].dones.append(bool(dones[i]))

            if dones[i]:
                finished[i] = True

        states = next_states

    envs.close()

    return trajectories


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
            action = np.clip(action, env.action_space.low, env.action_space.high)

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