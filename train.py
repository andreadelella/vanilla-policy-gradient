import json
import time

import gymnasium as gym
import numpy as np
import torch

from data_collection import collect_parallel_trajectories
from policy import GaussianPolicy
from gpomdp import compute_gpomdp_loss


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


def make_env(env_id: str, seed: int | None):
    def thunk():
        env = gym.make(env_id)
        if seed is not None:
            env.reset(seed=seed)
        return env

    return thunk


def evaluate_policy(env, policy, n_episodes=5, seed=None):
    episode_rewards = []

    for i in range(n_episodes):
        eval_seed = None if seed is None else seed + i

        state, _ = env.reset(seed=eval_seed)
        done = False
        total_reward = 0.0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)

            with torch.no_grad():
                action = policy.sample_action(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            total_reward += reward
            state = next_state

        episode_rewards.append(total_reward)

    return float(np.mean(episode_rewards))


def main(config_path="config.json"):
    cfg = load_config(config_path)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    env = gym.make(cfg["env_id"])
    eval_env = gym.make(cfg["env_id"])

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = GaussianPolicy(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_sizes=tuple(cfg["hidden_sizes"]),
        init_log_std=cfg["init_log_std"],
        learn_std=cfg["learn_std"],
    )

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])

    env_fns = [
        make_env(cfg["env_id"], cfg["seed"] + i)
        for i in range(cfg["n_envs"])
    ]

    train_envs = gym.vector.AsyncVectorEnv(env_fns)

    training_rewards = []
    evaluation_rewards = []

    try:
        for iteration in range(cfg["n_iterations"]):
            t0 = time.perf_counter()
            trajectories = collect_parallel_trajectories(
                envs=train_envs,
                policy=policy,
            )

            t1 = time.perf_counter()

            batch_reward = float(np.mean([
                sum(traj.rewards) for traj in trajectories
            ]))

            debug = iteration == 0

            optimizer.zero_grad()

            loss = compute_gpomdp_loss(
                policy=policy,
                trajectories=trajectories,
                gamma=cfg["gamma"],
                normalize_returns=cfg["normalize_returns"],
                debug=debug,
            )

            loss.backward()
            optimizer.step()

            t2 = time.perf_counter()
            rollout_time = t1 - t0
            update_time = t2 - t1
            iteration_time = t2 - t0

            n_steps = sum(len(traj.rewards) for traj in trajectories)
            samples_per_sec = n_steps / iteration_time

            training_rewards.append(batch_reward)

            if iteration % cfg["eval_every"] == 0:
                eval_reward = evaluate_policy(
                    env=eval_env,
                    policy=policy,
                    n_episodes=cfg["n_eval_episodes"],
                    seed=cfg["seed"] + 10_000 + iteration,
                )

                evaluation_rewards.append(eval_reward)

                print(
                    f"Iteration {iteration:04d} | "
                    f"train reward: {batch_reward:.2f} | "
                    f"eval reward: {eval_reward:.2f} | "
                    f"rollout: {rollout_time:.3f}s | "
                    f"update: {update_time:.3f}s | "
                    f"total: {iteration_time:.3f}s | "
                    f"samples/s: {samples_per_sec:.0f}"
                )

    finally:
        train_envs.close()
        env.close()
        eval_env.close()

    return policy, training_rewards, evaluation_rewards


if __name__ == "__main__":
    main()