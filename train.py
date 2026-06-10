import json
import os
import time

import gymnasium as gym
from gymnasium.spaces import Box, Discrete
import matplotlib.pyplot as plt
import numpy as np
import torch

from utils import plot_training_curves, record_policy_video
from data_collection import collect_parallel_trajectories
from policy import GaussianPolicy, MLPSoftmaxPolicy
from gpomdp import compute_gpomdp_loss


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


def make_env(env_id: str, seed: int | None, horizon: int | None = None):
    def thunk():
        env = gym.make(env_id)

        if horizon is not None and horizon > 0:
            env = gym.wrappers.TimeLimit(
                env,
                max_episode_steps=horizon,
            )

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


def run_single_training(config_path="config.json"):
    cfg = load_config(config_path)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    env = gym.make(cfg["env_id"])
    eval_env = gym.make(cfg["env_id"])

    state_dim = env.observation_space.shape[0]

    if isinstance(env.action_space, Box):
        action_dim = env.action_space.shape[0]

        policy = GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_sizes=tuple(cfg["hidden_sizes"]),
            init_log_std=cfg["init_log_std"],
            learn_std=cfg["learn_std"],
        )

    elif isinstance(env.action_space, Discrete):
        action_dim = env.action_space.n

        policy = MLPSoftmaxPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=cfg.get("hidden_dim", 32),
        )

    else:
        raise ValueError(f"Unsupported action space: {env.action_space}")

    optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])

    env_fns = [
        make_env(
            env_id=cfg["env_id"],
            seed=cfg["seed"] + i,
            horizon=cfg.get("horizon", None),
        )
        for i in range(cfg["n_envs"])
    ]

    train_envs = gym.vector.AsyncVectorEnv(env_fns)

    training_rewards = []
    evaluation_rewards = []

    training_start = time.perf_counter()

    try:
        for iteration in range(cfg["n_iterations"]):
            t0 = time.perf_counter()

            trajectories = collect_parallel_trajectories(
                envs=train_envs,
                policy=policy,
                n_trajectories_per_env=cfg["n_trajectories"],
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
                center_returns=cfg["center_returns"],
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
                eval_start = time.perf_counter()

                eval_reward = evaluate_policy(
                    env=eval_env,
                    policy=policy,
                    n_episodes=cfg["n_eval_episodes"],
                    seed=cfg["seed"] + 10_000 + iteration,
                )

                eval_time = time.perf_counter() - eval_start
                evaluation_rewards.append(eval_reward)

                total_time = iteration_time + eval_time

                print(
                    f"Iteration {iteration:04d} | "
                    f"train reward: {batch_reward:.2f} | "
                    f"eval reward: {eval_reward:.2f} | "
                    f"eval time: {eval_time:.3f}s | "
                    f"rollout: {rollout_time:.3f}s | "
                    f"update: {update_time:.3f}s | "
                    f"total: {total_time:.3f}s | "
                    f"samples/s: {samples_per_sec:.0f}"
                )

    finally:
        train_envs.close()
        env.close()
        eval_env.close()

    training_time = time.perf_counter() - training_start
    print(f"training time {training_time:.2f}s")

    if cfg.get("save_plots", True):
        plot_training_curves(
            training_rewards=training_rewards,
            evaluation_rewards=evaluation_rewards,
            eval_every=cfg["eval_every"],
        )

    if cfg.get("record_video", False):
        record_policy_video(
            env_id=cfg["env_id"],
            policy=policy,
            video_dir="videos",
            seed=cfg["seed"] + 20_000,
        )

    return policy, training_rewards, evaluation_rewards


def mean_confidence_interval(data, confidence_z=1.96):
    data = np.asarray(data, dtype=np.float32)

    mean = data.mean(axis=0)
    std = data.std(axis=0, ddof=1)
    sem = std / np.sqrt(data.shape[0])

    lower = mean - confidence_z * sem
    upper = mean + confidence_z * sem

    return mean, lower, upper


def plot_ci(curves, title, ylabel, xlabel, save_path, x_values=None):
    mean, lower, upper = mean_confidence_interval(curves)

    if x_values is None:
        x_values = np.arange(len(mean))

    plt.figure()
    plt.plot(x_values, mean, label="Mean")
    plt.fill_between(x_values, lower, upper, alpha=0.25, label="95% CI")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()


def run_multiseed(config_path="config.json"):
    base_cfg = load_config(config_path)
    seeds = base_cfg.get("seeds", [base_cfg["seed"]])

    os.makedirs("multiseed_results", exist_ok=True)

    all_training_rewards = []
    all_evaluation_rewards = []

    for seed in seeds:
        print(f"\n========== Running seed {seed} ==========")

        cfg = dict(base_cfg)
        cfg["seed"] = seed
        cfg["record_video"] = False

        temp_config_path = f"multiseed_results/config_seed_{seed}.json"

        with open(temp_config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        _, training_rewards, evaluation_rewards = run_single_training(temp_config_path)

        all_training_rewards.append(training_rewards)
        all_evaluation_rewards.append(evaluation_rewards)

    all_training_rewards = np.asarray(all_training_rewards, dtype=np.float32)
    all_evaluation_rewards = np.asarray(all_evaluation_rewards, dtype=np.float32)

    np.save("multiseed_results/training_rewards.npy", all_training_rewards)
    np.save("multiseed_results/evaluation_rewards.npy", all_evaluation_rewards)

    plot_ci(
        curves=all_training_rewards,
        title="Training reward across seeds",
        ylabel="Average training return",
        xlabel="Iteration",
        save_path="multiseed_results/training_rewards_ci.png",
    )

    eval_every = base_cfg["eval_every"]
    eval_x = np.arange(all_evaluation_rewards.shape[1]) * eval_every

    plot_ci(
        curves=all_evaluation_rewards,
        title="Evaluation reward across seeds",
        ylabel="Average evaluation return",
        xlabel="Iteration",
        save_path="multiseed_results/evaluation_rewards_ci.png",
        x_values=eval_x,
    )


def main(config_path="config.json"):
    cfg = load_config(config_path)
    run_mode = cfg.get("run_mode", "single")

    if run_mode == "single":
        return run_single_training(config_path)

    if run_mode == "multiseed":
        return run_multiseed(config_path)

    raise ValueError(f"Unknown run_mode: {run_mode}")


if __name__ == "__main__":
    main()