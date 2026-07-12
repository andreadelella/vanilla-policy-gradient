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
from policy import GaussianPolicy, MLPSoftmaxPolicy, LinearSoftmaxPolicy
from gpomdp import compute_gpomdp_loss, apply_npg_preconditioning


def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


def make_env(env_id: str, seed: int | None, horizon: int | None = None):
    # AsyncVectorEnv requires a factory (thunk) rather than an env instance
    # so each worker can construct its own isolated copy.
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


def run_single_training(cfg: dict):
    output_dir = cfg.get("output_dir", "runs")
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    env = gym.make(cfg["env_id"])

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

        if cfg.get("policy", "mlp") == "linear":
            policy = LinearSoftmaxPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
            )
        else:
            policy = MLPSoftmaxPolicy(
                state_dim=state_dim,
                action_dim=action_dim,
                hidden_sizes=tuple(cfg["hidden_sizes"]),
            )

    else:
        raise ValueError(f"Unsupported action space: {env.action_space}")

    if cfg.get("use_npg", False):
        optimizer = torch.optim.SGD(policy.parameters(), lr=cfg["lr"])
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])

    env_fns = [
        make_env(
            env_id=cfg["env_id"],
            seed=cfg["seed"] + i,
            horizon=cfg.get("horizon", None),
        )
        for i in range(cfg["n_envs"])
    ]

    # Rollout collection is the dominant wall-clock cost; N workers run concurrently to amortise it.
    train_envs = gym.vector.AsyncVectorEnv(env_fns)

    training_rewards = []
    best_reward = float("-inf")
    best_state_dict = None

    training_start = time.perf_counter()

    try:
        for iteration in range(cfg["n_iterations"]):
            t0 = time.perf_counter()

            trajectories = collect_parallel_trajectories(
                envs=train_envs,
                policy=policy,
                n_trajectories_per_env=cfg["n_trajectories"],
                clip_actions=cfg.get("clip_actions", True),
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
                entropy_coeff=cfg.get("entropy_coeff", 0.0),
                debug=debug,
            )

            loss.backward()

            if cfg.get("use_npg", False):
                # Note: re-derives states/actions/mask from `trajectories` internally
                # (trajectories_to_tensors runs again here) rather than reusing the ones
                # already built inside compute_gpomdp_loss above -- redundant but not incorrect.
                apply_npg_preconditioning(
                    policy=policy,
                    trajectories=trajectories,
                    damping=cfg.get("npg_damping", 1e-2),
                    debug=debug,
                )

            optimizer.step()

            t2 = time.perf_counter()

            rollout_time = t1 - t0
            update_time = t2 - t1
            iteration_time = t2 - t0

            n_steps = sum(len(traj.rewards) for traj in trajectories)
            samples_per_sec = n_steps / iteration_time

            training_rewards.append(batch_reward)

            if batch_reward > best_reward:
                best_reward = batch_reward
                best_state_dict = {k: v.cpu().clone() for k, v in policy.state_dict().items()}

            print(
                f"Iteration {iteration:04d} | "
                f"train reward: {batch_reward:.2f} | "
                f"best: {best_reward:.2f} | "
                f"rollout: {rollout_time:.3f}s | "
                f"update: {update_time:.3f}s | "
                f"total: {iteration_time:.3f}s | "
                f"samples/s: {samples_per_sec:.0f}"
            )

    finally:
        train_envs.close()
        env.close()

    training_time = time.perf_counter() - training_start
    print(f"training time {training_time:.2f}s")

    np.save(os.path.join(output_dir, "training_rewards.npy"), np.array([training_rewards], dtype=np.float32))

    if cfg.get("save_plots", True):
        plot_training_curves(
            training_rewards=training_rewards,
            save_dir=output_dir,
        )

    if cfg.get("save_checkpoints", True):
        checkpoint_dir = os.path.join(output_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)
        scored = cfg.get("scored_checkpoints", False)
        if best_state_dict is not None:
            best_name = f"best_{best_reward:.1f}.pt" if scored else "best.pt"
            torch.save(best_state_dict, os.path.join(checkpoint_dir, best_name))
        final_score = training_rewards[-1] if training_rewards else 0.0
        final_name = f"final_{final_score:.1f}.pt" if scored else "final.pt"
        torch.save(policy.state_dict(), os.path.join(checkpoint_dir, final_name))

    if cfg.get("record_video", False):
        # Restore the best weights found during training before recording.
        if best_state_dict is not None:
            policy.load_state_dict(best_state_dict)
        record_policy_video(
            env_id=cfg["env_id"],
            policy=policy,
            video_dir=os.path.join(output_dir, "videos"),
            seed=cfg["seed"] + 20_000,
        )

    return policy, training_rewards


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


def run_multiseed(cfg: dict):
    seeds = cfg.get("seeds", [cfg["seed"]])
    output_dir = cfg.get("output_dir", "runs")
    os.makedirs(output_dir, exist_ok=True)

    all_training_rewards = []

    for seed in seeds:
        print(f"\n========== Running seed {seed} ==========")

        seed_cfg = dict(cfg)
        seed_cfg["seed"] = seed
        seed_cfg["record_video"] = False
        seed_cfg["save_plots"] = False        # CI plot is produced once after all seeds finish
        seed_cfg["save_checkpoints"] = False  # checkpoints per seed would overwrite each other

        _, seed_rewards = run_single_training(seed_cfg)

        all_training_rewards.append(seed_rewards)

    all_training_rewards = np.asarray(all_training_rewards, dtype=np.float32)

    np.save(os.path.join(output_dir, "training_rewards.npy"), all_training_rewards)

    plot_ci(
        curves=all_training_rewards,
        title="Training reward across seeds",
        ylabel="Average training return",
        xlabel="Iteration",
        save_path=os.path.join(output_dir, "training_rewards_ci.png"),
    )


def train_from_config(config_path="config.json"):
    cfg = load_config(config_path)
    run_mode = cfg.get("run_mode", "single")

    if run_mode == "single":
        return run_single_training(cfg)

    if run_mode == "multiseed":
        return run_multiseed(cfg)

    raise ValueError(f"Unknown run_mode: {run_mode}")


if __name__ == "__main__":
    train_from_config()