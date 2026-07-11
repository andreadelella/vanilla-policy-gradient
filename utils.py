import os
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
import imageio
import torch


_ALGO_STYLES = {
    "gpomdp": {"color": "steelblue",  "label": "GPOMDP (Adam)"},
    "npg":    {"color": "darkorange", "label": "NPG"},
}


def plot_comparison(rewards_dict, save_dir, env_id="", confidence_z=1.96):
    """
    rewards_dict: {algo_name: array of shape [n_seeds, n_iters] or [1, n_iters]}

    Overlays mean ± 95% CI curves for each algorithm on the same axes.
    Single-seed runs are plotted without a CI band.
    """
    plt.figure(figsize=(8, 5))

    for algo, rewards in rewards_dict.items():
        rewards = np.asarray(rewards, dtype=np.float32)
        style = _ALGO_STYLES.get(algo, {"color": None, "label": algo.upper()})
        color = style["color"]
        label = style["label"]
        x = np.arange(rewards.shape[1])

        if rewards.shape[0] == 1:
            plt.plot(x, rewards[0], label=label, color=color)
        else:
            mean = rewards.mean(axis=0)
            sem = rewards.std(axis=0, ddof=1) / np.sqrt(rewards.shape[0])
            lower = mean - confidence_z * sem
            upper = mean + confidence_z * sem
            plt.plot(x, mean, label=label, color=color)
            plt.fill_between(x, lower, upper, alpha=0.25, color=color)

    title = f"GPOMDP vs NPG — {env_id}" if env_id else "GPOMDP vs NPG"
    plt.xlabel("Iteration")
    plt.ylabel("Average training return")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "comparison.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot: {save_path}")


def plot_training_curves(training_rewards, save_dir="plots"):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(training_rewards)
    plt.xlabel("Iteration")
    plt.ylabel("Average training return")
    plt.title("Training reward")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "training_rewards.png"), dpi=300)
    plt.close()


def record_policy_video(
    env_id,
    policy,
    video_dir="videos",
    seed=23,
    n_episodes=3,
    fps=30,
):
    """Record n_episodes of the policy and save each as an MP4."""
    os.makedirs(video_dir, exist_ok=True)

    env = gym.make(env_id, render_mode="rgb_array")  # no TimeLimit: natural episode length

    for ep in range(n_episodes):
        frames = []
        state, _ = env.reset(seed=seed + ep)
        done = False

        while not done:
            frames.append(env.render())

            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                action = policy.sample_action(state_tensor)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        frames.append(env.render())  # final frame

        video_path = os.path.join(video_dir, f"policy-episode-{ep}.mp4")
        imageio.mimwrite(video_path, frames, fps=fps)
        print(f"Saved video ({len(frames)} frames, {len(frames) / fps:.1f}s): {video_path}")

    env.close()
