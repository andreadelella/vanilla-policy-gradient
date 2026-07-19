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

# Two-sided 95% Student-t critical values for 1..30 degrees of freedom.
# Keeping the small-sample values local avoids adding SciPy solely for a
# single quantile lookup. For larger samples, the Cornish-Fisher expansion
# below rapidly approaches the standard-normal critical value.
_T_CRITICAL_95 = (
    None,
    12.7062047364,
    4.3026527297,
    3.1824463053,
    2.7764451052,
    2.5705818356,
    2.4469118511,
    2.3646242510,
    2.3060041352,
    2.2621571629,
    2.2281388520,
    2.2009851601,
    2.1788128297,
    2.1603686565,
    2.1447866879,
    2.1314495456,
    2.1199052992,
    2.1098155778,
    2.1009220402,
    2.0930240544,
    2.0859634473,
    2.0796138447,
    2.0738730679,
    2.0686576104,
    2.0638985616,
    2.0595385528,
    2.0555294386,
    2.0518305165,
    2.0484071418,
    2.0452296421,
    2.0422724563,
)


def student_t_critical_95(n_samples):
    """Return the two-sided 95% Student-t critical value for a sample mean."""
    if n_samples < 2:
        raise ValueError("At least two independent samples are required for a confidence interval")

    degrees_of_freedom = n_samples - 1
    if degrees_of_freedom <= 30:
        return _T_CRITICAL_95[degrees_of_freedom]

    # Asymptotic expansion of the t quantile around z_(0.975).
    z = 1.959963984540054
    df = float(degrees_of_freedom)
    return (
        z
        + (z**3 + z) / (4.0 * df)
        + (5.0 * z**5 + 16.0 * z**3 + 3.0 * z) / (96.0 * df**2)
        + (3.0 * z**7 + 19.0 * z**5 + 17.0 * z**3 - 15.0 * z)
        / (384.0 * df**3)
    )


def mean_confidence_interval(data):
    """Mean and two-sided 95% Student-t interval across independent samples."""
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 0 or data.shape[0] < 2:
        raise ValueError("At least two independent samples are required for a confidence interval")

    mean = data.mean(axis=0)
    sem = data.std(axis=0, ddof=1) / np.sqrt(data.shape[0])
    margin = student_t_critical_95(data.shape[0]) * sem
    return mean, mean - margin, mean + margin


def plot_comparison(rewards_dict, save_dir, env_id=""):
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
            mean, lower, upper = mean_confidence_interval(rewards)
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

            if isinstance(env.action_space, gym.spaces.Box):
                action = np.clip(action, env.action_space.low, env.action_space.high)

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

        frames.append(env.render())  # final frame

        video_path = os.path.join(video_dir, f"policy-episode-{ep}.mp4")
        imageio.mimwrite(video_path, frames, fps=fps)
        print(f"Saved video ({len(frames)} frames, {len(frames) / fps:.1f}s): {video_path}")

    env.close()
