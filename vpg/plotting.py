"""Live training-time plots. See analysis.py for post-hoc plotting of saved reward files."""

import os
import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from vpg.stats import mean_confidence_interval


_ALGO_STYLES = {
    "gpomdp": {"color": "steelblue",  "label": "GPOMDP (Adam)"},
    "npg":    {"color": "darkorange", "label": "NPG"},
}


def plot_algorithm_comparison(
    rewards_dict,
    save_dir,
    env_id="",
    title=None,
    filename="comparison.png",
):
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
            # Capture the line's actual color (matplotlib may auto-assign it when
            # color is None) so the CI band matches its mean line exactly.
            line, = plt.plot(x, mean, label=label, color=color)
            plt.fill_between(x, lower, upper, alpha=0.25, color=line.get_color())

    if title is None:
        title = f"GPOMDP vs NPG — {env_id}" if env_id else "GPOMDP vs NPG"
    plt.xlabel("Iteration")
    plt.ylabel("Average training return")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved comparison plot: {save_path}")


def plot_seed_ci(curves, save_path=None, title="Training reward across seeds",
                 ylabel="Average training return", xlabel="Iteration", label=None):
    """Plot mean +/- 95% CI for a single run's per-seed curves.

    curves: array of shape [n_seeds, n_iterations] (as saved in training_rewards.npy).
    A single-seed run ([1, n_iters]) is plotted as a bare mean with no band.
    """
    curves = np.asarray(curves, dtype=np.float64)
    if curves.ndim == 1:
        curves = curves[None, :]

    if curves.shape[0] == 1:
        mean = curves[0]
        lower = upper = None
    else:
        mean, lower, upper = mean_confidence_interval(curves)

    x_values = np.arange(len(mean))

    plt.figure()
    plt.plot(x_values, mean, label=label or "Mean")
    if lower is not None:
        plt.fill_between(x_values, lower, upper, alpha=0.25, label="95% CI")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=300)
        print(f"Saved CI plot: {save_path}")


def plot_training_curves(
    training_rewards,
    save_dir="plots",
    title="Training reward",
    filename="training_rewards.png",
):
    os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.plot(training_rewards)
    plt.xlabel("Iteration")
    plt.ylabel("Average training return")
    plt.title(title)
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, filename), dpi=300)
    plt.close()
