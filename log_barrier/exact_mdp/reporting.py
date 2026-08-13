"""CSV and plot output for the exact finite-MDP comparison."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt

from .training import ExactTrainingResult


COLORS = {
    "reward_only": "#1f77b4",
    "policy_fisher_logdet": "#2ca02c",
    "joint_fisher_logdet": "#d62728",
}


def write_csv(path: Path, rows) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(materialized[0]))
        writer.writeheader()
        writer.writerows(materialized)


def plot_behavior(results: list[ExactTrainingResult], output: Path) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for result in results:
        rows = result.trajectory
        method = result.config.method
        color = COLORS[method]
        updates = [row["update"] for row in rows]
        axes[0, 0].plot(updates, [row["return"] for row in rows], label=method, color=color)
        axes[0, 1].plot(updates, [row["q0"] for row in rows], color=color, label=f"{method}: q0")
        axes[0, 1].plot(updates, [row["q1"] for row in rows], color=color, linestyle="--", label=f"{method}: q1")
        axes[1, 0].plot(updates, [row["d1"] for row in rows], color=color, label=f"{method}: d1")
        axes[1, 0].plot(updates, [row["d2"] for row in rows], color=color, linestyle="--", label=f"{method}: d2")
        axes[1, 1].plot(updates, [row["V1"] for row in rows], color=color, label=f"{method}: V1")
        axes[1, 1].plot(updates, [row["V2"] for row in rows], color=color, linestyle="--", label=f"{method}: V2")
    axes[1, 1].axhline(0.5, color="black", linestyle=":", linewidth=1)
    axes[1, 1].axhline(0.55, color="gray", linestyle=":", linewidth=1)
    titles = ("Exact return", "Continuation probabilities", "Discounted visitation", "Downstream values")
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title)
        axis.set_xlabel("update")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_spectra(results: list[ExactTrainingResult], output: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for result in results:
        color = COLORS[result.config.method]
        for column, fisher_name in enumerate(("policy", "joint")):
            rows = [row for row in result.spectra if row["fisher"] == fisher_name]
            updates = [row["update"] for row in rows]
            for index in range(1, 7):
                axes[0, column].plot(
                    updates,
                    [row[f"eigenvalue_{index}"] for row in rows],
                    color=color,
                    alpha=0.35 + 0.1 * (7 - index),
                    linewidth=1,
                )
            axes[1, column].plot(updates, [row["condition_number"] for row in rows], color=color, label=result.config.method)
        policy_rows = [row for row in result.spectra if row["fisher"] == "policy"]
        joint_rows = [row for row in result.spectra if row["fisher"] == "joint"]
        updates = [row["update"] for row in policy_rows]
        axes[0, 2].plot(updates, [row["half_logdet"] for row in policy_rows], color=color, linestyle="--", label=f"{result.config.method}: BP")
        axes[0, 2].plot(updates, [row["half_logdet"] for row in joint_rows], color=color, label=f"{result.config.method}: BJ")
        axes[1, 2].plot(updates, [row["trace"] for row in policy_rows], color=color, linestyle="--", label=f"{result.config.method}: tr FP")
        axes[1, 2].plot(updates, [row["trace"] for row in joint_rows], color=color, label=f"{result.config.method}: tr FJ")
    axes[0, 0].set_title("Policy-Fisher eigenvalues")
    axes[0, 1].set_title("Joint-Fisher eigenvalues")
    axes[0, 2].set_title("Half log-determinants")
    axes[1, 0].set_title("Policy-Fisher condition number")
    axes[1, 1].set_title("Joint-Fisher condition number")
    axes[1, 2].set_title("Fisher traces")
    for axis in axes.flat:
        axis.set_xlabel("update")
        axis.grid(alpha=0.2)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=6)
    axes[0, 0].set_yscale("log")
    axes[0, 1].set_yscale("log")
    axes[1, 0].set_yscale("log")
    axes[1, 1].set_yscale("log")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_explained_trace(results: list[ExactTrainingResult], output: Path) -> None:
    """Plot cumulative trace and k90 for exact policy and joint Fishers."""

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for result in results:
        method = result.config.method
        color = COLORS[method]
        for column, fisher_name in enumerate(("policy", "joint")):
            rows = [row for row in result.spectra if row["fisher"] == fisher_name]
            updates = [row["update"] for row in rows]
            axes[1, column].plot(updates, [row["k90"] for row in rows], marker="o", color=color, label=method)
            final = rows[-1]
            values = [final[f"eigenvalue_{index}"] for index in range(1, 7)]
            total = sum(values)
            cumulative = []
            running = 0.0
            for value in values:
                running += value
                cumulative.append(running / total)
            axes[0, column].plot(range(1, 7), cumulative, marker="o", color=color, label=method)
    for column, fisher_name in enumerate(("Policy Fisher", "Joint Fisher")):
        for threshold in (0.90, 0.95, 0.99):
            axes[0, column].axhline(threshold, color="#6B7280", linewidth=0.8, linestyle="--")
        axes[0, column].set(title=f"{fisher_name}: final cumulative trace", xlabel="Principal-component count", ylabel="Explained trace")
        axes[0, column].set_ylim(0.0, 1.01)
        axes[1, column].set(title=f"{fisher_name}: k90 through training", xlabel="Update", ylabel="Components for 90% trace")
        for row in range(2):
            axes[row, column].grid(alpha=0.25)
            axes[row, column].legend(fontsize=8)
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)
    plt.close(figure)
