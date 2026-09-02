"""Plot an illustrative MountainCarContinuous return/geometry Pareto case."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "reward_only": "#222222",
    "barrier": "#16835d",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _rolling_mean(values: list[float], window: int = 10) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return np.asarray(
        [
            array[max(0, index - window + 1) : index + 1].mean()
            for index in range(len(array))
        ]
    )


def _load_training(
    path: Path,
    *,
    method: str,
    seed: int,
) -> list[dict[str, str]]:
    return [
        row
        for row in _read_csv(path)
        if row["method"] == method and int(row["seed"]) == seed
    ]


def _geometry_by_update(path: Path) -> dict[int, dict[str, str]]:
    return {int(row["update"]): row for row in _read_csv(path)}


def _write_comparison(
    reward: dict[int, dict[str, str]],
    barrier: dict[int, dict[str, str]],
    output_path: Path,
) -> None:
    fields = (
        "update",
        "reward_only_return",
        "barrier_return",
        "return_gain",
        "minimum_eigenvalue_gain_orders",
        "conditioning_gain_orders",
        "normalized_logdet_gain",
        "effective_rank_gain",
    )
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for update in sorted(reward):
            baseline = reward[update]
            regularized = barrier[update]
            writer.writerow(
                {
                    "update": update,
                    "reward_only_return": baseline["mean_return"],
                    "barrier_return": regularized["mean_return"],
                    "return_gain": (
                        float(regularized["mean_return"])
                        - float(baseline["mean_return"])
                    ),
                    "minimum_eigenvalue_gain_orders": math.log10(
                        float(regularized["minimum_eigenvalue"])
                        / float(baseline["minimum_eigenvalue"])
                    ),
                    "conditioning_gain_orders": math.log10(
                        float(baseline["positive_condition_number"])
                        / float(regularized["positive_condition_number"])
                    ),
                    "normalized_logdet_gain": (
                        float(regularized["normalized_logdet_positive_spectrum"])
                        - float(baseline["normalized_logdet_positive_spectrum"])
                    ),
                    "effective_rank_gain": (
                        float(regularized["effective_rank"])
                        - float(baseline["effective_rank"])
                    ),
                }
            )


def _plot(
    reward_training: list[dict[str, str]],
    barrier_training: list[dict[str, str]],
    reward_geometry: dict[int, dict[str, str]],
    barrier_geometry: dict[int, dict[str, str]],
    output_path: Path,
) -> None:
    checkpoints = sorted(reward_geometry)
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))

    axis = axes[0, 0]
    for rows, label, color in (
        (reward_training, "GPOMDP, lr=0.015", COLORS["reward_only"]),
        (barrier_training, r"Log-barrier, $\beta=2\times10^{-6}$, lr=0.02", COLORS["barrier"]),
    ):
        updates = [int(row["update"]) for row in rows]
        returns = [float(row["mean_batch_return"]) for row in rows]
        axis.plot(updates, _rolling_mean(returns), color=color, linewidth=2.2, label=label)
    axis.axhline(90.0, color="#777777", linestyle="--", linewidth=1.2)
    axis.set(title="Native return during training", ylabel="10-update mean return")
    axis.legend(frameon=False, fontsize=9)

    axis = axes[0, 1]
    for geometry, label, color in (
        (reward_geometry, "GPOMDP", COLORS["reward_only"]),
        (barrier_geometry, "Log-barrier", COLORS["barrier"]),
    ):
        axis.plot(
            checkpoints,
            [float(geometry[update]["mean_return"]) for update in checkpoints],
            color=color,
            marker="o",
            linewidth=2.2,
            label=label,
        )
    axis.axhline(90.0, color="#777777", linestyle="--", linewidth=1.2)
    axis.set(title="Matched diagnostic return", ylabel="Mean return, 4,096 trajectories")

    minimum_gain = [
        math.log10(
            float(barrier_geometry[update]["minimum_eigenvalue"])
            / float(reward_geometry[update]["minimum_eigenvalue"])
        )
        for update in checkpoints
    ]
    condition_gain = [
        math.log10(
            float(reward_geometry[update]["positive_condition_number"])
            / float(barrier_geometry[update]["positive_condition_number"])
        )
        for update in checkpoints
    ]
    axis = axes[1, 0]
    axis.plot(
        checkpoints,
        minimum_gain,
        color="#2563a6",
        marker="o",
        linewidth=2.2,
        label=r"$\lambda_{\min}$ gain",
    )
    axis.plot(
        checkpoints,
        condition_gain,
        color="#d97706",
        marker="s",
        linewidth=2.2,
        label="Conditioning gain",
    )
    axis.axhline(0.0, color="#222222", linewidth=1.1)
    axis.set(
        title="Fisher non-degeneracy relative to GPOMDP",
        ylabel="Base-10 orders of magnitude",
    )
    axis.legend(frameon=False, fontsize=9)

    logdet_gain = [
        float(barrier_geometry[update]["normalized_logdet_positive_spectrum"])
        - float(reward_geometry[update]["normalized_logdet_positive_spectrum"])
        for update in checkpoints
    ]
    axis = axes[1, 1]
    axis.plot(
        checkpoints,
        logdet_gain,
        color="#8b4ba5",
        marker="o",
        linewidth=2.2,
    )
    axis.axhline(0.0, color="#222222", linewidth=1.1)
    axis.set(
        title="Regularized geometry relative to GPOMDP",
        ylabel="Normalized log-determinant gain",
    )

    for axis in axes.flat:
        axis.set_xlabel("Policy update")
        axis.set_xticks(checkpoints)
        axis.grid(alpha=0.22)
        axis.spines[["top", "right"]].set_visible(False)

    figure.suptitle(
        "MountainCarContinuous tuned illustrative case, seed 102\n"
        "Positive geometry values mean the log-barrier Fisher is better",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=220)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sweep_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    seed = 102
    reward_run = args.sweep_root / "runs" / "reward_only" / f"seed_{seed}"
    barrier_run = args.sweep_root / "runs" / "beta_2e-6" / f"seed_{seed}"
    reward_training = _load_training(
        args.sweep_root / "aggregate" / "training_long.csv",
        method="reward_only",
        seed=seed,
    )
    barrier_training = _load_training(
        args.sweep_root / "aggregate" / "training_long.csv",
        method="beta_2e-6",
        seed=seed,
    )
    reward_geometry = _geometry_by_update(
        reward_run / "trajectory_fisher_analysis_4096" / "summary.csv"
    )
    barrier_geometry = _geometry_by_update(
        barrier_run / "trajectory_fisher_analysis_4096" / "summary.csv"
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_comparison(
        reward_geometry,
        barrier_geometry,
        args.output_dir / "comparison.csv",
    )
    _plot(
        reward_training,
        barrier_training,
        reward_geometry,
        barrier_geometry,
        args.output_dir / "return_and_geometry.png",
    )


if __name__ == "__main__":
    main()
