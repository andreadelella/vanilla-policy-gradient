"""Statistical summaries and plots for saved categorical-bandit runs."""

import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from exploration.categorical_bandit.environment import BanditBatch
from exploration.categorical_bandit.experiment import TrainingResult
from exploration.categorical_bandit.presets import configuration_key
from vpg.stats import mean_confidence_interval


SUMMARY_METRICS = (
    "optimal_arm_probability",
    "expected_reward",
    "expected_pseudo_regret",
    "cumulative_pseudo_regret",
    "minimum_probability",
    "normalized_log_fisher_volume",
    "entropy",
    "numerically_collapsed",
)


def finite_mean_ci(values: np.ndarray) -> tuple[float, float, float, int]:
    """Use the project Student-t helper after removing recorded failures."""
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    count = int(finite.size)
    if count == 0:
        return math.nan, math.nan, math.nan, 0
    if count == 1:
        value = float(finite[0])
        return value, math.nan, math.nan, 1
    mean, lower, upper = mean_confidence_interval(finite[:, None])
    return float(mean[0]), float(lower[0]), float(upper[0]), count


def curve_mean_ci(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    means, lowers, uppers, counts = [], [], [], []
    for column in range(values.shape[1]):
        mean, lower, upper, count = finite_mean_ci(values[:, column])
        means.append(mean)
        lowers.append(lower)
        uppers.append(upper)
        counts.append(count)
    return tuple(np.asarray(x) for x in (means, lowers, uppers, counts))


def write_run_status(
    path: Path,
    results: Iterable[TrainingResult],
    bandits: dict[int, BanditBatch],
) -> None:
    fields = [
        "configuration",
        "algorithm",
        "run",
        "failed",
        "failure_step",
        "final_modal_action_correct",
        "final_optimal_probability",
        *[f"final_{name}" for name in SUMMARY_METRICS if name != "optimal_arm_probability"],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            optimal = bandits[result.config.num_actions].optimal_actions.numpy()
            for run in range(result.config.num_runs):
                probabilities = result.final_probabilities[run]
                correct = "" if result.failed[run] else int(np.argmax(probabilities) == optimal[run])
                row = {
                    "configuration": configuration_key(result.config),
                    "algorithm": result.config.algorithm.key,
                    "run": run,
                    "failed": int(result.failed[run]),
                    "failure_step": int(result.failure_steps[run]),
                    "final_modal_action_correct": correct,
                    "final_optimal_probability": result.metrics["optimal_arm_probability"][run, -1],
                }
                for name in SUMMARY_METRICS:
                    if name != "optimal_arm_probability":
                        row[f"final_{name}"] = result.metrics[name][run, -1]
                writer.writerow(row)


def write_summary(
    path: Path,
    results: Iterable[TrainingResult],
    bandits: dict[int, BanditBatch],
) -> None:
    fields = [
        "configuration",
        "algorithm",
        "num_runs",
        "finite_runs",
        "failed_fraction",
        "incorrect_modal_action_rate",
        "optimal_probability_below_0p5_fraction",
        "metric",
        "mean",
        "ci95_lower",
        "ci95_upper",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            finite = ~result.failed
            optimal = bandits[result.config.num_actions].optimal_actions.numpy()
            modal = np.argmax(np.nan_to_num(result.final_probabilities, nan=-np.inf), axis=1)
            incorrect = float(np.mean(modal[finite] != optimal[finite])) if finite.any() else math.nan
            below = result.metrics["optimal_arm_probability"][:, -1] < 0.5
            below_rate = float(np.mean(below[finite])) if finite.any() else math.nan
            shared = {
                "configuration": configuration_key(result.config),
                "algorithm": result.config.algorithm.key,
                "num_runs": result.config.num_runs,
                "finite_runs": int(finite.sum()),
                "failed_fraction": float(result.failed.mean()),
                "incorrect_modal_action_rate": incorrect,
                "optimal_probability_below_0p5_fraction": below_rate,
            }
            for metric in SUMMARY_METRICS:
                mean, lower, upper, _ = finite_mean_ci(result.metrics[metric][:, -1])
                writer.writerow(
                    {**shared, "metric": metric, "mean": mean, "ci95_lower": lower, "ci95_upper": upper}
                )


def write_paired_differences(path: Path, results: Iterable[TrainingResult]) -> None:
    grouped: dict[str, list[TrainingResult]] = {}
    for result in results:
        grouped.setdefault(configuration_key(result.config), []).append(result)
    fields = [
        "configuration",
        "lb_algorithm",
        "baseline",
        "metric",
        "paired_runs",
        "mean_lb_minus_baseline",
        "ci95_lower",
        "ci95_upper",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for key, group in grouped.items():
            barriers = [x for x in group if x.config.algorithm.kind == "lb_sgb"]
            baselines = [x for x in group if x.config.algorithm.kind != "lb_sgb"]
            for barrier in barriers:
                for baseline in baselines:
                    for metric in SUMMARY_METRICS:
                        difference = (
                            barrier.metrics[metric][:, -1] - baseline.metrics[metric][:, -1]
                        )
                        mean, lower, upper, count = finite_mean_ci(difference)
                        writer.writerow(
                            {
                                "configuration": key,
                                "lb_algorithm": barrier.config.algorithm.key,
                                "baseline": baseline.config.algorithm.key,
                                "metric": metric,
                                "paired_runs": count,
                                "mean_lb_minus_baseline": mean,
                                "ci95_lower": lower,
                                "ci95_upper": upper,
                            }
                        )


def _plot_metric(ax: plt.Axes, group: list[TrainingResult], metric: str, title: str) -> None:
    for result in group:
        mean, lower, upper, _ = curve_mean_ci(result.metrics[metric])
        ax.plot(result.steps, mean, label=result.config.algorithm.key)
        ax.fill_between(result.steps, lower, upper, alpha=0.16)
    ax.set_title(title)
    ax.set_xlabel("update")
    ax.grid(alpha=0.25)


def make_configuration_plots(output_dir: Path, results: Iterable[TrainingResult]) -> None:
    grouped: dict[str, list[TrainingResult]] = {}
    for result in results:
        grouped.setdefault(configuration_key(result.config), []).append(result)
    for key, group in grouped.items():
        target = output_dir / key
        target.mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        _plot_metric(axes[0], group, "optimal_arm_probability", "Optimal-arm probability")
        _plot_metric(axes[1], group, "cumulative_pseudo_regret", "Cumulative pseudo-regret")
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(target / "performance.png", dpi=160)
        plt.close(fig)

        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        _plot_metric(axes[0, 0], group, "minimum_probability", "Minimum probability")
        axes[0, 0].set_yscale("log")
        _plot_metric(
            axes[0, 1], group, "normalized_log_fisher_volume", "Normalized log Fisher volume"
        )
        _plot_metric(axes[1, 0], group, "entropy", "Entropy")
        _plot_metric(axes[1, 1], group, "numerically_collapsed", "Collapsed-run fraction")
        axes[1, 1].set_ylim(-0.02, 1.02)
        axes[0, 0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(target / "geometry.png", dpi=160)
        plt.close(fig)

        # NPG can be hundreds or thousands of log-volume units below the
        # remaining methods.  Preserve the complete plot above, and also make
        # a readable comparison of the non-NPG geometry on its natural scale.
        if any(result.config.algorithm.kind == "npg" for result in group):
            non_npg = [
                result for result in group if result.config.algorithm.kind != "npg"
            ]
            fig, axes = plt.subplots(1, 3, figsize=(15, 4))
            _plot_metric(axes[0], non_npg, "minimum_probability", "Minimum probability")
            axes[0].set_yscale("log")
            _plot_metric(
                axes[1],
                non_npg,
                "normalized_log_fisher_volume",
                "Normalized log Fisher volume",
            )
            _plot_metric(axes[2], non_npg, "entropy", "Entropy")
            axes[0].legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(target / "geometry_without_npg.png", dpi=160)
            plt.close(fig)

        if group[0].config.num_actions == 10:
            available = [x for x in group if x.positive_eigenspectra.size]
            if available:
                matrices = [
                    np.log10(
                        np.maximum(
                            np.nanmean(result.positive_eigenspectra, axis=0).T,
                            np.finfo(np.float64).tiny,
                        )
                    )
                    for result in available
                ]
                finite_values = np.concatenate(
                    [matrix[np.isfinite(matrix)] for matrix in matrices]
                )
                color_min = float(finite_values.min())
                color_max = float(finite_values.max())
                fig, axes = plt.subplots(
                    1,
                    len(available),
                    figsize=(4 * len(available), 3.8),
                    squeeze=False,
                    constrained_layout=True,
                )
                for ax, result, matrix in zip(axes[0], available, matrices):
                    image = ax.imshow(
                        matrix,
                        origin="lower",
                        aspect="auto",
                        extent=(0, result.config.horizon, 1, result.config.num_actions - 1),
                        vmin=color_min,
                        vmax=color_max,
                    )
                    ax.set_title(result.config.algorithm.key)
                    ax.set_xlabel("update")
                    ax.set_ylabel("positive eigenvalue index")
                fig.colorbar(
                    image,
                    ax=list(axes[0]),
                    label="log10 eigenvalue (shared scale)",
                    fraction=0.025,
                    pad=0.02,
                )
                fig.savefig(target / "positive_fisher_eigenspectra.png", dpi=160)
                plt.close(fig)
