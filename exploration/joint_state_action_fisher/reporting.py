"""CSV summaries and deterministic plots for the exact Step 2 experiment."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .exact_two_state import ALL_METHODS, ExactRunResult


COLORS = {
    "reward_only": "#1f77b4",
    "statewise_conditional_barrier": "#ff7f0e",
    "pooled_policy_logdet": "#2ca02c",
    "joint_state_action_logdet": "#d62728",
    "state_distribution_only": "#9467bd",
    "joint_correction_only": "#8c564b",
}


def write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        raise ValueError(f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def endpoint_rows(results: Iterable[ExactRunResult]) -> tuple[dict[str, object], ...]:
    rows = []
    for result in results:
        endpoint = dict(result.endpoint)
        endpoint["run_finite"] = result.finite
        endpoint["completed_updates"] = int(endpoint["update"])
        rows.append(endpoint)
    return tuple(rows)


def compact_summary(results: Iterable[ExactRunResult]) -> tuple[dict[str, object], ...]:
    keep = (
        "return",
        "q",
        "p_good",
        "value_s1",
        "mu1",
        "min_pi0",
        "min_pi1",
        "pooled_policy_fisher_smallest_positive_eigenvalue",
        "joint_state_action_fisher_smallest_positive_eigenvalue",
        "pooled_policy_fisher_logdet",
        "joint_state_action_fisher_logdet",
        "cosine_reward_method",
    )
    rows: list[dict[str, object]] = []
    for result in results:
        end = result.endpoint
        row = {
            "protocol": result.config.protocol,
            "initialization": result.config.initialization,
            "method": result.config.method,
            "alpha": result.config.alpha,
            "beta": result.config.beta,
            "updates": result.config.updates,
            "finite": result.finite,
        }
        row.update({f"final_{key}": end[key] for key in keep})
        rows.append(row)

    lookup = {
        (result.config.protocol, result.config.initialization, result.config.beta, result.config.method): result
        for result in results
    }
    for key, joint in lookup.items():
        protocol, initialization, beta, method = key
        if method != "joint_state_action_logdet":
            continue
        pooled = lookup.get((protocol, initialization, beta, "pooled_policy_logdet"))
        if pooled is None:
            continue
        rows.append(
            {
                "protocol": protocol,
                "initialization": initialization,
                "method": "joint_minus_pooled_contrast",
                "alpha": joint.config.alpha,
                "beta": beta,
                "updates": joint.config.updates,
                "finite": joint.finite and pooled.finite,
                **{
                    f"final_{metric}": float(joint.endpoint[metric]) - float(pooled.endpoint[metric])
                    for metric in keep
                    if isinstance(joint.endpoint[metric], (int, float))
                    and isinstance(pooled.endpoint[metric], (int, float))
                },
            }
        )
    return tuple(rows)


def gradient_rows(results: Iterable[ExactRunResult]) -> tuple[dict[str, object], ...]:
    prefixes = (
        "reward_gradient_",
        "pooled_policy_gradient_",
        "joint_gradient_",
        "joint_correction_gradient_",
        "method_regularizer_gradient_",
        "total_gradient_",
        "cosine_",
    )
    base = {
        "run_id",
        "protocol",
        "initialization",
        "method",
        "alpha",
        "beta",
        "update",
        "q",
        "p_good",
        "reward_gradient_norm",
        "pooled_policy_gradient_norm",
        "joint_gradient_norm",
        "joint_correction_gradient_norm",
        "joint_correction_to_pooled_norm_ratio",
        "method_regularizer_gradient_norm",
        "applied_regularizer_gradient_norm",
        "total_gradient_norm",
    }
    rows = []
    for result in results:
        for checkpoint in result.checkpoints:
            rows.append(
                {
                    key: value
                    for key, value in checkpoint.items()
                    if key in base or any(key.startswith(prefix) for prefix in prefixes)
                }
            )
    return tuple(rows)


def _series(result: ExactRunResult, key: str) -> np.ndarray:
    return np.asarray([float(row[key]) for row in result.checkpoints], dtype=float)


def _updates(result: ExactRunResult) -> np.ndarray:
    return np.asarray([int(row["update"]) for row in result.checkpoints], dtype=int)


def _plot_lines(axis, results: list[ExactRunResult], key: str, *, title: str, ylabel: str) -> None:
    for result in results:
        axis.plot(
            _updates(result),
            _series(result, key),
            color=COLORS[result.config.method],
            label=result.config.method,
            linewidth=1.6,
        )
    axis.set_title(title)
    axis.set_xlabel("update")
    axis.set_ylabel(ylabel)
    axis.grid(alpha=0.2)


def plot_scenario(results: list[ExactRunResult], output_dir: Path, title: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered = sorted(results, key=lambda result: ALL_METHODS.index(result.config.method))

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _plot_lines(axes[0, 0], ordered, "return", title="Exact return", ylabel="J")
    _plot_lines(axes[0, 1], ordered, "q", title="Downstream access", ylabel="q")
    _plot_lines(axes[1, 0], ordered, "p_good", title="Good-action probability at s1", ylabel="pi1(a0)")
    for result in ordered:
        axes[1, 1].plot(
            _series(result, "q"),
            _series(result, "p_good"),
            color=COLORS[result.config.method],
            label=result.config.method,
            linewidth=1.6,
        )
        axes[1, 1].scatter(
            [_series(result, "q")[-1]], [_series(result, "p_good")[-1]],
            color=COLORS[result.config.method], s=18,
        )
    axes[1, 1].set(title="Phase trajectory", xlabel="q", ylabel="pi1(a0)")
    axes[1, 1].grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(output_dir / "behavior_and_phase.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _plot_lines(axes[0, 0], ordered, "mu1", title="Transition-pooled downstream weight", ylabel="mu1")
    for result in ordered:
        updates = _updates(result)
        color = COLORS[result.config.method]
        axes[0, 1].plot(
            updates, _series(result, "pooled_policy_fisher_smallest_positive_eigenvalue"),
            color=color, linewidth=1.5, label=f"{result.config.method}: policy",
        )
        axes[0, 1].plot(
            updates, _series(result, "joint_state_action_fisher_smallest_positive_eigenvalue"),
            color=color, linewidth=1.2, linestyle="--", label=f"{result.config.method}: joint",
        )
        axes[1, 0].plot(
            updates, _series(result, "pooled_policy_fisher_logdet"), color=color, linewidth=1.5,
        )
        axes[1, 0].plot(
            updates, _series(result, "joint_state_action_fisher_logdet"), color=color,
            linewidth=1.2, linestyle="--",
        )
    axes[0, 1].set(title="Smallest positive eigenvalue (solid policy, dashed joint)", xlabel="update", ylabel="lambda+")
    axes[1, 0].set(title="Log determinant (solid policy, dashed joint)", xlabel="update", ylabel="logdet")
    axes[0, 1].grid(alpha=0.2)
    axes[1, 0].grid(alpha=0.2)
    _plot_lines(
        axes[1, 1], ordered, "joint_correction_to_pooled_norm_ratio",
        title="Joint correction / pooled-gradient norm", ylabel="norm ratio",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(output_dir / "weighting_and_geometry.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    _plot_lines(axes[0, 0], ordered, "reward_gradient_norm", title="Reward-gradient norm", ylabel="norm")
    _plot_lines(
        axes[0, 1], ordered, "applied_regularizer_gradient_norm",
        title="Applied regularizer-gradient norm", ylabel="beta * norm",
    )
    _plot_lines(
        axes[1, 0], ordered, "cosine_reward_method",
        title="Reward versus method regularizer", ylabel="cosine",
    )
    _plot_lines(
        axes[1, 1], ordered, "cosine_pooled_joint",
        title="Pooled versus joint regularizer", ylabel="cosine",
    )
    for axis in axes.flat:
        if "cosine" in axis.get_ylabel():
            axis.set_ylim(-1.05, 1.05)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(output_dir / "gradient_diagnostics.png", dpi=170)
    plt.close(fig)


def plot_vector_fields(rows: Iterable[dict[str, object]], path: Path) -> None:
    materialized = list(rows)
    methods = ("reward_only", "pooled_policy_logdet", "joint_state_action_logdet")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True, sharey=True)
    for axis, method in zip(axes, methods):
        selected = [row for row in materialized if row["method"] == method]
        q = np.asarray([row["q"] for row in selected], dtype=float)
        p = np.asarray([row["p_good"] for row in selected], dtype=float)
        dq = np.asarray([row["dq_dt"] for row in selected], dtype=float)
        dp = np.asarray([row["dp_good_dt"] for row in selected], dtype=float)
        speed = np.sqrt(dq * dq + dp * dp)
        scale = np.where(speed > 0, speed, 1.0)
        axis.quiver(q, p, dq / scale, dp / scale, speed, cmap="viridis", angles="xy", scale=24)
        axis.axvline(2.0 / 3.0, color="black", linestyle=":", linewidth=1, alpha=0.7)
        axis.set(title=method, xlabel="q", ylabel="pi1(a0)")
        axis.grid(alpha=0.15)
    fig.suptitle("Exact objective vector fields (regularized panels use beta=0.1)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def create_all_plots(results: tuple[ExactRunResult, ...], vector_rows, output_root: Path) -> None:
    same_beta = [result for result in results if result.config.protocol == "same_beta"]
    initializations = sorted({result.config.initialization for result in same_beta})
    regularized_betas = sorted({result.config.beta for result in same_beta if result.config.method != "reward_only"})
    for initialization in initializations:
        reward = [
            result for result in same_beta
            if result.config.initialization == initialization and result.config.method == "reward_only"
        ]
        for beta in regularized_betas:
            selected = reward + [
                result for result in same_beta
                if result.config.initialization == initialization
                and result.config.method != "reward_only"
                and result.config.beta == beta
            ]
            label = format(beta, ".12g").replace(".", "p")
            plot_scenario(
                selected,
                output_root / "same_beta" / f"{initialization}_beta_{label}",
                f"Same-beta exact comparison: {initialization}, beta={beta:g}",
            )
    magnitude = [result for result in results if result.config.protocol == "magnitude_matched"]
    plot_scenario(
        magnitude,
        output_root / "magnitude_matched" / "adverse_kappa_1",
        "Magnitude-matched exact comparison: adverse, kappa=1",
    )
    plot_vector_fields(vector_rows, output_root / "vector_fields" / "objective_vector_fields.png")
