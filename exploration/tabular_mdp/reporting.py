"""CSV and plot generation for the exact two-state experiment."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .experiment import ExactTrainingResult


DISPLAY_NAMES = {
    "reward_only": "reward only",
    "detached_conditional": "detached conditional",
    "complete_weighted": "complete weighted",
    "uniform_action": "uniform action",
    "visitation_only": "visitation only",
    "full_pooled_fisher": "full pooled Fisher",
}


def write_summary(path: Path, groups: Iterable[tuple[str, ExactTrainingResult]]) -> None:
    fields = [
        "experiment", "label", "method", "alpha", "beta", "updates", "run",
        "finite", "final_return", "final_q", "final_p_good", "final_min_pi0",
        "final_min_pi1", "final_lambda_min_f_pool", "final_logdet_f_pool",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for experiment, result in groups:
            for run in range(result.phi.shape[1]):
                writer.writerow(
                    {
                        "experiment": experiment,
                        "label": result.config.label,
                        "method": result.config.method,
                        "alpha": result.config.alpha,
                        "beta": result.config.beta,
                        "updates": result.config.updates,
                        "run": run,
                        "finite": int(result.finite[run]),
                        "final_return": result.metrics["return"][-1, run],
                        "final_q": result.metrics["q"][-1, run],
                        "final_p_good": result.metrics["pi1_a0"][-1, run],
                        "final_min_pi0": result.metrics["min_pi0"][-1, run],
                        "final_min_pi1": result.metrics["min_pi1"][-1, run],
                        "final_lambda_min_f_pool": result.metrics["lambda_min_f_pool"][-1, run],
                        "final_logdet_f_pool": result.metrics["logdet_f_pool"][-1, run],
                    }
                )


def _plot_lines(ax, results: list[ExactTrainingResult], metric: str, title: str, *, log=False) -> None:
    for result in results:
        ax.plot(result.steps, result.metrics[metric][:, 0], label=DISPLAY_NAMES[result.config.method])
    ax.set_title(title)
    ax.set_xlabel("update")
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)


def make_main_plots(target: Path, results: list[ExactTrainingResult]) -> None:
    target.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _plot_lines(axes[0, 0], results, "return", "Exact return")
    _plot_lines(axes[0, 1], results, "q", r"Visit probability $q=\pi_0(a_1)$")
    _plot_lines(axes[1, 0], results, "pi1_a0", r"Good action probability $\pi_1(a_0)$")
    _plot_lines(axes[1, 1], results, "min_pi0", r"Minimum action probability at $s_0$", log=True)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(target / "performance_policy.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _plot_lines(axes[0, 0], results, "min_pi0", r"$\min_a\pi_0(a)$", log=True)
    _plot_lines(axes[0, 1], results, "min_pi1", r"$\min_a\pi_1(a)$", log=True)
    _plot_lines(axes[1, 0], results, "lambda_min_f0", r"$\lambda_{\min}(F_0)$", log=True)
    _plot_lines(axes[1, 1], results, "lambda_min_f1", r"$\lambda_{\min}(F_1)$", log=True)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(target / "statewise_geometry.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    _plot_lines(axes[0, 0], results, "lambda_min_f_pool", r"$\lambda_{\min}(F_{pool})$", log=True)
    _plot_lines(axes[0, 1], results, "logdet_f0", r"$\log\det F_0$")
    _plot_lines(axes[1, 0], results, "logdet_f1", r"$\log\det F_1$")
    _plot_lines(axes[1, 1], results, "logdet_f_pool", r"$\log\det F_{pool}$")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(target / "pooled_statewise_logdet.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _plot_lines(axes[0], results, "b_uniform", r"Action term $B_{uniform}$")
    _plot_lines(axes[1], results, "b_visit", r"Visitation term $B_{visit}$")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(target / "action_visitation_barriers.png", dpi=170); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    _plot_lines(axes[0], results, "grad_reward_norm", r"$\|\nabla J\|_2$", log=True)
    _plot_lines(axes[1], results, "grad_regularizer_scaled_norm", r"$\|\beta\nabla B\|_2$", log=True)
    _plot_lines(axes[2], results, "cosine_reward_regularizer", "Reward/regularizer cosine")
    axes[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(target / "gradient_diagnostics.png", dpi=170); plt.close(fig)

    full = next((x for x in results if x.config.method == "full_pooled_fisher"), None)
    weighted = next((x for x in results if x.config.method == "complete_weighted"), None)
    if full is not None and weighted is not None:
        fig, axes = plt.subplots(2, 2, figsize=(11, 8))
        axes[0, 0].plot(full.steps, full.metrics["grad_uniform_scaled_norm"][:, 0], label="uniform action")
        axes[0, 0].plot(full.steps, full.metrics["grad_visit_scaled_norm"][:, 0], label="visitation")
        axes[0, 0].set_yscale("log"); axes[0, 0].set_title("Full-barrier component norms")
        axes[0, 1].plot(full.steps, full.metrics["cosine_reward_uniform"][:, 0], label="uniform action")
        axes[0, 1].plot(full.steps, full.metrics["cosine_reward_visit"][:, 0], label="visitation")
        axes[0, 1].set_title("Full-barrier component cosines")
        axes[1, 0].plot(weighted.steps, weighted.metrics["grad_conditional_scaled_norm"][:, 0], label="detached conditional")
        axes[1, 0].plot(weighted.steps, weighted.metrics["grad_weight_state_scaled_norm"][:, 0], label="state-weight derivative")
        axes[1, 0].set_yscale("log"); axes[1, 0].set_title("Complete-weighted component norms")
        axes[1, 1].plot(weighted.steps, weighted.metrics["cosine_reward_conditional"][:, 0], label="detached conditional")
        axes[1, 1].plot(weighted.steps, weighted.metrics["cosine_reward_weight_state"][:, 0], label="state-weight derivative")
        axes[1, 1].set_title("Complete-weighted component cosines")
        for ax in axes.flat:
            ax.set_xlabel("update"); ax.grid(alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(target / "gradient_decompositions.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    for result in results:
        ax.plot(result.metrics["q"][:, 0], result.metrics["pi1_a0"][:, 0], label=DISPLAY_NAMES[result.config.method])
    ax.set_xlabel(r"$q=\pi_0(a_1)$"); ax.set_ylabel(r"$\pi_1(a_0)$")
    ax.set_title("Policy-space phase trajectory"); ax.grid(alpha=0.25); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(target / "phase_q_vs_good.png", dpi=170); plt.close(fig)


def make_basin_plot(path: Path, result: ExactTrainingResult, grid: np.ndarray) -> None:
    size = len(grid)
    fields = (
        ("return", "Final return"), ("q", "Final q"),
        ("pi1_a0", "Final good-action probability"), ("near", "Near-optimal region"),
    )
    final = {key: result.metrics[key][-1].reshape(size, size) for key in ("return", "q", "pi1_a0")}
    final["near"] = ((final["q"] >= 0.9) & (final["pi1_a0"] >= 0.9)).astype(float)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, fields):
        image = ax.imshow(final[key], origin="lower", extent=(grid[0], grid[-1], grid[0], grid[-1]), aspect="auto")
        ax.set_title(title); ax.set_xlabel(r"initial $p_{good}$"); ax.set_ylabel(r"initial $q$")
        fig.colorbar(image, ax=ax, fraction=0.046)
    fig.suptitle(DISPLAY_NAMES[result.config.method])
    fig.savefig(path, dpi=170); plt.close(fig)
