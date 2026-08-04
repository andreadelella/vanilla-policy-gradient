"""CSV summaries and deterministic plots for Step 4."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vpg.stats import mean_confidence_interval

from .experiment import EstimatorAuditResult, SampledTrainingResult


LABELS = {
    "reward_only": "reward only",
    "detached_conditional_oracle": "conditional oracle",
    "complete_weighted_oracle": "complete weighted oracle",
    "uniform_action_oracle": "uniform action oracle",
    "visitation_only_oracle": "visitation oracle",
    "full_pooled_fisher_oracle": "full pooled-Fisher oracle",
    "detached_conditional_sampled": "conditional sampled",
}


def write_rows(path: str | Path, rows: Iterable[dict]) -> None:
    rows = list(rows)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_audit_csv(path: str | Path, result: EstimatorAuditResult) -> None:
    write_rows(path, result.rows)


def _ci(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return mean_confidence_interval(np.asarray(data, dtype=np.float64).T)


def training_summary_rows(results: list[tuple[str, SampledTrainingResult]]) -> list[dict]:
    rows: list[dict] = []
    for experiment, result in results:
        cfg = result.config
        for metric in ("population_return", "q", "pi1_a0", "zero_s1", "empirical_fisher_rank"):
            values = result.metrics[metric][-1].astype(np.float64)
            mean, lower, upper = mean_confidence_interval(values)
            rows.append(
                {
                    "experiment": experiment,
                    "label": cfg.label,
                    "method": cfg.method,
                    "initialization": cfg.initialization,
                    "n": cfg.n_trajectories,
                    "beta": cfg.beta,
                    "beta_after": cfg.beta_after,
                    "handoff_update": cfg.handoff_update,
                    "alpha": cfg.alpha,
                    "n_seeds": cfg.n_seeds,
                    "reward_noise_std": cfg.reward_noise_std,
                    "center_returns": int(cfg.center_returns),
                    "normalize_returns": int(cfg.normalize_returns),
                    "metric": metric,
                    "mean": float(mean),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "finite_fraction": float(result.finite.mean()),
                }
            )
    return rows


def paired_difference_rows(results: list[tuple[str, SampledTrainingResult]]) -> list[dict]:
    rewards: dict[tuple, SampledTrainingResult] = {}
    for experiment, result in results:
        cfg = result.config
        if cfg.method == "reward_only":
            rewards[(experiment, cfg.initialization, cfg.n_trajectories, cfg.reward_noise_std, cfg.center_returns, cfg.normalize_returns)] = result
    rows: list[dict] = []
    for experiment, result in results:
        cfg = result.config
        if cfg.method == "reward_only":
            continue
        key = (experiment, cfg.initialization, cfg.n_trajectories, cfg.reward_noise_std, cfg.center_returns, cfg.normalize_returns)
        baseline = rewards[key]
        for metric in ("population_return", "q", "pi1_a0"):
            difference = result.metrics[metric][-1] - baseline.metrics[metric][-1]
            mean, lower, upper = mean_confidence_interval(difference)
            rows.append(
                {
                    "experiment": experiment,
                    "label": cfg.label,
                    "method": cfg.method,
                    "initialization": cfg.initialization,
                    "n": cfg.n_trajectories,
                    "beta": cfg.beta,
                    "beta_after": cfg.beta_after,
                    "handoff_update": cfg.handoff_update,
                    "metric": metric,
                    "mean_paired_difference": float(mean),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "n_pairs": cfg.n_seeds,
                }
            )
    return rows


def make_audit_plots(output: str | Path, result: EstimatorAuditResult) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = result.rows
    policies = list(dict.fromkeys(row["policy"] for row in rows))
    specifications = (
        ("exact_mu1_bias", "Finite-batch bias of $\\widehat\\mu_1$", "mu1_bias.png", False),
        ("zero_s1_probability", "Probability of no $s_1$ transition", "zero_s1.png", True),
        ("conditional_exact_bias_norm", "Conditional-barrier bias norm", "conditional_bias.png", True),
        ("fisher_exact_bias_fro", "Expected empirical-Fisher bias", "fisher_bias.png", True),
        ("fisher_full_rank_fraction", "Empirical Fisher full-rank fraction", "fisher_rank.png", False),
        ("fisher_logdet_defined_fraction", "Defined empirical log-determinant fraction", "logdet_defined.png", False),
    )
    for key, title, filename, log_y in specifications:
        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        for policy in policies:
            selected = [row for row in rows if row["policy"] == policy]
            x = np.asarray([row["n"] for row in selected])
            y = np.asarray([row[key] for row in selected], dtype=np.float64)
            if log_y:
                y = np.maximum(np.abs(y), np.finfo(np.float64).tiny)
            ax.plot(x, y, marker="o", label=policy.replace("_", " "))
        ax.set_xscale("log", base=2)
        if log_y:
            ax.set_yscale("log")
        if key == "zero_s1_probability":
            ax.set_ylim(top=1.05)
        ax.set_xlabel("trajectories per batch N")
        ax.set_title(title)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=160)
        plt.close(fig)


def make_training_group_plots(output: str | Path, results: list[SampledTrainingResult]) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    panels = (
        ("population_return", "Exact population return"),
        ("q", "$q=\\pi_0(a_1)$"),
        ("pi1_a0", "$\\pi_1(a_0)$"),
        ("zero_s1", "Fraction with no $s_1$ transition"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for result in results:
        for ax, (metric, title) in zip(axes.flat, panels):
            values = result.metrics[metric]
            steps = result.steps
            if metric == "zero_s1":
                values = values[1:]
                steps = steps[1:]
            mean, low, high = _ci(values)
            if metric == "zero_s1":
                low, high = np.clip(low, 0, 1), np.clip(high, 0, 1)
            label = LABELS[result.config.method]
            ax.plot(steps, mean, label=label)
            ax.fill_between(steps, low, high, alpha=0.12)
            ax.set_title(title)
            ax.set_xlabel("update")
            ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "performance_and_visitation.png", dpi=160)
    plt.close(fig)

    diagnostics = (
        ("sampled_reward_gradient_error", "Reward-gradient error"),
        ("sampled_conditional_error", "Sampled conditional error"),
        ("empirical_fisher_fro_error", "Empirical-Fisher Frobenius error"),
        ("empirical_fisher_rank", "Empirical Fisher rank"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for result in results:
        for ax, (metric, title) in zip(axes.flat, diagnostics):
            values = result.metrics[metric]
            mean, low, high = _ci(values)
            low = np.maximum(low, 0)
            if metric == "empirical_fisher_rank":
                high = np.minimum(high, 4)
            ax.plot(result.steps, mean, label=LABELS[result.config.method])
            ax.fill_between(result.steps, low, high, alpha=0.1)
            ax.set_title(title)
            ax.set_xlabel("update")
            ax.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "estimator_diagnostics.png", dpi=160)
    plt.close(fig)


def make_endpoint_heatmaps(output: str | Path, results: list[tuple[str, SampledTrainingResult]]) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    main = [result for experiment, result in results if experiment == "main" and result.config.method != "reward_only"]
    for initialization in sorted({result.config.initialization for result in main}):
        for method in sorted({result.config.method for result in main}):
            selected = [r for r in main if r.config.initialization == initialization and r.config.method == method]
            ns = sorted({r.config.n_trajectories for r in selected})
            betas = sorted({r.config.beta for r in selected})
            matrix = np.full((len(ns), len(betas)), np.nan)
            for result in selected:
                matrix[ns.index(result.config.n_trajectories), betas.index(result.config.beta)] = np.mean(result.metrics["population_return"][-1])
            fig, ax = plt.subplots(figsize=(5.0, 3.6))
            image = ax.imshow(matrix, origin="lower", aspect="auto")
            ax.set_xticks(range(len(betas)), [f"{value:g}" for value in betas])
            ax.set_yticks(range(len(ns)), [str(value) for value in ns])
            ax.set_xlabel("beta")
            ax.set_ylabel("batch trajectories N")
            ax.set_title(f"Final return: {LABELS[method]} ({initialization})")
            fig.colorbar(image, ax=ax)
            fig.tight_layout()
            fig.savefig(output / f"{initialization}__{method}.png", dpi=160)
            plt.close(fig)


def make_handoff_plots(output: str | Path, results: list[SampledTrainingResult]) -> None:
    """Plot the dedicated temporary-barrier experiment."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    labels = {
        "reward_only": "reward only",
        "sampled_conditional_fixed": "sampled conditional, fixed beta=0.2",
        "sampled_conditional_handoff": "sampled conditional, beta=0.2 then 0",
        "full_oracle_handoff": "full oracle, beta=0.1 then 0",
    }
    panels = (
        ("population_return", "Exact population return"),
        ("q", "$q=\\pi_0(a_1)$"),
        ("pi1_a0", "$\\pi_1(a_0)$"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for result in results:
        label = labels[result.config.label]
        for ax, (metric, title) in zip(axes.flat[:3], panels):
            mean, low, high = _ci(result.metrics[metric])
            ax.plot(result.steps, mean, label=label)
            ax.fill_between(result.steps, low, high, alpha=0.12)
            ax.set_title(title)
            ax.set_xlabel("update")
            ax.grid(alpha=0.2)
        if result.config.handoff_update is None:
            beta_steps = (0, result.config.updates)
            beta_values = (result.config.beta, result.config.beta)
        else:
            beta_steps = (
                0,
                result.config.handoff_update,
                result.config.handoff_update,
                result.config.updates,
            )
            beta_values = (
                result.config.beta,
                result.config.beta,
                result.config.beta_after,
                result.config.beta_after,
            )
        axes.flat[3].plot(beta_steps, beta_values, label=label)
    axes.flat[3].set_title("Applied regularizer coefficient")
    axes.flat[3].set_xlabel("update")
    axes.flat[3].set_ylabel("beta")
    axes.flat[3].grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "performance_and_handoff.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for result in results:
        q_mean = np.nanmean(result.metrics["q"], axis=1)
        good_mean = np.nanmean(result.metrics["pi1_a0"], axis=1)
        (line,) = ax.plot(q_mean, good_mean, label=labels[result.config.label])
        handoff = result.config.handoff_update
        if handoff is not None:
            index = int(np.flatnonzero(result.steps == handoff)[0])
            ax.scatter(
                q_mean[index], good_mean[index], s=28, marker="o",
                color=line.get_color(), zorder=3,
            )
    ax.set_xlabel("$q=\\pi_0(a_1)$")
    ax.set_ylabel("$\\pi_1(a_0)$")
    ax.set_title("Mean policy phase trajectory; dots mark handoff")
    ax.grid(alpha=0.2)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output / "phase_handoff.png", dpi=180)
    plt.close(fig)


def handoff_checkpoint_rows(
    results: list[tuple[str, SampledTrainingResult]], checkpoints: tuple[int, ...]
) -> list[dict]:
    rows: list[dict] = []
    for experiment, result in results:
        for checkpoint in checkpoints:
            matches = np.flatnonzero(result.steps == checkpoint)
            if len(matches) != 1:
                raise ValueError(f"checkpoint {checkpoint} is not recorded")
            index = int(matches[0])
            for metric in ("population_return", "q", "pi1_a0"):
                mean, lower, upper = mean_confidence_interval(result.metrics[metric][index])
                rows.append(
                    {
                        "experiment": experiment,
                        "label": result.config.label,
                        "initialization": result.config.initialization,
                        "checkpoint": checkpoint,
                        "metric": metric,
                        "mean": float(mean),
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                        "n_seeds": result.config.n_seeds,
                    }
                )
    return rows


def handoff_change_rows(
    results: list[tuple[str, SampledTrainingResult]], handoff_update: int
) -> list[dict]:
    rows: list[dict] = []
    for experiment, result in results:
        matches = np.flatnonzero(result.steps == handoff_update)
        if len(matches) != 1:
            raise ValueError(f"handoff checkpoint {handoff_update} is not recorded")
        index = int(matches[0])
        for metric in ("population_return", "q", "pi1_a0"):
            difference = result.metrics[metric][-1] - result.metrics[metric][index]
            mean, lower, upper = mean_confidence_interval(difference)
            rows.append(
                {
                    "experiment": experiment,
                    "label": result.config.label,
                    "initialization": result.config.initialization,
                    "from_update": handoff_update,
                    "to_update": result.config.updates,
                    "metric": metric,
                    "mean_change": float(mean),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "n_seeds": result.config.n_seeds,
                }
            )
    return rows


def handoff_pairwise_rows(
    results: list[tuple[str, SampledTrainingResult]],
) -> list[dict]:
    comparisons = (
        ("sampled_conditional_handoff", "sampled_conditional_fixed"),
        ("sampled_conditional_handoff", "reward_only"),
        ("full_oracle_handoff", "sampled_conditional_handoff"),
    )
    rows: list[dict] = []
    initializations = sorted({result.config.initialization for _, result in results})
    for initialization in initializations:
        by_label = {
            result.config.label: result
            for _, result in results
            if result.config.initialization == initialization
        }
        for left_label, right_label in comparisons:
            left, right = by_label[left_label], by_label[right_label]
            for metric in ("population_return", "q", "pi1_a0"):
                difference = left.metrics[metric][-1] - right.metrics[metric][-1]
                mean, lower, upper = mean_confidence_interval(difference)
                rows.append(
                    {
                        "initialization": initialization,
                        "left": left_label,
                        "right": right_label,
                        "metric": metric,
                        "mean_paired_difference": float(mean),
                        "ci_lower": float(lower),
                        "ci_upper": float(upper),
                        "n_pairs": left.config.n_seeds,
                    }
                )
    return rows
