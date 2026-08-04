"""Run the sampled-conditional barrier switch-time robustness sweep."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from vpg.stats import mean_confidence_interval

from .experiment import SampledTrainingConfig, SampledTrainingResult
from .reporting import write_rows
from .run_experiment import _run_unit, prepare_output
from .verify import TOLERANCES, run_verification


DEFAULT_OUTPUT = Path(
    "exploration/results/tabular_mdp/two_step_trap_sampled/handoff/robustness"
)
SWITCH_TIMES = (500, 1000, 1500, 2000, 2500)
INITIALIZATIONS = ("uniform", "adverse")


def manifest() -> dict:
    return {
        "schema_version": 1,
        "stage": "sampled_conditional_switch_time_robustness",
        "dtype": "torch.float64",
        "device": "cpu",
        "alpha": 0.05,
        "beta": 0.2,
        "beta_after": 0.0,
        "updates": 4000,
        "n_trajectories": 32,
        "n_seeds": 100,
        "record_interval": 10,
        "base_seed": 23,
        "switch_times": list(SWITCH_TIMES),
        "initializations": list(INITIALIZATIONS),
        "returns": "raw reward-to-go; uncentered and unnormalized",
        "rewards": "deterministic",
        "failure_definition": "q < 0.9 or pi1(a0) < 0.9; non-finite seeds also fail",
        "fisher_rank_definition": (
            "numerical rank of the undamped sampled-action empirical Fisher "
            "from the batch used at the recorded optimizer update"
        ),
        "checkpoint_offsets": [0, 250],
        "final_checkpoint": 4000,
        "schedule_indexing": (
            "zero-based t; updates t < switch_time use beta=0.2 and later "
            "updates use beta=0"
        ),
        "stream_derivation": (
            "one CPU torch.Generator seeded 23 per unit; identical tensor shapes "
            "give paired base-uniform streams across switch times"
        ),
        "torch_version": torch.__version__,
    }


def _config(initialization: str, switch_time: int, suite: dict) -> SampledTrainingConfig:
    return SampledTrainingConfig(
        method="detached_conditional_sampled",
        initialization=initialization,
        n_trajectories=suite["n_trajectories"],
        n_seeds=suite["n_seeds"],
        alpha=suite["alpha"],
        beta=suite["beta"],
        beta_after=suite["beta_after"],
        handoff_update=switch_time,
        updates=suite["updates"],
        record_interval=suite["record_interval"],
        base_seed=suite["base_seed"],
        reward_noise_std=0.0,
        center_returns=False,
        normalize_returns=False,
        label=f"switch_{switch_time}",
    )


def _ci(values: np.ndarray) -> tuple[float, float, float]:
    mean, lower, upper = mean_confidence_interval(np.asarray(values, dtype=np.float64))
    return float(mean), float(lower), float(upper)


def final_endpoint_rows(results: list[SampledTrainingResult]) -> list[dict]:
    rows: list[dict] = []
    for result in results:
        for metric in ("population_return", "q", "pi1_a0"):
            mean, lower, upper = _ci(result.metrics[metric][-1])
            rows.append(
                {
                    "initialization": result.config.initialization,
                    "switch_time": result.config.handoff_update,
                    "metric": metric,
                    "mean": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                    "n_seeds": result.config.n_seeds,
                }
            )
    return rows


def checkpoint_diagnostic_rows(results: list[SampledTrainingResult]) -> list[dict]:
    rows: list[dict] = []
    for result in results:
        switch = result.config.handoff_update
        checkpoints = (
            ("switch", switch),
            ("switch_plus_250", switch + 250),
            ("final", result.config.updates),
        )
        for checkpoint_name, checkpoint in checkpoints:
            matches = np.flatnonzero(result.steps == checkpoint)
            if len(matches) != 1:
                raise ValueError(f"checkpoint {checkpoint} is not recorded")
            index = int(matches[0])
            q = result.metrics["q"][index]
            good = result.metrics["pi1_a0"][index]
            population_return = result.metrics["population_return"][index]
            finite = np.isfinite(q) & np.isfinite(good)
            near_optimal = finite & (q >= 0.9) & (good >= 0.9)
            failure = (~near_optimal).astype(np.float64)
            numerical_failure = (~finite).astype(np.float64)
            rank = result.metrics["empirical_fisher_rank"][index].astype(np.float64)
            full_rank = (rank == 4).astype(np.float64)
            fail_mean, fail_low, fail_high = _ci(failure)
            numerical_mean, numerical_low, numerical_high = _ci(numerical_failure)
            rank_mean, rank_low, rank_high = _ci(rank)
            full_mean, full_low, full_high = _ci(full_rank)
            return_mean, return_low, return_high = _ci(population_return)
            q_mean, q_low, q_high = _ci(q)
            good_mean, good_low, good_high = _ci(good)
            rows.append(
                {
                    "initialization": result.config.initialization,
                    "switch_time": switch,
                    "checkpoint": checkpoint_name,
                    "update": checkpoint,
                    "population_return_mean": return_mean,
                    "population_return_ci_lower": return_low,
                    "population_return_ci_upper": return_high,
                    "q_mean": q_mean,
                    "q_ci_lower": q_low,
                    "q_ci_upper": q_high,
                    "pi1_a0_mean": good_mean,
                    "pi1_a0_ci_lower": good_low,
                    "pi1_a0_ci_upper": good_high,
                    "near_optimal_failure_rate": fail_mean,
                    "failure_ci_lower": fail_low,
                    "failure_ci_upper": fail_high,
                    "numerical_failure_rate": numerical_mean,
                    "numerical_failure_ci_lower": numerical_low,
                    "numerical_failure_ci_upper": numerical_high,
                    "empirical_fisher_rank_mean": rank_mean,
                    "rank_ci_lower": rank_low,
                    "rank_ci_upper": rank_high,
                    "empirical_fisher_full_rank_fraction": full_mean,
                    "full_rank_ci_lower": full_low,
                    "full_rank_ci_upper": full_high,
                    "n_seeds": result.config.n_seeds,
                }
            )
    return rows


def paired_switch_rows(results: list[SampledTrainingResult]) -> list[dict]:
    """Paired final differences relative to the declared switch at 2000."""
    rows: list[dict] = []
    for initialization in INITIALIZATIONS:
        selected = {
            result.config.handoff_update: result
            for result in results
            if result.config.initialization == initialization
        }
        reference = selected[2000]
        for switch, result in sorted(selected.items()):
            if switch == 2000:
                continue
            for metric in ("population_return", "q", "pi1_a0"):
                difference = result.metrics[metric][-1] - reference.metrics[metric][-1]
                mean, lower, upper = _ci(difference)
                rows.append(
                    {
                        "initialization": initialization,
                        "switch_time": switch,
                        "reference_switch_time": 2000,
                        "metric": metric,
                        "mean_paired_difference": mean,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "n_pairs": result.config.n_seeds,
                    }
                )
    return rows


def make_plots(
    output: Path, endpoints: list[dict], diagnostics: list[dict]
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    metrics = (
        ("population_return", "$J_{4000}$"),
        ("q", "$q_{4000}$"),
        ("pi1_a0", "$\\pi_1(a_0)_{4000}$"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for initialization in INITIALIZATIONS:
        for ax, (metric, title) in zip(axes, metrics):
            selected = [
                row for row in endpoints
                if row["initialization"] == initialization and row["metric"] == metric
            ]
            selected.sort(key=lambda row: row["switch_time"])
            x = np.asarray([row["switch_time"] for row in selected])
            mean = np.asarray([row["mean"] for row in selected])
            low = np.asarray([row["ci_lower"] for row in selected])
            high = np.asarray([row["ci_upper"] for row in selected])
            ax.plot(x, mean, marker="o", label=initialization)
            ax.fill_between(x, low, high, alpha=0.15)
            ax.set_title(title)
            ax.set_xlabel("switch update")
            ax.grid(alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(output / "final_endpoints.png", dpi=180)
    plt.close(fig)

    checkpoint_order = ("switch", "switch_plus_250", "final")
    checkpoint_labels = {
        "switch": "at switch",
        "switch_plus_250": "switch + 250",
        "final": "final",
    }
    for field, ylabel, filename in (
        ("near_optimal_failure_rate", "failure rate", "failure_rate.png"),
        ("empirical_fisher_rank_mean", "mean empirical Fisher rank", "fisher_rank.png"),
        (
            "empirical_fisher_full_rank_fraction",
            "empirical Fisher full-rank fraction",
            "fisher_full_rank_fraction.png",
        ),
    ):
        height = 4.8 if field == "empirical_fisher_rank_mean" else 3.8
        fig, axes = plt.subplots(1, 2, figsize=(10, height), sharey=True)
        for ax, initialization in zip(axes, INITIALIZATIONS):
            for checkpoint in checkpoint_order:
                selected = [
                    row for row in diagnostics
                    if row["initialization"] == initialization
                    and row["checkpoint"] == checkpoint
                ]
                selected.sort(key=lambda row: row["switch_time"])
                ax.plot(
                    [row["switch_time"] for row in selected],
                    [row[field] for row in selected],
                    marker="o",
                    label=checkpoint_labels[checkpoint],
                )
            ax.set_title(initialization)
            ax.set_xlabel("switch update")
            ax.grid(alpha=0.25)
            if field == "empirical_fisher_rank_mean":
                ax.set_ylim(0, 4.1)
            else:
                ax.set_ylim(-0.02, 1.05)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(output / filename, dpi=180)
        plt.close(fig)


def run_switch_sweep(output: Path, *, resume: bool = False) -> list[SampledTrainingResult]:
    suite = manifest()
    prepare_output(output, suite, resume)
    verification = run_verification()
    (output / "verification.json").write_text(
        json.dumps(
            {"schema_version": 1, "tolerances": TOLERANCES, **verification.to_dict()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not verification.passed:
        raise ValueError("sampled estimator verification failed")

    results: list[SampledTrainingResult] = []
    status: list[dict] = []
    for initialization in INITIALIZATIONS:
        for switch_time in SWITCH_TIMES:
            config = _config(initialization, switch_time, suite)
            print(
                f"Running {initialization}: switch={switch_time} "
                f"({config.n_seeds} seeds, {config.updates} updates)",
                flush=True,
            )
            result, skipped = _run_unit(output, "switch_sweep", config, resume)
            results.append(result)
            status.append(
                {
                    "initialization": initialization,
                    "switch_time": switch_time,
                    "skipped": int(skipped),
                    "finite_fraction": float(result.finite.mean()),
                }
            )

    endpoints = final_endpoint_rows(results)
    diagnostics = checkpoint_diagnostic_rows(results)
    write_rows(output / "final_endpoints.csv", endpoints)
    write_rows(output / "checkpoint_diagnostics.csv", diagnostics)
    write_rows(output / "paired_vs_switch_2000.csv", paired_switch_rows(results))
    write_rows(output / "run_status.csv", status)
    make_plots(output / "plots", endpoints, diagnostics)
    print(f"Completed {len(results)} switch-sweep units in {output}", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_switch_sweep(args.output_dir, resume=args.resume)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
