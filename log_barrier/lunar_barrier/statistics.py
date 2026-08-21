"""Selection and paired reliability summaries."""

from __future__ import annotations

import math

import numpy as np


def _two_sided_exact_mcnemar(first_only: int, second_only: int) -> float:
    discordant = first_only + second_only
    if discordant == 0:
        return 1.0
    lower = min(first_only, second_only)
    one_sided = sum(math.comb(discordant, value) for value in range(lower + 1))
    return min(1.0, 2.0 * one_sided / (2.0**discordant))


def rank_configurations(
    results: list[dict],
    fields: tuple[str, ...],
    lower_quantile: float = 0.25,
) -> list[dict]:
    """Rank configurations by robust held-out return across seeds."""

    if not 0.0 <= lower_quantile <= 1.0:
        raise ValueError("lower_quantile must be between zero and one")
    groups: dict[tuple, list[dict]] = {}
    for result in results:
        config = result["config"]
        groups.setdefault(tuple(config[field] for field in fields), []).append(result)
    if not groups:
        raise ValueError("no smoke results found")
    ranking = []
    for group_key, rows in groups.items():
        evaluation_values = np.asarray(
            [row["stochastic_evaluation_mean"] for row in rows], dtype=np.float64
        )
        row = dict(zip(fields, group_key))
        row.update(
            seed_count=int(evaluation_values.size),
            lower_tail_score=float(np.quantile(evaluation_values, lower_quantile)),
            mean_evaluation_return=float(evaluation_values.mean()),
            evaluation_std=(
                float(evaluation_values.std(ddof=1))
                if evaluation_values.size > 1
                else 0.0
            ),
        )
        ranking.append(row)
    ranking.sort(
        key=lambda row: (
            row["lower_tail_score"],
            row["mean_evaluation_return"],
        ),
        reverse=True,
    )
    return ranking


def select_barrier_configuration(
    baseline_results: list[dict],
    barrier_results: list[dict],
    lower_quantile: float = 0.25,
) -> dict:
    baselines = {
        int(result["seed"]): float(result["stochastic_evaluation_mean"])
        for result in baseline_results
    }
    ranking = rank_configurations(
        barrier_results, ("beta", "handoff_fraction"), lower_quantile
    )
    for candidate in ranking:
        rows = [
            result
            for result in barrier_results
            if result["config"]["beta"] == candidate["beta"]
            and result["config"]["handoff_fraction"] == candidate["handoff_fraction"]
        ]
        missing = [int(row["seed"]) for row in rows if int(row["seed"]) not in baselines]
        if missing:
            raise ValueError(f"missing reward-only pairs for seeds: {missing}")
        differences = np.asarray(
            [
                row["stochastic_evaluation_mean"] - baselines[int(row["seed"])]
                for row in rows
            ],
            dtype=np.float64,
        )
        candidate["mean_paired_difference"] = float(differences.mean())
    ranking.sort(
        key=lambda row: (
            row["lower_tail_score"],
            row["mean_evaluation_return"],
            row["mean_paired_difference"],
        ),
        reverse=True,
    )
    return {"selected": ranking[0], "ranking": ranking, "lower_quantile": lower_quantile}


def paired_reliability_summary(
    results: list[dict],
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 20260821,
    failure_threshold: float = -100.0,
) -> dict:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    by_seed: dict[int, dict[str, dict]] = {}
    for result in results:
        by_seed.setdefault(int(result["seed"]), {})[result["method"]] = result
    incomplete = [seed for seed, methods in by_seed.items() if set(methods) != {"reward_only", "log_barrier"}]
    if incomplete:
        raise ValueError(f"incomplete method pairs for seeds: {incomplete[:10]}")
    if len(by_seed) < 2:
        raise ValueError("at least two complete seed pairs are required")

    seeds = sorted(by_seed)
    baseline = np.asarray(
        [by_seed[seed]["reward_only"]["stochastic_evaluation_mean"] for seed in seeds],
        dtype=np.float64,
    )
    barrier = np.asarray(
        [by_seed[seed]["log_barrier"]["stochastic_evaluation_mean"] for seed in seeds],
        dtype=np.float64,
    )
    differences = barrier - baseline
    evaluation_counts = {
        len(result["stochastic_evaluation_returns"]) for result in results
    }
    if len(evaluation_counts) != 1:
        raise ValueError("all runs must use the same number of evaluation episodes")
    evaluation_episodes = evaluation_counts.pop()
    baseline_failures = baseline < failure_threshold
    barrier_failures = barrier < failure_threshold
    reward_fails_barrier_survives = int(
        np.sum(baseline_failures & ~barrier_failures)
    )
    barrier_fails_reward_survives = int(
        np.sum(~baseline_failures & barrier_failures)
    )
    rng = np.random.default_rng(bootstrap_seed)
    indices = rng.integers(0, len(seeds), size=(bootstrap_samples, len(seeds)))
    bootstrap_means = differences[indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, (0.025, 0.975))

    def method_summary(values: np.ndarray) -> dict:
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "median": float(np.median(values)),
            "solved_rate_at_200": float(np.mean(values >= 200.0)),
            "catastrophic_failure_rate": float(np.mean(values < failure_threshold)),
        }

    standard_error = float(differences.std(ddof=1) / math.sqrt(differences.size))
    return {
        "seed_count": len(seeds),
        "seeds": seeds,
        "evaluation_episodes_per_policy": evaluation_episodes,
        "catastrophic_failure_threshold": failure_threshold,
        "reward_only": method_summary(baseline),
        "log_barrier": method_summary(barrier),
        "paired_difference_log_barrier_minus_reward_only": {
            "mean": float(differences.mean()),
            "median": float(np.median(differences)),
            "std": float(differences.std(ddof=1)),
            "standard_error": standard_error,
            "bootstrap_95_percent_ci": [float(low), float(high)],
            "win_rate": float(np.mean(differences > 0.0)),
            "tie_rate": float(np.mean(differences == 0.0)),
        },
        "paired_catastrophic_failures": {
            "both_fail": int(np.sum(baseline_failures & barrier_failures)),
            "neither_fails": int(np.sum(~baseline_failures & ~barrier_failures)),
            "reward_fails_barrier_survives": reward_fails_barrier_survives,
            "barrier_fails_reward_survives": barrier_fails_reward_survives,
            "exact_two_sided_mcnemar_p": _two_sided_exact_mcnemar(
                reward_fails_barrier_survives,
                barrier_fails_reward_survives,
            ),
        },
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
    }
