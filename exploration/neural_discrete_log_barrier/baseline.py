"""Baseline-gated Acrobot GPOMDP learning-rate experiment.

This module deliberately excludes every barrier and NPG method.  Its only job
is to establish that the sampled GPOMDP/REINFORCE implementation learns on
Acrobot before any regularizer comparison is interpreted.
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from vpg.stats import mean_confidence_interval

from .training import NeuralTrainingConfig, train_policy


BASELINE_DIRECTORY_NAME = "acrobot_gpomdp_baseline"
LEARNING_RATES = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
SCREEN_SEEDS = (191, 192)
CONFIRMATION_SEEDS = (211, 212, 213, 214, 215)
EPISODES_PER_UPDATE = 8
SCREEN_UPDATES = 300
CONTINUATION_UPDATES = 1000


def _learning_rate_label(value: float) -> str:
    return f"lr_{value:.0e}".replace("-", "m").replace("+", "p")


def _checkpoint_fractions(updates: int, interval: int = 50) -> tuple[float, ...]:
    checkpoints = set(range(0, updates + 1, interval))
    checkpoints.add(updates)
    return tuple(value / updates for value in sorted(checkpoints))


def baseline_config(seed: int, learning_rate: float, updates: int) -> NeuralTrainingConfig:
    """Return the frozen GPOMDP protocol used in screening and confirmation."""

    return NeuralTrainingConfig(
        environment="Acrobot-v1",
        method="gpomdp_reward_only",
        seed=seed,
        hidden_sizes=(8, 8),
        learning_rate=learning_rate,
        gamma=0.99,
        updates=updates,
        # Unused in the episode/update collector, retained for compatibility
        # with the shared training configuration.
        batch_steps=500,
        horizon=500,
        center_returns=True,
        normalize_returns=False,
        beta=0.0,
        entropy_coefficient=0.0,
        evaluation_episodes=32,
        checkpoint_fractions=_checkpoint_fractions(updates),
        collector_mode="complete_episodes_by_update",
        parallel_environments=8,
        episodes_per_update=EPISODES_PER_UPDATE,
    )


def _train_task(config: NeuralTrainingConfig, output_directory: str) -> dict:
    torch.set_num_threads(1)
    return train_policy(config, output_directory)


def _run_tasks(
    tasks: list[tuple[NeuralTrainingConfig, Path]],
    *,
    parallel_workers: int,
) -> list[dict]:
    summaries: list[dict] = []
    if parallel_workers <= 1:
        for index, (config, output) in enumerate(tasks, start=1):
            summaries.append(train_policy(config, output))
            print(f"completed {index}/{len(tasks)}: lr={config.learning_rate:g}, seed={config.seed}")
        return summaries

    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {
            executor.submit(_train_task, config, str(output)): config
            for config, output in tasks
        }
        for index, future in enumerate(as_completed(futures), start=1):
            config = futures[future]
            summaries.append(future.result())
            print(f"completed {index}/{len(tasks)}: lr={config.learning_rate:g}, seed={config.seed}")
    return summaries


def _read_behavior(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: int(row["update"]))


def _run_metrics(run_directory: Path) -> dict:
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    rows = _read_behavior(run_directory / "checkpoint_behavior.csv")
    updates = np.asarray([int(row["update"]) for row in rows], dtype=np.float64)
    stochastic = np.asarray([float(row["stochastic_return"]) for row in rows])
    deterministic = np.asarray([float(row["deterministic_return"]) for row in rows])
    terminations = np.asarray([
        float(row["stochastic_termination_rate"]) for row in rows
    ])
    if updates[-1] == updates[0]:
        stochastic_auc = float(stochastic[-1])
    else:
        stochastic_auc = float(
            np.trapezoid(stochastic, updates) / (updates[-1] - updates[0])
        )
    return {
        "seed": int(summary["seed"]),
        "learning_rate": float(summary["learning_rate"]),
        "finite": bool(summary["finite"]),
        "actual_optimizer_updates": int(summary["actual_optimizer_updates"]),
        "actual_training_episodes": int(summary["actual_training_episodes"]),
        "environment_steps": int(summary["total_environment_steps"]),
        "initial_stochastic_return": float(stochastic[0]),
        "final_stochastic_return": float(stochastic[-1]),
        "stochastic_improvement": float(stochastic[-1] - stochastic[0]),
        "final_deterministic_return": float(deterministic[-1]),
        "final_stochastic_termination_rate": float(terminations[-1]),
        "stochastic_return_auc": stochastic_auc,
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _aggregate_candidates(run_rows: list[dict]) -> list[dict]:
    candidates: list[dict] = []
    for learning_rate in sorted({float(row["learning_rate"]) for row in run_rows}):
        group = [row for row in run_rows if float(row["learning_rate"]) == learning_rate]
        candidates.append({
            "learning_rate": learning_rate,
            "seed_count": len(group),
            "all_finite": all(bool(row["finite"]) for row in group),
            "median_stochastic_return_auc": float(np.median([
                row["stochastic_return_auc"] for row in group
            ])),
            "mean_final_stochastic_return": float(np.mean([
                row["final_stochastic_return"] for row in group
            ])),
            "median_final_stochastic_return": float(np.median([
                row["final_stochastic_return"] for row in group
            ])),
            "minimum_final_stochastic_return": float(np.min([
                row["final_stochastic_return"] for row in group
            ])),
            "mean_stochastic_improvement": float(np.mean([
                row["stochastic_improvement"] for row in group
            ])),
            "minimum_stochastic_improvement": float(np.min([
                row["stochastic_improvement"] for row in group
            ])),
            "mean_final_termination_rate": float(np.mean([
                row["final_stochastic_termination_rate"] for row in group
            ])),
        })
    candidates.sort(
        key=lambda row: (
            bool(row["all_finite"]),
            float(row["median_stochastic_return_auc"]),
            float(row["median_final_stochastic_return"]),
            -float(row["learning_rate"]),
        ),
        reverse=True,
    )
    for rank, row in enumerate(candidates, start=1):
        row["rank"] = rank
    return candidates


def _write_protocol(root: Path) -> None:
    protocol = {
        "schema_version": 1,
        "environment": "Acrobot-v1",
        "method": "GPOMDP / reward-to-go REINFORCE",
        "architecture": [8, 8],
        "gamma": 0.99,
        "optimizer": "Adam",
        "center_returns": True,
        "normalize_returns": False,
        "entropy_coefficient": 0.0,
        "episodes_per_update": EPISODES_PER_UPDATE,
        "evaluation_episodes_per_checkpoint": 32,
        "evaluation_interval_updates": 50,
        "learning_rate_candidates": list(LEARNING_RATES),
        "screen_seeds": list(SCREEN_SEEDS),
        "screen_updates": SCREEN_UPDATES,
        "continuation_updates": CONTINUATION_UPDATES,
        "confirmation_seeds": list(CONFIRMATION_SEEDS),
        "reward_reference": {
            "random_policy_mean_1000_episodes": -499.239,
            "random_policy_goal_termination_rate": 0.013,
            "official_solved_threshold": -100.0,
        },
        "gate": {
            "at_least_four_of_five_improve_by": 100.0,
            "median_final_stochastic_return_at_least": -300.0,
            "median_final_stochastic_termination_rate_at_least": 0.8,
            "paired_improvement_95pct_ci_lower_bound_above": 0.0,
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "protocol.json").write_text(
        json.dumps(protocol, indent=2, sort_keys=True), encoding="utf-8"
    )


def run_learning_rate_screen(
    output_root: Path,
    *,
    parallel_workers: int = 2,
) -> dict:
    root = output_root / BASELINE_DIRECTORY_NAME
    _write_protocol(root)
    stage = root / "lr_screen_300_updates"
    tasks = [
        (
            baseline_config(seed, learning_rate, SCREEN_UPDATES),
            stage / "runs" / _learning_rate_label(learning_rate) / f"seed_{seed:03d}",
        )
        for learning_rate in LEARNING_RATES
        for seed in SCREEN_SEEDS
    ]
    _run_tasks(tasks, parallel_workers=parallel_workers)
    run_rows = [_run_metrics(path) for _, path in tasks]
    candidates = _aggregate_candidates(run_rows)
    top_two = [float(row["learning_rate"]) for row in candidates[:2]]
    result = {
        "schema_version": 1,
        "stage": "learning_rate_screen",
        "complete": True,
        "run_count": len(run_rows),
        "updates_per_run": SCREEN_UPDATES,
        "episodes_per_run": SCREEN_UPDATES * EPISODES_PER_UPDATE,
        "top_two_learning_rates": top_two,
        "ranking_rule": (
            "finite first, then median stochastic evaluation-return AUC, then median "
            "final stochastic return; prefer the smaller rate only on an exact tie"
        ),
        "candidates": candidates,
    }
    _write_csv(stage / "seed_results.csv", run_rows)
    _write_csv(stage / "candidate_results.csv", candidates)
    (stage / "screen_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _load_screen(output_root: Path) -> dict:
    path = (
        output_root
        / BASELINE_DIRECTORY_NAME
        / "lr_screen_300_updates"
        / "screen_result.json"
    )
    if not path.exists():
        raise FileNotFoundError("run the Acrobot learning-rate screen first")
    return json.loads(path.read_text(encoding="utf-8"))


def run_learning_rate_continuation(
    output_root: Path,
    *,
    parallel_workers: int = 2,
) -> dict:
    root = output_root / BASELINE_DIRECTORY_NAME
    screen = _load_screen(output_root)
    candidates = tuple(float(value) for value in screen["top_two_learning_rates"])
    stage = root / "lr_continuation_1000_updates"
    tasks = [
        (
            baseline_config(seed, learning_rate, CONTINUATION_UPDATES),
            stage / "runs" / _learning_rate_label(learning_rate) / f"seed_{seed:03d}",
        )
        for learning_rate in candidates
        for seed in SCREEN_SEEDS
    ]
    _run_tasks(tasks, parallel_workers=parallel_workers)
    run_rows = [_run_metrics(path) for _, path in tasks]
    ranked = _aggregate_candidates(run_rows)
    viable = [
        row for row in ranked
        if row["all_finite"]
        and row["minimum_final_stochastic_return"] > -450.0
        and row["minimum_stochastic_improvement"] > 0.0
    ]
    provisional = float(viable[0]["learning_rate"]) if viable else None
    result = {
        "schema_version": 1,
        "stage": "learning_rate_continuation",
        "complete": True,
        "run_count": len(run_rows),
        "updates_per_run": CONTINUATION_UPDATES,
        "episodes_per_run": CONTINUATION_UPDATES * EPISODES_PER_UPDATE,
        "ranked_candidates": ranked,
        "provisional_learning_rate": provisional,
        "viability_rule": (
            "both screening seeds finite, final stochastic return above -450, "
            "and positive initial-to-final improvement"
        ),
    }
    _write_csv(stage / "seed_results.csv", run_rows)
    _write_csv(stage / "candidate_results.csv", ranked)
    (stage / "continuation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _load_continuation(output_root: Path) -> dict:
    path = (
        output_root
        / BASELINE_DIRECTORY_NAME
        / "lr_continuation_1000_updates"
        / "continuation_result.json"
    )
    if not path.exists():
        raise FileNotFoundError("run the Acrobot learning-rate continuation first")
    return json.loads(path.read_text(encoding="utf-8"))


def run_gpomdp_confirmation(
    output_root: Path,
    *,
    parallel_workers: int = 2,
) -> dict:
    root = output_root / BASELINE_DIRECTORY_NAME
    continuation = _load_continuation(output_root)
    selected = continuation["provisional_learning_rate"]
    if selected is None:
        raise RuntimeError(
            "no learning rate passed the two-seed viability gate; do not run method ablations"
        )
    learning_rate = float(selected)
    stage = root / "gpomdp_confirmation_1000_updates"
    tasks = [
        (
            baseline_config(seed, learning_rate, CONTINUATION_UPDATES),
            stage / "runs" / _learning_rate_label(learning_rate) / f"seed_{seed:03d}",
        )
        for seed in CONFIRMATION_SEEDS
    ]
    _run_tasks(tasks, parallel_workers=parallel_workers)
    run_rows = [_run_metrics(path) for _, path in tasks]
    improvements = np.asarray([row["stochastic_improvement"] for row in run_rows])
    improvement_mean, improvement_low, improvement_high = mean_confidence_interval(improvements)
    median_final = float(np.median([row["final_stochastic_return"] for row in run_rows]))
    median_termination = float(np.median([
        row["final_stochastic_termination_rate"] for row in run_rows
    ]))
    improved_seed_count = int(np.sum(improvements >= 100.0))
    gate_checks = {
        "all_finite": all(bool(row["finite"]) for row in run_rows),
        "four_of_five_improve_by_100": improved_seed_count >= 4,
        "median_final_return_at_least_minus_300": median_final >= -300.0,
        "median_termination_rate_at_least_0p8": median_termination >= 0.8,
        "improvement_ci_lower_above_zero": float(improvement_low) > 0.0,
    }
    result = {
        "schema_version": 1,
        "stage": "gpomdp_confirmation",
        "complete": True,
        "learning_rate": learning_rate,
        "seed_count": len(run_rows),
        "updates_per_seed": CONTINUATION_UPDATES,
        "episodes_per_seed": CONTINUATION_UPDATES * EPISODES_PER_UPDATE,
        "median_final_stochastic_return": median_final,
        "median_final_stochastic_termination_rate": median_termination,
        "improved_seed_count_at_least_100": improved_seed_count,
        "mean_stochastic_improvement": float(improvement_mean),
        "improvement_95pct_ci": [float(improvement_low), float(improvement_high)],
        "gate_checks": gate_checks,
        "gpomdp_baseline_gate_passed": all(gate_checks.values()),
        "ablation_authorized_by_gate": all(gate_checks.values()),
    }
    _write_csv(stage / "seed_results.csv", run_rows)
    (stage / "confirmation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
