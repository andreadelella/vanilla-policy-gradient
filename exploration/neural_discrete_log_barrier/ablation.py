"""Episode-based GPOMDP regularizer ablation after the Acrobot baseline gate."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from vpg.data_collection import collect_parallel_trajectories
from vpg.gpomdp import compute_gpomdp_loss
from vpg.stats import mean_confidence_interval

from .barrier import categorical_log_barrier
from .baseline import BASELINE_DIRECTORY_NAME, baseline_config
from .training import (
    NeuralTrainingConfig,
    _flat_gradient,
    _flatten_valid_states,
    build_seeded_policy,
    train_policy,
)


ABLATION_DIRECTORY_NAME = "acrobot_gpomdp_regularizer_ablation"
CALIBRATION_SEEDS = (221, 222, 223, 224, 225)
ABLATION_SEEDS = (401, 402, 403, 404, 405)
TARGET_INITIAL_GRADIENT_RATIO = 0.3
HANDOFF_FRACTION = 0.25


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


def _vector_environments(config: NeuralTrainingConfig):
    def make_factory(index: int):
        def factory():
            env = gym.make(config.environment)
            env.reset(seed=config.seed * 100_000 + index)
            env.action_space.seed(config.seed * 100_000 + index + 50_000)
            return env

        return factory

    return gym.vector.SyncVectorEnv([
        make_factory(index) for index in range(config.parallel_environments)
    ])


def calibrate_episode_regularizers(output_root: Path) -> dict:
    """Match barrier and entropy gradient norms without looking at outcomes."""

    root = output_root / ABLATION_DIRECTORY_NAME / "calibration"
    result_path = root / "regularizer_calibration.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    barrier_ratios: list[float] = []
    entropy_ratios: list[float] = []
    for seed in CALIBRATION_SEEDS:
        config = baseline_config(seed, learning_rate=3e-3, updates=5)
        policy, _ = build_seeded_policy(config)
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        environments = _vector_environments(config)
        try:
            for update in range(5):
                trajectories = collect_parallel_trajectories(
                    environments,
                    policy,
                    n_trajectories_per_env=1,
                    clip_actions=False,
                    device="cpu",
                )
                states = _flatten_valid_states(trajectories)
                reward_objective = -compute_gpomdp_loss(
                    policy,
                    trajectories,
                    gamma=config.gamma,
                    center_returns=config.center_returns,
                    normalize_returns=config.normalize_returns,
                    entropy_coeff=0.0,
                    device="cpu",
                )
                barrier, _ = categorical_log_barrier(policy(states.detach()))
                entropy = policy.distribution(states.detach()).entropy().mean()
                reward_gradient = _flat_gradient(reward_objective, policy, retain_graph=True)
                barrier_gradient = _flat_gradient(barrier, policy, retain_graph=True)
                entropy_gradient = _flat_gradient(entropy, policy, retain_graph=True)
                reward_norm = float(reward_gradient.norm())
                barrier_norm = float(barrier_gradient.norm())
                entropy_norm = float(entropy_gradient.norm())
                barrier_ratio = barrier_norm / reward_norm if reward_norm > 0.0 else float("nan")
                entropy_ratio = entropy_norm / reward_norm if reward_norm > 0.0 else float("nan")
                if np.isfinite(barrier_ratio) and barrier_ratio > 0.0:
                    barrier_ratios.append(barrier_ratio)
                if np.isfinite(entropy_ratio) and entropy_ratio > 0.0:
                    entropy_ratios.append(entropy_ratio)
                rows.append({
                    "seed": seed,
                    "update": update,
                    "training_episodes": len(trajectories),
                    "environment_steps": sum(len(item.rewards) for item in trajectories),
                    "reward_gradient_norm": reward_norm,
                    "unscaled_barrier_gradient_norm": barrier_norm,
                    "unscaled_entropy_gradient_norm": entropy_norm,
                    "unscaled_barrier_to_reward_ratio": barrier_ratio,
                    "unscaled_entropy_to_reward_ratio": entropy_ratio,
                })
                optimizer.zero_grad()
                (-reward_objective).backward()
                optimizer.step()
        finally:
            environments.close()

    if not barrier_ratios or not entropy_ratios:
        raise RuntimeError("regularizer calibration produced no finite nonzero ratios")
    median_barrier_ratio = float(np.median(barrier_ratios))
    median_entropy_ratio = float(np.median(entropy_ratios))
    beta = TARGET_INITIAL_GRADIENT_RATIO / median_barrier_ratio
    entropy_coefficient = TARGET_INITIAL_GRADIENT_RATIO / median_entropy_ratio
    result = {
        "schema_version": 1,
        "calibration_seeds": list(CALIBRATION_SEEDS),
        "calibration_updates_per_seed": 5,
        "episodes_per_update": 8,
        "learning_rate": 3e-3,
        "target_median_regularizer_to_reward_gradient_norm": TARGET_INITIAL_GRADIENT_RATIO,
        "median_unscaled_barrier_to_reward_ratio": median_barrier_ratio,
        "median_unscaled_entropy_to_reward_ratio": median_entropy_ratio,
        "selected_beta": beta,
        "selected_entropy_coefficient": entropy_coefficient,
        "outcomes_used_for_selection": False,
        "finite": bool(np.isfinite(beta) and np.isfinite(entropy_coefficient)),
    }
    _write_csv(root / "early_gradient_audit.csv", rows)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _train_task(config: NeuralTrainingConfig, output_directory: str) -> dict:
    torch.set_num_threads(1)
    return train_policy(config, output_directory)


def _read_behavior(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return sorted(rows, key=lambda row: int(row["update"]))


def _endpoint(run_directory: Path, label: str) -> dict:
    summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
    rows = _read_behavior(run_directory / "checkpoint_behavior.csv")
    updates = np.asarray([int(row["update"]) for row in rows], dtype=np.float64)
    returns = np.asarray([float(row["stochastic_return"]) for row in rows])
    auc = float(np.trapezoid(returns, updates) / (updates[-1] - updates[0]))
    return {
        "run_label": label,
        "method": summary["method"],
        "seed": int(summary["seed"]),
        "finite": bool(summary["finite"]),
        "learning_rate": float(summary["learning_rate"]),
        "final_stochastic_return": float(rows[-1]["stochastic_return"]),
        "final_deterministic_return": float(rows[-1]["deterministic_return"]),
        "final_stochastic_termination_rate": float(rows[-1]["stochastic_termination_rate"]),
        "stochastic_return_auc": auc,
        "environment_steps": int(summary["total_environment_steps"]),
        "optimizer_updates": int(summary["actual_optimizer_updates"]),
        "training_episodes": int(summary["actual_training_episodes"]),
    }


def _paired_differences(endpoints: list[dict]) -> list[dict]:
    reward = {
        int(row["seed"]): row
        for row in endpoints
        if row["run_label"] == "reward_only"
    }
    rows: list[dict] = []
    for label in sorted({row["run_label"] for row in endpoints} - {"reward_only"}):
        method = {int(row["seed"]): row for row in endpoints if row["run_label"] == label}
        shared = sorted(set(reward) & set(method))
        for metric in (
            "final_stochastic_return",
            "stochastic_return_auc",
            "final_stochastic_termination_rate",
            "environment_steps",
        ):
            differences = np.asarray([
                float(method[seed][metric]) - float(reward[seed][metric])
                for seed in shared
            ])
            mean, low, high = mean_confidence_interval(differences)
            rows.append({
                "run_label": label,
                "metric": metric,
                "paired_seed_count": len(shared),
                "mean_difference_method_minus_reward_only": float(mean),
                "ci95_low": float(low),
                "ci95_high": float(high),
            })
    return rows


def run_gpomdp_regularizer_ablation(
    output_root: Path,
    *,
    parallel_workers: int = 2,
) -> dict:
    confirmation_path = (
        output_root
        / BASELINE_DIRECTORY_NAME
        / "gpomdp_confirmation_1000_updates"
        / "confirmation_result.json"
    )
    if not confirmation_path.exists():
        raise FileNotFoundError("run the GPOMDP confirmation before the ablation")
    confirmation = json.loads(confirmation_path.read_text(encoding="utf-8"))
    if not confirmation.get("gpomdp_baseline_gate_passed", False):
        raise RuntimeError("GPOMDP did not pass its baseline gate; ablation remains closed")

    calibration = calibrate_episode_regularizers(output_root)
    root = output_root / ABLATION_DIRECTORY_NAME
    stage = root / "five_seed_1000_updates"
    methods = (
        ("reward_only", "gpomdp_reward_only", None),
        ("entropy_fixed", "gpomdp_entropy_fixed", None),
        ("logbarrier_fixed", "gpomdp_logbarrier_fixed", None),
        ("logbarrier_handoff_h25", "gpomdp_logbarrier_handoff", HANDOFF_FRACTION),
    )
    tasks: list[tuple[NeuralTrainingConfig, Path, str]] = []
    for seed in ABLATION_SEEDS:
        for label, method, handoff_fraction in methods:
            config = replace(
                baseline_config(seed, learning_rate=3e-3, updates=1000),
                method=method,
                beta=(
                    float(calibration["selected_beta"])
                    if "logbarrier" in method
                    else 0.0
                ),
                entropy_coefficient=(
                    float(calibration["selected_entropy_coefficient"])
                    if method == "gpomdp_entropy_fixed"
                    else 0.0
                ),
                handoff_fraction=handoff_fraction,
            )
            tasks.append((config, stage / "runs" / label / f"seed_{seed:03d}", label))

    pending_tasks = [
        task for task in tasks if not (task[1] / "summary.json").exists()
    ]
    print(
        f"ablation status: {len(tasks) - len(pending_tasks)}/{len(tasks)} complete; "
        f"{len(pending_tasks)} pending",
        flush=True,
    )

    if parallel_workers <= 1:
        for index, (config, path, label) in enumerate(pending_tasks, start=1):
            train_policy(config, path)
            print(
                f"completed {index}/{len(pending_tasks)} pending: "
                f"{label}, seed={config.seed}",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(_train_task, config, str(path)): (config, label)
                for config, path, label in pending_tasks
            }
            for index, future in enumerate(as_completed(futures), start=1):
                config, label = futures[future]
                future.result()
                print(
                    f"completed {index}/{len(pending_tasks)} pending: "
                    f"{label}, seed={config.seed}",
                    flush=True,
                )

    endpoints = [_endpoint(path, label) for _, path, label in tasks]
    paired = _paired_differences(endpoints)
    result = {
        "schema_version": 1,
        "complete": True,
        "learning_rate": 3e-3,
        "updates_per_seed_method": 1000,
        "episodes_per_seed_method": 8000,
        "seeds": list(ABLATION_SEEDS),
        "methods": [item[0] for item in methods],
        "calibration": calibration,
        "handoff_fraction": HANDOFF_FRACTION,
        "npg_included": False,
        "npg_exclusion_reason": (
            "NPG requires a separate learning-rate and damping baseline gate; "
            "the old hard-coded rate is not reused"
        ),
        "scientific_winner_asserted": False,
    }
    _write_csv(stage / "seed_endpoints.csv", endpoints)
    _write_csv(stage / "paired_differences.csv", paired)
    (stage / "ablation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
