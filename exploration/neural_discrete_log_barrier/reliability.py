"""Seed-divergence audit and frozen Acrobot reliability confirmation."""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from vpg.stats import mean_confidence_interval

from .ablation import (
    ABLATION_DIRECTORY_NAME,
    HANDOFF_FRACTION,
    _endpoint,
    _train_task,
    calibrate_episode_regularizers,
)
from .baseline import baseline_config
from .fisher import action_enumerated_fisher_spectrum
from .training import (
    NeuralTrainingConfig,
    _flatten_valid_states,
    collect_fixed_step_trajectories,
    restore_policy,
    train_policy,
)


PRIMARY_SEEDS = tuple(range(501, 521))
CONTROL_SEEDS = tuple(range(501, 511))
FAILURE_RETURN_THRESHOLD = -300.0
FAILURE_TERMINATION_THRESHOLD = 0.8
ANALYSIS_SEEDS = (402, 403)
ANALYSIS_METHODS = ("reward_only", "logbarrier_handoff_h25")


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _config_from_archive(path: Path) -> NeuralTrainingConfig:
    stored = json.loads(path.read_text(encoding="utf-8"))
    fields = {
        key: value
        for key, value in stored.items()
        if key in NeuralTrainingConfig.__dataclass_fields__
    }
    fields["hidden_sizes"] = tuple(fields["hidden_sizes"])
    fields["checkpoint_fractions"] = tuple(fields["checkpoint_fractions"])
    return NeuralTrainingConfig(**fields)


def _true_step_map(run_directory: Path) -> dict[int, int]:
    rows = _read_csv(run_directory / "training.csv")
    result = {0: 0}
    result.update({int(row["update"]): int(row["environment_steps"]) for row in rows})
    return result


def run_pilot_divergence_analysis(output_root: Path) -> dict:
    """Reconstruct Fisher and behavior diagnostics for seeds 402 and 403."""

    pilot = output_root / ABLATION_DIRECTORY_NAME / "five_seed_1000_updates"
    stage = output_root / ABLATION_DIRECTORY_NAME / "seed_402_divergence_audit"
    stage.mkdir(parents=True, exist_ok=True)
    bank_path = output_root / "state_banks" / "acrobot_reference_states.npz"
    if not bank_path.exists():
        raise FileNotFoundError("the frozen Acrobot reference-state bank is missing")
    with np.load(bank_path) as archive:
        fixed_states = torch.as_tensor(archive["states"])

    rows: list[dict] = []
    spectra_directory = stage / "spectra"
    spectra_directory.mkdir(exist_ok=True)
    for seed in ANALYSIS_SEEDS:
        for label in ANALYSIS_METHODS:
            run = pilot / "runs" / label / f"seed_{seed:03d}"
            config = _config_from_archive(run / "config.json")
            behavior = {int(row["update"]): row for row in _read_csv(run / "checkpoint_behavior.csv")}
            gradients = {int(row["update"]): row for row in _read_csv(run / "checkpoint_gradients.csv")}
            true_steps = _true_step_map(run)
            for checkpoint in sorted((run / "checkpoints").glob("checkpoint_update_*.pt")):
                update = int(checkpoint.stem.split("_")[-1])
                if update not in behavior:
                    continue
                policy = restore_policy(config, checkpoint)
                generator = torch.Generator(device="cpu").manual_seed(
                    71_000_000 + seed * 10_000 + update
                )
                trajectories = collect_fixed_step_trajectories(
                    config.environment,
                    policy,
                    step_count=128,
                    horizon=config.horizon,
                    reset_seed_base=72_000_000 + seed * 10_000 + update,
                    action_generator=generator,
                )
                on_states = _flatten_valid_states(trajectories)
                metric_sets = {}
                for fisher_name, states in (
                    ("on_policy", on_states),
                    ("fixed_reference", fixed_states),
                ):
                    spectrum = action_enumerated_fisher_spectrum(policy, states)
                    metric_sets[fisher_name] = spectrum.metrics.to_dict()
                    np.savez_compressed(
                        spectra_directory
                        / f"{label}__seed{seed:03d}__u{update:06d}__{fisher_name}.npz",
                        eigenvalues=spectrum.eigenvalues.numpy(),
                    )
                gradient_update = min(update, config.updates - 1)
                gradient = gradients[gradient_update]
                item = behavior[update]
                row = {
                    "seed": seed,
                    "run_label": label,
                    "update": update,
                    "true_environment_steps": true_steps[update],
                    "stochastic_return": float(item["stochastic_return"]),
                    "deterministic_return": float(item["deterministic_return"]),
                    "deterministic_episode_length": float(item["episode_length"]),
                    "entropy": float(item["entropy"]),
                    "mean_min_probability": float(item["mean_min_probability"]),
                    "global_min_probability": float(item["global_min_probability"]),
                    "reward_gradient_norm": float(gradient["reward_gradient_norm"]),
                    "barrier_gradient_norm": float(gradient["barrier_gradient_norm"]),
                    "regularizer_gradient_norm": float(gradient["regularizer_gradient_norm"]),
                    "barrier_active": str(gradient["beta"]).lower() not in ("0", "0.0"),
                }
                for fisher_name, metrics in metric_sets.items():
                    for key in (
                        "numerical_rank",
                        "trace",
                        "smallest_positive_eigenvalue",
                        "largest_eigenvalue",
                        "entropy_effective_rank",
                        "log_pseudodeterminant",
                        "positive_spectrum_condition_number",
                    ):
                        row[f"{fisher_name}_{key}"] = metrics[key]
                rows.append(row)

    rows.sort(key=lambda row: (row["seed"], row["run_label"], row["update"]))
    _write_csv(stage / "checkpoint_audit.csv", rows)
    _plot_divergence(rows, stage / "seed_divergence.png")
    result = {
        "schema_version": 1,
        "complete": True,
        "seeds": list(ANALYSIS_SEEDS),
        "methods": list(ANALYSIS_METHODS),
        "on_policy_states_per_checkpoint": 128,
        "fixed_reference_bank": str(bank_path),
        "training_rerun": False,
        "checkpoint_rows": len(rows),
        "interpretation_deferred_until_rows_are_compared": True,
    }
    (stage / "analysis_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _plot_divergence(
    rows: list[dict],
    path: Path,
    *,
    seeds: tuple[int, ...] = ANALYSIS_SEEDS,
    methods: tuple[str, ...] = ANALYSIS_METHODS,
) -> None:
    fields = (
        ("stochastic_return", "stochastic return"),
        ("entropy", "entropy"),
        ("global_min_probability", "global min probability"),
        ("reward_gradient_norm", "reward-gradient norm"),
        ("on_policy_smallest_positive_eigenvalue", "on-policy smallest eigenvalue"),
        ("fixed_reference_smallest_positive_eigenvalue", "fixed-reference smallest eigenvalue"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(13, 7))
    for axis, (field, title) in zip(axes.reshape(-1), fields):
        for seed in seeds:
            for label in methods:
                subset = [row for row in rows if row["seed"] == seed and row["run_label"] == label]
                axis.plot(
                    [row["update"] for row in subset],
                    [row[field] for row in subset],
                    label=f"{seed} {label}",
                )
        if "eigenvalue" in field or "probability" in field:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("update")
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_failed_seed_diagnostics(
    output_root: Path,
    *,
    parallel_workers: int = 2,
) -> dict:
    """Outcome-selected mechanism audit for failed reward-only confirmation seeds."""

    confirmation = (
        output_root
        / ABLATION_DIRECTORY_NAME
        / "reliability_confirmation_20_seeds"
    )
    reward_root = confirmation / "runs" / "reward_only"
    handoff_root = confirmation / "runs" / "logbarrier_handoff_h25"
    if not (confirmation / "confirmation_result.json").exists():
        raise FileNotFoundError("the 20-seed reliability confirmation is incomplete")
    failed_seeds = []
    for run in sorted(reward_root.glob("seed_*")):
        if not (run / "summary.json").exists():
            continue
        endpoint = _environment_step_endpoint(run, "reward_only")
        if endpoint["failure"]:
            failed_seeds.append(int(endpoint["seed"]))
    if not failed_seeds:
        raise RuntimeError("no failed reward-only seeds were found")

    stage = confirmation / "failure_diagnostics"
    calibration = calibrate_episode_regularizers(output_root)
    fixed_tasks: list[tuple[NeuralTrainingConfig, Path]] = []
    for seed in failed_seeds:
        config = replace(
            baseline_config(seed, learning_rate=3e-3, updates=1000),
            method="gpomdp_logbarrier_fixed",
            beta=float(calibration["selected_beta"]),
        )
        fixed_tasks.append((
            config,
            stage / "runs" / "logbarrier_fixed" / f"seed_{seed:03d}",
        ))
    pending = [task for task in fixed_tasks if not (task[1] / "summary.json").exists()]
    if parallel_workers <= 1:
        for config, path in pending:
            train_policy(config, path)
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(_train_task, config, str(path)): config
                for config, path in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                config = futures[future]
                future.result()
                print(
                    f"failed-seed diagnostic {index}/{len(pending)}: "
                    f"fixed barrier, seed={config.seed}",
                    flush=True,
                )

    bank_path = output_root / "state_banks" / "acrobot_reference_states.npz"
    with np.load(bank_path) as archive:
        fixed_states = torch.as_tensor(archive["states"])
    methods = (
        "reward_only",
        "logbarrier_handoff_h25",
        "logbarrier_fixed",
    )
    rows: list[dict] = []
    spectra_directory = stage / "spectra"
    spectra_directory.mkdir(parents=True, exist_ok=True)
    for seed in failed_seeds:
        run_directories = {
            "reward_only": reward_root / f"seed_{seed:03d}",
            "logbarrier_handoff_h25": handoff_root / f"seed_{seed:03d}",
            "logbarrier_fixed": stage / "runs" / "logbarrier_fixed" / f"seed_{seed:03d}",
        }
        for label, run in run_directories.items():
            config = _config_from_archive(run / "config.json")
            behavior = {
                int(row["update"]): row
                for row in _read_csv(run / "checkpoint_behavior.csv")
            }
            gradients = {
                int(row["update"]): row
                for row in _read_csv(run / "checkpoint_gradients.csv")
            }
            true_steps = _true_step_map(run)
            for checkpoint in sorted((run / "checkpoints").glob("checkpoint_update_*.pt")):
                update = int(checkpoint.stem.split("_")[-1])
                if update not in behavior:
                    continue
                policy = restore_policy(config, checkpoint)
                generator = torch.Generator(device="cpu").manual_seed(
                    73_000_000 + seed * 10_000 + update
                )
                trajectories = collect_fixed_step_trajectories(
                    config.environment,
                    policy,
                    step_count=128,
                    horizon=config.horizon,
                    reset_seed_base=74_000_000 + seed * 10_000 + update,
                    action_generator=generator,
                )
                on_states = _flatten_valid_states(trajectories)
                spectra = {}
                for fisher_name, states in (
                    ("on_policy", on_states),
                    ("fixed_reference", fixed_states),
                ):
                    spectrum = action_enumerated_fisher_spectrum(policy, states)
                    spectra[fisher_name] = spectrum.metrics.to_dict()
                    np.savez_compressed(
                        spectra_directory
                        / f"{label}__seed{seed:03d}__u{update:06d}__{fisher_name}.npz",
                        eigenvalues=spectrum.eigenvalues.numpy(),
                    )
                gradient = gradients[min(update, config.updates - 1)]
                item = behavior[update]
                row = {
                    "seed": seed,
                    "run_label": label,
                    "update": update,
                    "true_environment_steps": true_steps[update],
                    "stochastic_return": float(item["stochastic_return"]),
                    "deterministic_return": float(item["deterministic_return"]),
                    "deterministic_episode_length": float(item["episode_length"]),
                    "entropy": float(item["entropy"]),
                    "mean_min_probability": float(item["mean_min_probability"]),
                    "global_min_probability": float(item["global_min_probability"]),
                    "reward_gradient_norm": float(gradient["reward_gradient_norm"]),
                    "barrier_gradient_norm": float(gradient["barrier_gradient_norm"]),
                    "regularizer_gradient_norm": float(gradient["regularizer_gradient_norm"]),
                    "beta": float(gradient["beta"]),
                }
                for fisher_name, metrics in spectra.items():
                    for key in (
                        "numerical_rank",
                        "trace",
                        "smallest_positive_eigenvalue",
                        "largest_eigenvalue",
                        "entropy_effective_rank",
                        "log_pseudodeterminant",
                        "positive_spectrum_condition_number",
                    ):
                        row[f"{fisher_name}_{key}"] = metrics[key]
                rows.append(row)
    rows.sort(key=lambda row: (row["seed"], row["run_label"], row["update"]))
    _write_csv(stage / "checkpoint_audit.csv", rows)
    _plot_divergence(
        rows,
        stage / "failed_seed_diagnostics.png",
        seeds=tuple(failed_seeds),
        methods=methods,
    )
    result = {
        "schema_version": 1,
        "complete": True,
        "failed_reward_only_seeds": failed_seeds,
        "methods": list(methods),
        "post_hoc_outcome_selected": True,
        "included_in_confirmatory_statistics": False,
        "fixed_barrier_extra_runs": len(failed_seeds),
        "training_rerun_for_existing_methods": False,
        "checkpoint_rows": len(rows),
    }
    (stage / "diagnostic_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _environment_step_endpoint(run: Path, label: str) -> dict:
    endpoint = _endpoint(run, label)
    behavior = _read_csv(run / "checkpoint_behavior.csv")
    steps = np.asarray([int(row["environment_steps"]) for row in behavior], dtype=np.float64)
    returns = np.asarray([float(row["stochastic_return"]) for row in behavior])
    endpoint["environment_step_return_auc"] = float(
        np.trapezoid(returns, steps) / (steps[-1] - steps[0])
    )
    endpoint["failure"] = bool(
        endpoint["final_stochastic_return"] < FAILURE_RETURN_THRESHOLD
        or endpoint["final_stochastic_termination_rate"] < FAILURE_TERMINATION_THRESHOLD
    )
    return endpoint


def _wilson_interval(successes: int, count: int) -> tuple[float, float]:
    z = 1.959963984540054
    p = successes / count
    denominator = 1.0 + z * z / count
    center = (p + z * z / (2.0 * count)) / denominator
    half = z * np.sqrt(p * (1.0 - p) / count + z * z / (4.0 * count * count)) / denominator
    return float(center - half), float(center + half)


def _summarize_confirmation(endpoints: list[dict]) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    for label in sorted({row["run_label"] for row in endpoints}):
        group = [row for row in endpoints if row["run_label"] == label]
        failures = sum(bool(row["failure"]) for row in group)
        low, high = _wilson_interval(failures, len(group))
        summaries.append({
            "run_label": label,
            "seed_count": len(group),
            "failures": failures,
            "failure_rate": failures / len(group),
            "failure_rate_wilson95_low": low,
            "failure_rate_wilson95_high": high,
            "mean_final_stochastic_return": float(np.mean([row["final_stochastic_return"] for row in group])),
            "mean_environment_step_return_auc": float(np.mean([row["environment_step_return_auc"] for row in group])),
        })
    reward = {row["seed"]: row for row in endpoints if row["run_label"] == "reward_only"}
    paired: list[dict] = []
    for label in sorted({row["run_label"] for row in endpoints} - {"reward_only"}):
        method = {row["seed"]: row for row in endpoints if row["run_label"] == label}
        seeds = sorted(set(reward) & set(method))
        for metric in ("final_stochastic_return", "environment_step_return_auc", "failure"):
            differences = np.asarray([
                float(method[seed][metric]) - float(reward[seed][metric]) for seed in seeds
            ])
            mean, low, high = mean_confidence_interval(differences)
            paired.append({
                "run_label": label,
                "metric": metric,
                "paired_seed_count": len(seeds),
                "mean_difference_method_minus_reward_only": float(mean),
                "ci95_low": float(low),
                "ci95_high": float(high),
            })
    return summaries, paired


def run_reliability_confirmation(output_root: Path, *, parallel_workers: int = 4) -> dict:
    calibration = calibrate_episode_regularizers(output_root)
    stage = output_root / ABLATION_DIRECTORY_NAME / "reliability_confirmation_20_seeds"
    methods = (
        ("reward_only", "gpomdp_reward_only", PRIMARY_SEEDS, None),
        ("logbarrier_handoff_h25", "gpomdp_logbarrier_handoff", PRIMARY_SEEDS, HANDOFF_FRACTION),
        ("entropy_fixed", "gpomdp_entropy_fixed", CONTROL_SEEDS, None),
        ("logbarrier_fixed", "gpomdp_logbarrier_fixed", CONTROL_SEEDS, None),
    )
    tasks: list[tuple[NeuralTrainingConfig, Path, str]] = []
    for label, method, seeds, handoff in methods:
        for seed in seeds:
            config = replace(
                baseline_config(seed, learning_rate=3e-3, updates=1000),
                method=method,
                beta=float(calibration["selected_beta"]) if "logbarrier" in method else 0.0,
                entropy_coefficient=(
                    float(calibration["selected_entropy_coefficient"])
                    if method == "gpomdp_entropy_fixed" else 0.0
                ),
                handoff_fraction=handoff,
            )
            tasks.append((config, stage / "runs" / label / f"seed_{seed:03d}", label))
    pending = [task for task in tasks if not (task[1] / "summary.json").exists()]
    print(f"reliability confirmation: {len(tasks)-len(pending)}/{len(tasks)} complete; {len(pending)} pending", flush=True)
    if parallel_workers <= 1:
        for index, (config, path, label) in enumerate(pending, 1):
            train_policy(config, path)
            print(f"completed {index}/{len(pending)}: {label}, seed={config.seed}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(_train_task, config, str(path)): (config, label)
                for config, path, label in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                config, label = futures[future]
                future.result()
                print(f"completed {index}/{len(pending)}: {label}, seed={config.seed}", flush=True)

    endpoints = [_environment_step_endpoint(path, label) for _, path, label in tasks]
    summaries, paired = _summarize_confirmation(endpoints)
    _write_csv(stage / "seed_endpoints.csv", endpoints)
    _write_csv(stage / "method_summaries.csv", summaries)
    _write_csv(stage / "paired_differences.csv", paired)
    result = {
        "schema_version": 1,
        "complete": True,
        "primary_seeds": list(PRIMARY_SEEDS),
        "control_seeds": list(CONTROL_SEEDS),
        "learning_rate": 3e-3,
        "updates": 1000,
        "episodes_per_update": 8,
        "selected_beta": calibration["selected_beta"],
        "selected_entropy_coefficient": calibration["selected_entropy_coefficient"],
        "handoff_fraction": HANDOFF_FRACTION,
        "failure_definition": (
            f"final stochastic return < {FAILURE_RETURN_THRESHOLD} OR final stochastic "
            f"termination rate < {FAILURE_TERMINATION_THRESHOLD}"
        ),
        "primary_methods": ["reward_only", "logbarrier_handoff_h25"],
        "secondary_controls": ["entropy_fixed", "logbarrier_fixed"],
        "scientific_winner_asserted": False,
    }
    (stage / "confirmation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
