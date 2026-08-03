"""Staged orchestration for neural discrete-policy log-barrier experiments."""

from __future__ import annotations

import copy
import csv
import json
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .barrier import categorical_log_barrier
from .fisher import (
    action_enumerated_fisher_spectrum,
    analysis_reward_gradient,
    reward_gradient_alignment,
    save_state_bank,
    state_bank_hash,
)
from .training import (
    NeuralTrainingConfig,
    _flat_gradient,
    _flatten_valid_states,
    collect_fixed_step_trajectories,
    evaluate_policy,
    restore_policy,
    train_policy,
)
from vpg.gpomdp import compute_gpomdp_loss


ROOT = Path("exploration/results/neural_discrete_log_barrier")
PRIMARY_METHODS = (
    "gpomdp_reward_only",
    "gpomdp_entropy_fixed",
    "gpomdp_logbarrier_fixed",
    "gpomdp_logbarrier_handoff",
)


def _train_task(config: NeuralTrainingConfig, output_directory: str) -> dict:
    # Avoid multiplying PyTorch's intra-op thread pool by the number of
    # independent seed workers.
    torch.set_num_threads(1)
    return train_policy(config, output_directory)


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


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def calibrate_regularizers(
    environment: str,
    *,
    seeds: tuple[int, ...],
    learning_rate: float,
    hidden_sizes: tuple[int, ...],
    batch_steps: int,
    horizon: int,
    output_directory: Path,
    center_returns: bool,
) -> dict[str, float]:
    """Calibrate coefficients from early reward-only policies, not final seeds."""

    output_directory.mkdir(parents=True, exist_ok=True)
    result_path = output_directory / "regularizer_calibration.json"
    if result_path.exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    base_barrier_ratios: list[float] = []
    base_entropy_ratios: list[float] = []
    for seed in seeds:
        base = NeuralTrainingConfig(
            environment=environment,
            method="gpomdp_reward_only",
            seed=seed,
            hidden_sizes=hidden_sizes,
            learning_rate=learning_rate,
            updates=5,
            batch_steps=batch_steps,
            horizon=horizon,
            evaluation_episodes=1,
            center_returns=center_returns,
        )
        from .training import build_seeded_policy

        policy, _ = build_seeded_policy(base)
        optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
        generator = torch.Generator(device="cpu").manual_seed(seed + 77_000)
        for update in range(5):
            trajectories = collect_fixed_step_trajectories(
                environment,
                policy,
                step_count=batch_steps,
                horizon=horizon,
                reset_seed_base=30_000_000 + seed * 10_000 + update * 100,
                action_generator=generator,
                maximum_parallel_environments=1,
            )
            states = _flatten_valid_states(trajectories)
            reward_objective = -compute_gpomdp_loss(
                policy,
                trajectories,
                gamma=base.gamma,
                center_returns=center_returns,
                normalize_returns=False,
                entropy_coeff=0.0,
                device="cpu",
            )
            logits = policy(states)
            barrier, _ = categorical_log_barrier(logits)
            entropy = policy.distribution(states).entropy().mean()
            reward_gradient = _flat_gradient(reward_objective, policy, retain_graph=True)
            barrier_gradient = _flat_gradient(barrier, policy, retain_graph=True)
            entropy_gradient = _flat_gradient(entropy, policy, retain_graph=True)
            reward_norm = float(reward_gradient.norm())
            barrier_norm = float(barrier_gradient.norm())
            entropy_norm = float(entropy_gradient.norm())
            if reward_norm > 0.0:
                base_barrier_ratios.append(barrier_norm / reward_norm)
                base_entropy_ratios.append(entropy_norm / reward_norm)
            rows.append({
                "seed": seed,
                "update": update,
                "reward_gradient_norm": reward_norm,
                "unscaled_barrier_gradient_norm": barrier_norm,
                "unscaled_entropy_gradient_norm": entropy_norm,
                "unscaled_barrier_to_reward_ratio": barrier_norm / reward_norm if reward_norm else float("nan"),
                "unscaled_entropy_to_reward_ratio": entropy_norm / reward_norm if reward_norm else float("nan"),
            })
            optimizer.zero_grad()
            (-reward_objective).backward()
            optimizer.step()
    barrier_scale = float(np.median(base_barrier_ratios))
    entropy_scale = float(np.median(base_entropy_ratios))
    beta = 0.3 / barrier_scale
    entropy_coefficient = 0.3 / entropy_scale
    beta_candidates = [beta / 3.0, beta, beta * 3.0]
    candidate_rows = []
    for candidate in beta_candidates:
        ratios = [candidate * value for value in base_barrier_ratios]
        candidate_rows.append({
            "beta": candidate,
            "median_regularizer_to_reward_ratio": float(np.median(ratios)),
            "target_label": "weak" if candidate < beta else "strong" if candidate > beta else "moderate",
        })
    result = {
        "environment": environment,
        "pilot_seeds": list(seeds),
        "selected_beta": beta,
        "selected_entropy_coefficient": entropy_coefficient,
        "selection_target_ratio": 0.3,
        "weak_target_ratio": 0.1,
        "strong_target_ratio": 0.9,
        "candidate_beta_values": beta_candidates,
        "return_centering": center_returns,
        "return_normalization": False,
        "finite": bool(np.isfinite(beta) and np.isfinite(entropy_coefficient)),
    }
    _write_csv(output_directory / "early_gradient_audit.csv", rows)
    _write_csv(output_directory / "beta_candidates.csv", candidate_rows)
    result_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def run_suite(
    stage_directory: Path,
    base_config: NeuralTrainingConfig,
    *,
    seeds: tuple[int, ...],
    beta: float,
    entropy_coefficient: float,
    handoff_fractions: tuple[float, ...],
    include_npg: bool,
    parallel_workers: int = 1,
) -> list[dict]:
    summaries: list[dict] = []
    methods: list[tuple[str, float | None]] = [
        ("gpomdp_reward_only", None),
        ("gpomdp_entropy_fixed", None),
        ("gpomdp_logbarrier_fixed", None),
    ]
    methods.extend(("gpomdp_logbarrier_handoff", fraction) for fraction in handoff_fractions)
    if include_npg:
        methods.append(("npg_reward_only", None))
    tasks: list[tuple[NeuralTrainingConfig, Path, str]] = []
    for seed in seeds:
        for method, fraction in methods:
            label = method if fraction is None else f"{method}_h{int(round(100 * fraction)):02d}"
            config = replace(
                base_config,
                method=method,
                seed=seed,
                beta=beta if "logbarrier" in method else 0.0,
                entropy_coefficient=entropy_coefficient if method == "gpomdp_entropy_fixed" else 0.0,
                handoff_fraction=fraction,
                learning_rate=(1e-2 if method == "npg_reward_only" else base_config.learning_rate),
            )
            tasks.append((config, stage_directory / "runs" / label / f"seed_{seed:03d}", label))
    if parallel_workers <= 1:
        for config, path, label in tasks:
            summary = train_policy(config, path)
            summary["run_label"] = label
            summaries.append(summary)
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(_train_task, config, str(path)): label
                for config, path, label in tasks
            }
            for future in as_completed(futures):
                summary = future.result()
                summary["run_label"] = futures[future]
                summaries.append(summary)
    summaries.sort(key=lambda item: (int(item["seed"]), item["run_label"]))
    (stage_directory / "run_summaries.json").write_text(json.dumps(summaries, indent=2, sort_keys=True), encoding="utf-8")
    return summaries


def select_learning_rate(
    stage_directory: Path,
    base_config: NeuralTrainingConfig,
    *,
    seeds: tuple[int, ...],
    candidates: tuple[float, ...],
) -> float:
    selection_path = stage_directory / "learning_rate_selection.json"
    if selection_path.exists():
        return float(json.loads(selection_path.read_text(encoding="utf-8"))["selected_learning_rate"])
    rows: list[dict] = []
    for learning_rate in candidates:
        endpoints: list[float] = []
        for seed in seeds:
            config = replace(
                base_config,
                method="gpomdp_reward_only",
                seed=seed,
                learning_rate=learning_rate,
                beta=0.0,
                entropy_coefficient=0.0,
                handoff_fraction=None,
            )
            label = f"lr_{str(learning_rate).replace('.', 'p')}"
            summary = train_policy(config, stage_directory / "lr_pilot" / label / f"seed_{seed:03d}")
            endpoints.append(float(summary["final"]["deterministic_return"]))
        rows.append({
            "learning_rate": learning_rate,
            "mean_final_deterministic_return": float(np.mean(endpoints)),
            "finite": True,
        })
    selected = max(rows, key=lambda row: row["mean_final_deterministic_return"])["learning_rate"]
    result = {
        "candidates": list(candidates),
        "pilot_seeds": list(seeds),
        "selected_learning_rate": selected,
        "selection_rule": "largest mean final deterministic return on independent pilot seeds",
    }
    _write_csv(stage_directory / "learning_rate_candidates.csv", rows)
    selection_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return float(selected)


def build_reference_state_bank(
    pilot_directory: Path,
    output_directory: Path,
    base_config: NeuralTrainingConfig,
    *,
    states_per_source: int = 64,
) -> tuple[np.ndarray, dict]:
    bank_path = output_directory / "acrobot_reference_states.npz"
    metadata_path = bank_path.with_suffix(".json")
    if bank_path.exists() and metadata_path.exists():
        with np.load(bank_path) as archive:
            states = archive["states"]
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if state_bank_hash(states) != metadata["sha256"]:
            raise RuntimeError("fixed-reference state bank hash mismatch")
        return states, metadata

    sources = [
        ("random_initial", None),
        ("reward_only_pilot", "gpomdp_reward_only"),
        ("fixed_barrier_pilot", "gpomdp_logbarrier_fixed"),
        ("temporary_barrier_pilot", "gpomdp_logbarrier_handoff_h25"),
    ]
    arrays: list[np.ndarray] = []
    source_records: list[dict] = []
    for source_index, (name, run_label) in enumerate(sources):
        if run_label is None:
            from .training import build_seeded_policy

            policy, _ = build_seeded_policy(replace(base_config, seed=910 + source_index))
            seed = 910 + source_index
            checkpoint = "initial seeded policy"
        else:
            candidate_paths = sorted((pilot_directory / "runs" / run_label).glob("seed_*/checkpoints/*.pt"))
            if not candidate_paths:
                raise FileNotFoundError(f"pilot checkpoint missing for state-bank source {run_label}")
            path = candidate_paths[-1]
            seed = int(path.parents[1].name.split("_")[-1])
            policy = restore_policy(replace(base_config, method="gpomdp_reward_only", seed=seed), path)
            checkpoint = str(path)
        _, _, visited = evaluate_policy(
            base_config.environment,
            policy,
            episodes=8,
            horizon=base_config.horizon,
            seed_base=40_000_000 + source_index * 10_000,
            deterministic=False,
        )
        visited_array = np.asarray(visited, dtype=np.float32)
        indices = np.linspace(0, len(visited_array) - 1, states_per_source, dtype=int)
        arrays.append(visited_array[indices])
        source_records.append({"name": name, "pilot_seed": seed, "checkpoint": checkpoint, "state_count": states_per_source})
    states = np.concatenate(arrays, axis=0)
    metadata = save_state_bank(bank_path, states, {
        "schema_version": 1,
        "environment": base_config.environment,
        "construction": "equal mixture of independent pilot-policy visitation states",
        "sources": source_records,
        "confirmatory_returns_used": False,
    })
    return states, metadata


def analyze_stage(
    stage_directory: Path,
    base_config: NeuralTrainingConfig,
    fixed_reference_states: np.ndarray,
    *,
    on_policy_state_count: int = 128,
) -> tuple[list[dict], list[dict], list[dict]]:
    fisher_rows: list[dict] = []
    alignment_rows: list[dict] = []
    behavior_rows: list[dict] = []
    spectra_directory = stage_directory / "fisher_spectra"
    spectra_directory.mkdir(parents=True, exist_ok=True)
    for config_path in sorted((stage_directory / "runs").glob("*/seed_*/config.json")):
        run_directory = config_path.parent
        stored = json.loads(config_path.read_text(encoding="utf-8"))
        fields = {
            key: value for key, value in stored.items()
            if key in NeuralTrainingConfig.__dataclass_fields__
        }
        fields["hidden_sizes"] = tuple(fields["hidden_sizes"])
        fields["checkpoint_fractions"] = tuple(fields["checkpoint_fractions"])
        config = NeuralTrainingConfig(**fields)
        run_label = run_directory.parent.name
        behavior_rows.extend(_read_csv(run_directory / "checkpoint_behavior.csv"))
        for checkpoint in sorted((run_directory / "checkpoints").glob("*.pt")):
            if checkpoint.stem.startswith("checkpoint_update_"):
                update = int(checkpoint.stem.split("_")[-1])
                checkpoint_steps = update * config.batch_steps
            else:
                parts = checkpoint.stem.split("_")
                checkpoint_steps = int(parts[2])
                update = checkpoint_steps
            policy = restore_policy(config, checkpoint)
            generator = torch.Generator(device="cpu").manual_seed(60_000_000 + config.seed * 10_000 + update)
            on_trajectories = collect_fixed_step_trajectories(
                config.environment,
                policy,
                step_count=on_policy_state_count,
                horizon=config.horizon,
                reset_seed_base=61_000_000 + config.seed * 10_000 + update,
                action_generator=generator,
            )
            on_states = _flatten_valid_states(on_trajectories)
            gradient_generator = torch.Generator(device="cpu").manual_seed(62_000_000 + config.seed * 10_000 + update)
            gradient_trajectories = collect_fixed_step_trajectories(
                config.environment,
                policy,
                step_count=on_policy_state_count,
                horizon=config.horizon,
                reset_seed_base=63_000_000 + config.seed * 10_000 + update,
                action_generator=gradient_generator,
            )
            reward_gradient = analysis_reward_gradient(copy.deepcopy(policy), gradient_trajectories, config.gamma)
            for fisher_name, states in (
                ("on_policy", on_states),
                ("fixed_reference", torch.as_tensor(fixed_reference_states)),
            ):
                spectrum = action_enumerated_fisher_spectrum(copy.deepcopy(policy), states)
                alignment, arrays = reward_gradient_alignment(spectrum, reward_gradient)
                common = {
                    "environment": config.environment,
                    "run_label": run_label,
                    "method": config.method,
                    "seed": config.seed,
                    "update": update,
                    "environment_steps": checkpoint_steps,
                    "fisher": fisher_name,
                    "analysis_batch_identifier": f"analysis-{fisher_name}-{config.seed}-{update}",
                    "training_batch_reused": False,
                }
                fisher_rows.append({**common, **spectrum.metrics.to_dict()})
                alignment_rows.append({**common, **alignment.to_dict()})
                np.savez_compressed(
                    spectra_directory / f"{run_label}__seed{config.seed:03d}__u{update:06d}__{fisher_name}.npz",
                    **arrays,
                )
    _write_csv(stage_directory / "checkpoint_fisher.csv", fisher_rows)
    _write_csv(stage_directory / "checkpoint_alignment.csv", alignment_rows)
    _write_csv(stage_directory / "checkpoint_behavior_all.csv", behavior_rows)
    return fisher_rows, alignment_rows, behavior_rows


def cartpole_smoke(root: Path = ROOT) -> dict:
    stage = root / "cartpole_smoke"
    base = NeuralTrainingConfig(
        environment="CartPole-v1",
        method="gpomdp_reward_only",
        seed=101,
        hidden_sizes=(8, 8),
        learning_rate=3e-3,
        updates=30,
        batch_steps=512,
        horizon=500,
        evaluation_episodes=3,
    )
    calibration = calibrate_regularizers(
        base.environment,
        seeds=(91, 92),
        learning_rate=base.learning_rate,
        hidden_sizes=base.hidden_sizes,
        batch_steps=256,
        horizon=base.horizon,
        output_directory=stage / "calibration",
        center_returns=base.center_returns,
    )
    summaries = run_suite(
        stage,
        base,
        seeds=(101, 102),
        beta=calibration["selected_beta"],
        entropy_coefficient=calibration["selected_entropy_coefficient"],
        handoff_fractions=(0.25, 0.35),
        include_npg=False,
    )
    # A smoke-only fixed bank verifies checkpoint/Fisher plumbing. It is not the
    # Acrobot confirmatory reference bank.
    first_config_path = next((stage / "runs").glob("*/seed_*/config.json"))
    stored = json.loads(first_config_path.read_text(encoding="utf-8"))
    fields = {key: value for key, value in stored.items() if key in NeuralTrainingConfig.__dataclass_fields__}
    fields["hidden_sizes"] = tuple(fields["hidden_sizes"])
    fields["checkpoint_fractions"] = tuple(fields["checkpoint_fractions"])
    first_config = NeuralTrainingConfig(**fields)
    checkpoint = sorted(first_config_path.parent.joinpath("checkpoints").glob("*.pt"))[0]
    policy = restore_policy(first_config, checkpoint)
    _, _, states = evaluate_policy(base.environment, policy, episodes=2, horizon=base.horizon, seed_base=55_000_000, deterministic=False)
    fixed = np.asarray(states[:64], dtype=np.float32)
    analyze_stage(stage, base, fixed, on_policy_state_count=64)
    result = {"finite": all(item["finite"] for item in summaries), "run_count": len(summaries), "calibration": calibration}
    (stage / "smoke_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def acrobot_pilot(root: Path = ROOT) -> dict:
    stage = root / "acrobot_pilot"
    base = NeuralTrainingConfig(
        environment="Acrobot-v1",
        method="gpomdp_reward_only",
        seed=201,
        hidden_sizes=(8, 8),
        learning_rate=1e-3,
        updates=50,
        batch_steps=1024,
        horizon=500,
        evaluation_episodes=3,
        center_returns=True,
        collector_mode="complete_episodes",
        parallel_environments=8,
    )
    learning_rate = select_learning_rate(stage, base, seeds=(191, 192), candidates=(3e-4, 1e-3, 3e-3))
    base = replace(base, learning_rate=learning_rate)
    calibration = calibrate_regularizers(
        base.environment,
        seeds=(193, 194),
        learning_rate=learning_rate,
        hidden_sizes=base.hidden_sizes,
        batch_steps=512,
        horizon=base.horizon,
        output_directory=stage / "calibration",
        center_returns=base.center_returns,
    )
    summaries = run_suite(
        stage,
        base,
        seeds=(201, 202, 203),
        beta=calibration["selected_beta"],
        entropy_coefficient=calibration["selected_entropy_coefficient"],
        handoff_fractions=(0.25, 0.35),
        include_npg=False,
    )
    # Choose 25% unless 35% improves mean final deterministic return by >5 points.
    means = {}
    for label in ("gpomdp_logbarrier_handoff_h25", "gpomdp_logbarrier_handoff_h35"):
        values = [float(item["final"]["deterministic_return"]) for item in summaries if item["run_label"] == label]
        means[label] = float(np.mean(values))
    fraction = 0.35 if means["gpomdp_logbarrier_handoff_h35"] > means["gpomdp_logbarrier_handoff_h25"] + 5.0 else 0.25
    result = {
        "selected_learning_rate": learning_rate,
        "selected_beta": calibration["selected_beta"],
        "selected_entropy_coefficient": calibration["selected_entropy_coefficient"],
        "selected_handoff_fraction": fraction,
        "handoff_pilot_means": means,
        "selection_rule": "prefer 25% unless 35% improves mean final deterministic return by more than 5 points",
        "pilot_seeds": [201, 202, 203],
    }
    (stage / "pilot_selection.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def acrobot_confirmatory(root: Path = ROOT) -> dict:
    pilot = acrobot_pilot(root)
    pilot_directory = root / "acrobot_pilot"
    stage = root / "acrobot_confirmatory"
    base = NeuralTrainingConfig(
        environment="Acrobot-v1",
        method="gpomdp_reward_only",
        seed=301,
        hidden_sizes=(8, 8),
        learning_rate=float(pilot["selected_learning_rate"]),
        updates=120,
        batch_steps=1024,
        horizon=500,
        evaluation_episodes=5,
        center_returns=True,
        collector_mode="complete_episodes",
        parallel_environments=8,
    )
    state_bank, bank_metadata = build_reference_state_bank(
        pilot_directory,
        root / "state_banks",
        base,
    )
    summaries = run_suite(
        stage,
        base,
        seeds=tuple(range(301, 311)),
        beta=float(pilot["selected_beta"]),
        entropy_coefficient=float(pilot["selected_entropy_coefficient"]),
        handoff_fractions=(float(pilot["selected_handoff_fraction"]),),
        include_npg=True,
        parallel_workers=4,
    )
    result = {
        "run_count": len(summaries),
        "complete_seeds": 10,
        "all_finite": all(item["finite"] for item in summaries),
        "state_bank_sha256": bank_metadata["sha256"],
        "frozen_hyperparameters": pilot,
        "git_commit": _git_commit(),
    }
    (stage / "confirmatory_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def analyze_acrobot(root: Path = ROOT) -> dict:
    pilot = json.loads((root / "acrobot_pilot" / "pilot_selection.json").read_text(encoding="utf-8"))
    base = NeuralTrainingConfig(
        environment="Acrobot-v1",
        method="gpomdp_reward_only",
        seed=301,
        hidden_sizes=(8, 8),
        learning_rate=float(pilot["selected_learning_rate"]),
        updates=120,
        batch_steps=1024,
        horizon=500,
        evaluation_episodes=5,
        center_returns=True,
        collector_mode="complete_episodes",
        parallel_environments=8,
    )
    bank_path = root / "state_banks" / "acrobot_reference_states.npz"
    with np.load(bank_path) as archive:
        bank = archive["states"]
    fisher_rows, alignment_rows, behavior_rows = analyze_stage(
        root / "acrobot_confirmatory", base, bank, on_policy_state_count=128
    )
    result = {
        "fisher_rows": len(fisher_rows),
        "alignment_rows": len(alignment_rows),
        "behavior_rows": len(behavior_rows),
        "state_bank_sha256": state_bank_hash(bank),
    }
    (root / "fisher_analysis" / "analysis_result.json").parent.mkdir(parents=True, exist_ok=True)
    (root / "fisher_analysis" / "analysis_result.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
