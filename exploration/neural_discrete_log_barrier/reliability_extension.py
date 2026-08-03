"""Frozen 60-pair Acrobot reliability extension and mechanism audit."""

from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .ablation import ABLATION_DIRECTORY_NAME, HANDOFF_FRACTION, _train_task
from .baseline import baseline_config
from .reliability import _environment_step_endpoint, _summarize_confirmation, _write_csv
from .training import (
    _flatten_valid_states,
    collect_fixed_step_trajectories,
    restore_policy,
    train_policy,
)


BASE_SEEDS = tuple(range(501, 521))
EXTENSION_SEEDS = tuple(range(521, 561))
ALL_SEEDS = BASE_SEEDS + EXTENSION_SEEDS
EXTENSION_METHODS = ("reward_only", "logbarrier_handoff_h25")
HANDOFF_UPDATE = 250
VISITATION_AUDIT_STEPS = 2048
SELECTED_BETA = 546.4135158976487


def _predeclaration() -> dict:
    return {
        "schema_version": 1,
        "base_seeds": list(BASE_SEEDS),
        "extension_seeds": list(EXTENSION_SEEDS),
        "total_paired_seeds": len(ALL_SEEDS),
        "methods": list(EXTENSION_METHODS),
        "learning_rate": 3e-3,
        "selected_beta": SELECTED_BETA,
        "updates": 1000,
        "episodes_per_update": 8,
        "training_episodes_per_seed": 8000,
        "architecture": [8, 8],
        "optimizer": "Adam",
        "gamma": 0.99,
        "center_returns": True,
        "normalize_returns": False,
        "horizon": 500,
        "evaluation_episodes_per_checkpoint": 32,
        "collector_mode": "complete_episodes_by_update",
        "handoff_fraction": HANDOFF_FRACTION,
        "primary_endpoint": "catastrophic failure rate",
        "failure_definition": "final stochastic return < -300 OR final stochastic termination rate < 0.8",
        "secondary_endpoints": ["final stochastic return", "true environment-step return AUC"],
        "mechanism_checkpoint_update": HANDOFF_UPDATE,
        "mechanism_diagnostics": [
            "deterministic return at handoff",
            "per-state action probability and entropy distributions on the frozen reference bank",
            "greedy-action agreement on the frozen reference bank",
            "best-minus-second-best action margins",
            "reference-bank states with different greedy actions",
            "frequency of the disagreement region under each policy's handoff visitation distribution",
        ],
        "visitation_frequency_definition": (
            "fraction of 2048 independently collected on-policy handoff states at which "
            "the paired reward-only and handoff policies choose different greedy actions"
        ),
        "broad_global_entropy_is_primary_mechanism_metric": False,
        "outcomes_used_for_configuration": False,
    }


def _stage(output_root: Path) -> Path:
    return output_root / ABLATION_DIRECTORY_NAME / "reliability_extension_60_total"


def _ensure_predeclaration(stage: Path) -> dict:
    declared = _predeclaration()
    path = stage / "predeclaration.json"
    stage.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != declared:
            raise RuntimeError("stored reliability-extension predeclaration is incompatible")
    else:
        path.write_text(json.dumps(declared, indent=2, sort_keys=True), encoding="utf-8")
    return declared


def run_extension_method(
    output_root: Path,
    *,
    method: str,
    parallel_workers: int,
) -> dict:
    """Run one frozen method across extension seeds, parallelized by seed."""

    if method not in EXTENSION_METHODS:
        raise ValueError(f"method must be one of {EXTENSION_METHODS}")
    stage = _stage(output_root)
    declaration = _ensure_predeclaration(stage)
    tasks = []
    for seed in EXTENSION_SEEDS:
        config = replace(
            baseline_config(seed, learning_rate=3e-3, updates=1000),
            method=(
                "gpomdp_reward_only"
                if method == "reward_only"
                else "gpomdp_logbarrier_handoff"
            ),
            beta=(
                0.0
                if method == "reward_only"
                else SELECTED_BETA
            ),
            handoff_fraction=(
                None if method == "reward_only" else HANDOFF_FRACTION
            ),
        )
        tasks.append((config, stage / "runs" / method / f"seed_{seed:03d}"))
    pending = [task for task in tasks if not (task[1] / "summary.json").exists()]
    print(
        f"{method}: {len(tasks)-len(pending)}/{len(tasks)} complete; "
        f"{len(pending)} pending; workers={parallel_workers}",
        flush=True,
    )
    if parallel_workers <= 1:
        for index, (config, path) in enumerate(pending, 1):
            train_policy(config, path)
            print(f"completed {index}/{len(pending)}: {method}, seed={config.seed}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(_train_task, config, str(path)): config
                for config, path in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                config = futures[future]
                future.result()
                print(f"completed {index}/{len(pending)}: {method}, seed={config.seed}", flush=True)
    result = {
        "schema_version": 1,
        "complete": True,
        "method": method,
        "seeds": list(EXTENSION_SEEDS),
        "run_count": len(tasks),
        "parallel_workers": parallel_workers,
        "predeclaration": declaration,
    }
    (stage / f"{method}_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _read_behavior_at(run: Path, update: int) -> dict[str, str]:
    with (run / "checkpoint_behavior.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    matches = [row for row in rows if int(row["update"]) == update]
    if len(matches) != 1:
        raise RuntimeError(f"expected one behavior row at update {update}: {run}")
    return matches[0]


def _archive_run_roots(output_root: Path, seed: int) -> tuple[Path, Path]:
    if seed in BASE_SEEDS:
        root = output_root / ABLATION_DIRECTORY_NAME / "reliability_confirmation_20_seeds" / "runs"
    else:
        root = _stage(output_root) / "runs"
    return (
        root / "reward_only" / f"seed_{seed:03d}",
        root / "logbarrier_handoff_h25" / f"seed_{seed:03d}",
    )


def _probability_metrics(probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    entropy = -(probabilities * np.log(probabilities)).sum(axis=1)
    ordered = np.sort(probabilities, axis=1)
    margin = ordered[:, -1] - ordered[:, -2]
    greedy = probabilities.argmax(axis=1)
    return entropy, margin, greedy


def _paired_mechanism_audit(output_root: Path, stage: Path) -> list[dict]:
    bank_path = output_root / "state_banks" / "acrobot_reference_states.npz"
    with np.load(bank_path) as archive:
        reference_states = np.asarray(archive["states"], dtype=np.float32)
    reference_tensor = torch.as_tensor(reference_states)
    arrays_directory = stage / "mechanism_arrays"
    arrays_directory.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for seed in ALL_SEEDS:
        reward_run, handoff_run = _archive_run_roots(output_root, seed)
        reward_config = baseline_config(seed, learning_rate=3e-3, updates=1000)
        handoff_config = replace(
            reward_config,
            method="gpomdp_logbarrier_handoff",
            beta=SELECTED_BETA,
            handoff_fraction=HANDOFF_FRACTION,
        )
        reward_policy = restore_policy(
            reward_config,
            reward_run / "checkpoints" / f"checkpoint_update_{HANDOFF_UPDATE:06d}.pt",
        )
        handoff_policy = restore_policy(
            handoff_config,
            handoff_run / "checkpoints" / f"checkpoint_update_{HANDOFF_UPDATE:06d}.pt",
        )
        with torch.no_grad():
            reward_probabilities = reward_policy.distribution(reference_tensor).probs.numpy()
            handoff_probabilities = handoff_policy.distribution(reference_tensor).probs.numpy()
        reward_entropy, reward_margin, reward_greedy = _probability_metrics(reward_probabilities)
        handoff_entropy, handoff_margin, handoff_greedy = _probability_metrics(handoff_probabilities)
        disagreement = reward_greedy != handoff_greedy

        visitation_frequencies = {}
        for index, (name, policy) in enumerate((
            ("reward_only", reward_policy),
            ("handoff", handoff_policy),
        )):
            generator = torch.Generator(device="cpu").manual_seed(81_000_000 + seed * 10 + index)
            trajectories = collect_fixed_step_trajectories(
                "Acrobot-v1",
                policy,
                step_count=VISITATION_AUDIT_STEPS,
                horizon=500,
                reset_seed_base=82_000_000 + seed * 10_000 + index * 1_000,
                action_generator=generator,
            )
            states = _flatten_valid_states(trajectories)
            with torch.no_grad():
                reward_actions = reward_policy(states).argmax(dim=1)
                handoff_actions = handoff_policy(states).argmax(dim=1)
            visitation_frequencies[name] = float((reward_actions != handoff_actions).float().mean())

        reward_behavior = _read_behavior_at(reward_run, HANDOFF_UPDATE)
        handoff_behavior = _read_behavior_at(handoff_run, HANDOFF_UPDATE)
        np.savez_compressed(
            arrays_directory / f"seed_{seed:03d}.npz",
            reference_states=reference_states,
            reward_probabilities=reward_probabilities,
            handoff_probabilities=handoff_probabilities,
            reward_entropy=reward_entropy,
            handoff_entropy=handoff_entropy,
            reward_margin=reward_margin,
            handoff_margin=handoff_margin,
            reward_greedy=reward_greedy,
            handoff_greedy=handoff_greedy,
            disagreement=disagreement,
        )
        rows.append({
            "seed": seed,
            "reward_deterministic_return_at_handoff": float(reward_behavior["deterministic_return"]),
            "handoff_deterministic_return_at_handoff": float(handoff_behavior["deterministic_return"]),
            "reference_greedy_action_agreement": float((~disagreement).mean()),
            "reference_disagreement_state_count": int(disagreement.sum()),
            "reference_state_count": int(len(disagreement)),
            "reward_reference_entropy_mean": float(reward_entropy.mean()),
            "handoff_reference_entropy_mean": float(handoff_entropy.mean()),
            "reward_reference_entropy_q10": float(np.quantile(reward_entropy, 0.1)),
            "reward_reference_entropy_q50": float(np.quantile(reward_entropy, 0.5)),
            "reward_reference_entropy_q90": float(np.quantile(reward_entropy, 0.9)),
            "handoff_reference_entropy_q10": float(np.quantile(handoff_entropy, 0.1)),
            "handoff_reference_entropy_q50": float(np.quantile(handoff_entropy, 0.5)),
            "handoff_reference_entropy_q90": float(np.quantile(handoff_entropy, 0.9)),
            "reward_reference_margin_mean": float(reward_margin.mean()),
            "handoff_reference_margin_mean": float(handoff_margin.mean()),
            "reward_visitation_frequency_of_disagreement_region": visitation_frequencies["reward_only"],
            "handoff_visitation_frequency_of_disagreement_region": visitation_frequencies["handoff"],
        })
        print(f"mechanism audit {seed}: {seed-ALL_SEEDS[0]+1}/{len(ALL_SEEDS)}", flush=True)
    _write_csv(stage / "mechanism_seed_summary.csv", rows)
    return rows


def _exact_mcnemar_p(discordant_a: int, discordant_b: int) -> float:
    count = discordant_a + discordant_b
    if count == 0:
        return 1.0
    tail = sum(math.comb(count, k) for k in range(min(discordant_a, discordant_b) + 1)) / (2**count)
    return min(1.0, 2.0 * tail)


def summarize_extension(output_root: Path) -> dict:
    stage = _stage(output_root)
    declaration = _ensure_predeclaration(stage)
    endpoints: list[dict] = []
    for seed in ALL_SEEDS:
        reward_run, handoff_run = _archive_run_roots(output_root, seed)
        for label, run in (("reward_only", reward_run), ("logbarrier_handoff_h25", handoff_run)):
            if not (run / "summary.json").exists():
                raise FileNotFoundError(f"missing completed run: {run}")
            endpoints.append(_environment_step_endpoint(run, label))
    summaries, paired = _summarize_confirmation(endpoints)
    reward = {row["seed"]: row for row in endpoints if row["run_label"] == "reward_only"}
    handoff = {row["seed"]: row for row in endpoints if row["run_label"] == "logbarrier_handoff_h25"}
    reward_failed_handoff_succeeded = sum(reward[s]["failure"] and not handoff[s]["failure"] for s in ALL_SEEDS)
    reward_succeeded_handoff_failed = sum(not reward[s]["failure"] and handoff[s]["failure"] for s in ALL_SEEDS)
    discordance = {
        "reward_failed_handoff_succeeded": reward_failed_handoff_succeeded,
        "reward_succeeded_handoff_failed": reward_succeeded_handoff_failed,
        "exact_two_sided_mcnemar_p": _exact_mcnemar_p(
            reward_failed_handoff_succeeded,
            reward_succeeded_handoff_failed,
        ),
    }
    _write_csv(stage / "combined_seed_endpoints.csv", endpoints)
    _write_csv(stage / "combined_method_summaries.csv", summaries)
    _write_csv(stage / "combined_paired_differences.csv", paired)
    mechanism = _paired_mechanism_audit(output_root, stage)
    result = {
        "schema_version": 1,
        "complete": True,
        "predeclaration": declaration,
        "paired_seed_count": len(ALL_SEEDS),
        "paired_failure_discordance": discordance,
        "mechanism_seed_rows": len(mechanism),
        "outcomes_used_to_change_configuration": False,
    }
    (stage / "combined_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
