"""Full geometric audit of the seeds that failed in the 200-pair replication.

Twenty-nine of the two hundred pairs contain at least one catastrophic failure
(24 reward-only, 6 handoff, 1 both). This reconstructs, at every archived
checkpoint of both arms of those pairs, the geometry the optimizer was moving
through: the Fisher spectrum, where the reward gradient sits in that spectrum,
and what a natural step would have done with it.

Both arms of a failed pair are analyzed, never just the failed one. The arms
share initial weights and environment seeds, so the surviving arm is the
control that says which geometric features belong to *failing* rather than to
the seed.

What is measured
----------------
Two Fishers per checkpoint, because they answer different questions:

* ``on_policy`` -- states the policy itself visits at that checkpoint. This is
  the metric the optimizer actually experiences, and it moves as the policy
  moves, so a collapse here is the collapse that mattered.
* ``fixed_reference`` -- one frozen state bank, identical at every checkpoint of
  every run. Holding the states fixed makes spectra comparable across updates
  and across arms; the on-policy spectrum cannot distinguish "the geometry
  degenerated" from "the policy went somewhere with different geometry".

Per Fisher: rank against the maximum possible rank, trace, extreme positive
eigenvalues, condition number, and four concentration measures (``k90``,
participation ratio, entropy effective rank, log pseudodeterminant). Degeneracy
shows up in different ones depending on whether the spectrum loses directions or
merely stretches, so all four are kept.

Per Fisher, the reward gradient decomposed in that Fisher's eigenbasis: how much
of its energy the positive spectrum captures at all, and how that energy
distributes over leading versus trailing directions in the Euclidean and the
natural inner product. ``natural_energy`` is g^T F^+ g -- the squared natural
gradient length -- which is the quantity that blows up as F degenerates.

That connects to the NPG scaling pathology this repository documents elsewhere:
``undamped_kl_scale`` uses ``sqrt(2*target_kl / g^T F^+ g)`` while
``damped_kl_scale`` uses ``F + lambda I``. Both are reported at every checkpoint
even though these runs are Euclidean Adam, because the whole point is whether
the geometry *would have* defeated a natural step. Energy outside the positive
span is charged to the damping term rather than dropped, so the damped quantity
is a bound and not an underestimate.

Alongside the geometry, the saturation and gradient-norm columns each run already
recorded: entropy, mean and global minimum action probability, barrier value, and
the reward/barrier/regularizer gradient norms with their cosine.

Selection and honesty
---------------------
The seeds analyzed here were chosen *because* they failed, so every number is
post-hoc and outcome-selected. ``post_hoc_outcome_selected: true`` is recorded in
the output and nothing here feeds any confirmatory statistic. The frozen state
bank is built by a rule fixed on seed index -- the first
:data:`BANK_SOURCE_SEED_COUNT` seeds of the replication, both arms -- decided
without reference to outcomes, and it therefore contains states from failed and
surviving runs alike. Its SHA-256 is recorded so a rebuild is verifiable.

Usage::

    python -m exploration.neural_discrete_log_barrier.run_experiment \
        --stage acrobot-replication-geometry --parallel-workers 10
"""

from __future__ import annotations

import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from .fisher import (
    action_enumerated_fisher_spectrum,
    analysis_reward_gradient,
    reward_gradient_alignment,
    save_state_bank,
    state_bank_hash,
)
from .reliability import _read_csv, _write_csv
from .reliability_replication import (
    METHODS,
    REPLICATION_SEEDS,
    _run_directory,
    _stage,
    paired_configs,
)
from .training import (
    _flatten_valid_states,
    collect_fixed_step_trajectories,
    restore_policy,
)

DIAGNOSTIC_DIRECTORY = "failed_seed_geometry"

# On-policy states collected per checkpoint. The Fisher is built by enumerating
# all three actions at every state, so this is 3x that many score rows against
# 155 parameters -- comfortably rank-saturating without making 1300 checkpoints
# expensive.
ON_POLICY_STATE_COUNT = 128

# The frozen bank: two states from each (seed, arm) source over the first N
# seeds. Fixed by seed index before any outcome was consulted.
BANK_SOURCE_SEED_COUNT = 64
BANK_STATES_PER_SOURCE = 2
BANK_SOURCE_UPDATE = 1000

# Damping for the reported natural step, matching NeuralTrainingConfig.npg_damping.
NPG_DAMPING = 1e-2
TARGET_KL = 1e-3

FISHER_METRIC_KEYS = (
    "numerical_rank",
    "maximum_possible_rank",
    "trace",
    "largest_eigenvalue",
    "smallest_positive_eigenvalue",
    "positive_spectrum_condition_number",
    "k90",
    "participation_ratio",
    "entropy_effective_rank",
    "log_pseudodeterminant",
)
ALIGNMENT_METRIC_KEYS = (
    "reward_gradient_norm",
    "captured_euclidean_energy_fraction",
    "leading_k90_euclidean_energy_fraction",
    "leading_k90_natural_energy_fraction",
    "natural_energy",
)


def failed_seeds(stage: Path) -> dict:
    """Read the replication pair table and classify which seeds failed how."""

    path = stage / "replication_pairs.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run the replication summary stage first"
        )
    rows = _read_csv(path)
    reward = [int(row["seed"]) for row in rows if row["reward_only_failure"] == "True"]
    handoff = [
        int(row["seed"]) for row in rows
        if row["logbarrier_handoff_h25_failure"] == "True"
    ]
    return {
        "reward_only_failures": reward,
        "handoff_failures": handoff,
        "both_arms_failed": sorted(set(reward) & set(handoff)),
        "any_arm_failed": sorted(set(reward) | set(handoff)),
    }


def _checkpoint_states(
    policy: torch.nn.Module,
    environment: str,
    horizon: int,
    *,
    seed: int,
    update: int,
    state_count: int,
) -> tuple[torch.Tensor, list]:
    """On-policy states at one checkpoint, from a seed-and-update keyed stream."""

    generator = torch.Generator(device="cpu").manual_seed(
        91_000_000 + seed * 10_000 + update
    )
    trajectories = collect_fixed_step_trajectories(
        environment,
        policy,
        step_count=state_count,
        horizon=horizon,
        reset_seed_base=92_000_000 + seed * 10_000 + update,
        action_generator=generator,
    )
    return _flatten_valid_states(trajectories), trajectories


def build_bank(stage: Path, destination: Path) -> tuple[np.ndarray, dict]:
    """Frozen reference states pooled over the first N seeds of both arms.

    The source rule is fixed on seed index, so the bank spans failed and
    surviving runs without having been selected on either.
    """

    bank_path = destination / "acrobot_replication_reference_states.npz"
    if bank_path.exists():
        with np.load(bank_path) as archive:
            states = np.asarray(archive["states"], dtype=np.float32)
        metadata = json.loads(bank_path.with_suffix(".json").read_text(encoding="utf-8"))
        if state_bank_hash(states) != metadata["sha256"]:
            raise RuntimeError("reference state bank hash mismatch")
        return states, metadata

    source_seeds = REPLICATION_SEEDS[:BANK_SOURCE_SEED_COUNT]
    collected: list[np.ndarray] = []
    for seed in source_seeds:
        configs = paired_configs(seed)
        for method in METHODS:
            config = configs[method]
            checkpoint = (
                _run_directory(stage, method, seed)
                / "checkpoints"
                / f"checkpoint_update_{BANK_SOURCE_UPDATE:06d}.pt"
            )
            policy = restore_policy(config, checkpoint)
            states, _ = _checkpoint_states(
                policy,
                config.environment,
                config.horizon,
                seed=seed,
                update=BANK_SOURCE_UPDATE,
                state_count=16,
            )
            # Evenly spaced picks rather than the first few, so the sample is
            # not all near-reset states.
            indices = np.linspace(0, len(states) - 1, BANK_STATES_PER_SOURCE).astype(int)
            collected.append(states.numpy()[indices])
    bank = np.concatenate(collected, axis=0).astype(np.float32)
    metadata = save_state_bank(
        bank_path,
        bank,
        {
            "purpose": "fixed-reference Fisher for the replication failed-seed geometry audit",
            "source_rule": (
                f"the first {BANK_SOURCE_SEED_COUNT} replication seeds "
                f"({source_seeds[0]}..{source_seeds[-1]}), both arms, at update "
                f"{BANK_SOURCE_UPDATE}; {BANK_STATES_PER_SOURCE} evenly spaced "
                "states from a short on-policy rollout of each"
            ),
            "source_rule_fixed_on_seed_index_not_outcome": True,
            "contains_states_from_failed_and_surviving_runs": True,
            "source_seeds": [int(seed) for seed in source_seeds],
            "source_methods": list(METHODS),
        },
    )
    return bank, metadata


def _natural_step_diagnostics(
    alignment,
    arrays: dict[str, np.ndarray],
    gradient_norm: float,
) -> dict:
    """What a natural step would have done with this gradient and this Fisher.

    ``natural_energy`` is g^T F^+ g over the positive spectrum only, so any
    gradient energy in the numerically null space is invisible to it. The damped
    quantity charges that residual to ``lambda`` instead of discarding it, which
    keeps it an upper bound on the damped step length.
    """

    eigenvalues = arrays["eigenvalues"]
    coordinates = arrays["gradient_coordinates"]
    captured = float((coordinates**2).sum())
    residual = max(0.0, gradient_norm**2 - captured)

    undamped_quadratic = float(alignment.natural_energy)
    damped_quadratic = float(
        (coordinates**2 / (eigenvalues + NPG_DAMPING)).sum() + residual / NPG_DAMPING
    )

    def scale(quadratic: float) -> float:
        if not math.isfinite(quadratic) or quadratic <= 0.0:
            return float("nan")
        return math.sqrt(2.0 * TARGET_KL / quadratic)

    undamped_scale = scale(undamped_quadratic)
    damped_scale = scale(damped_quadratic)
    # |step| = scale * |direction|; |F^+ g| and |(F+lambda I)^-1 g| follow from
    # the same coordinates, so no second solve is needed.
    undamped_direction = float(np.sqrt(((coordinates / eigenvalues) ** 2).sum())) if eigenvalues.size else float("nan")
    damped_direction = float(
        np.sqrt(((coordinates / (eigenvalues + NPG_DAMPING)) ** 2).sum() + residual / NPG_DAMPING**2)
    )
    return {
        "gradient_energy_outside_positive_span": residual,
        "undamped_natural_quadratic_form": undamped_quadratic,
        "damped_natural_quadratic_form": damped_quadratic,
        "undamped_kl_scale": undamped_scale,
        "damped_kl_scale": damped_scale,
        "undamped_natural_direction_norm": undamped_direction,
        "damped_natural_direction_norm": damped_direction,
        "undamped_natural_step_norm": undamped_scale * undamped_direction,
        "damped_natural_step_norm": damped_scale * damped_direction,
        "damped_step_norm_bound": math.sqrt(2.0 * TARGET_KL / NPG_DAMPING),
    }


def _audit_one_run(
    stage_path: str,
    method: str,
    seed: int,
    bank_array: np.ndarray,
    failure_flags: dict,
) -> list[dict]:
    """Every archived checkpoint of one run. Runs inside a worker process."""

    torch.set_num_threads(1)
    stage = Path(stage_path)
    config = paired_configs(seed)[method]
    run = _run_directory(stage, method, seed)
    behavior = {int(row["update"]): row for row in _read_csv(run / "checkpoint_behavior.csv")}
    gradients = {int(row["update"]): row for row in _read_csv(run / "checkpoint_gradients.csv")}
    steps = {0: 0}
    steps.update({
        int(row["update"]): int(row["environment_steps"])
        for row in _read_csv(run / "training.csv")
    })
    bank = torch.as_tensor(bank_array, dtype=torch.float32)

    rows: list[dict] = []
    for checkpoint in sorted((run / "checkpoints").glob("checkpoint_update_*.pt")):
        update = int(checkpoint.stem.split("_")[-1])
        if update not in behavior:
            continue
        policy = restore_policy(config, checkpoint)
        on_states, trajectories = _checkpoint_states(
            policy,
            config.environment,
            config.horizon,
            seed=seed,
            update=update,
            state_count=ON_POLICY_STATE_COUNT,
        )
        reward_gradient = analysis_reward_gradient(policy, trajectories, config.gamma)

        item = behavior[update]
        # The gradient archive is indexed by pre-update index, so the final
        # checkpoint has no row of its own.
        gradient_row = gradients[min(update, config.updates - 1)]
        row = {
            "seed": seed,
            "run_label": method,
            "arm_failed": bool(failure_flags[method]),
            "paired_arm_failed": bool(failure_flags[
                METHODS[1] if method == METHODS[0] else METHODS[0]
            ]),
            "update": update,
            "environment_steps": steps.get(update, ""),
            "beta": float(gradient_row["beta"]),
            "barrier_active": float(gradient_row["beta"]) > 0.0,
            "stochastic_return": float(item["stochastic_return"]),
            "deterministic_return": float(item["deterministic_return"]),
            "stochastic_termination_rate": float(item["stochastic_termination_rate"]),
            "deterministic_episode_length": float(item["episode_length"]),
            "entropy": float(item["entropy"]),
            "mean_min_probability": float(item["mean_min_probability"]),
            "global_min_probability": float(item["global_min_probability"]),
            "barrier_value": float(item["barrier_value"]),
            "archived_reward_gradient_norm": float(gradient_row["reward_gradient_norm"]),
            "archived_barrier_gradient_norm": float(gradient_row["barrier_gradient_norm"]),
            "archived_regularizer_gradient_norm": float(gradient_row["regularizer_gradient_norm"]),
            "archived_reward_regularizer_cosine": float(gradient_row["reward_regularizer_cosine"]),
        }

        for name, states in (("on_policy", on_states), ("fixed_reference", bank)):
            spectrum = action_enumerated_fisher_spectrum(policy, states)
            alignment, arrays = reward_gradient_alignment(spectrum, reward_gradient)
            metrics = spectrum.metrics.to_dict()
            for key in FISHER_METRIC_KEYS:
                row[f"{name}_{key}"] = metrics[key]
            row[f"{name}_rank_deficit"] = (
                metrics["maximum_possible_rank"] - metrics["numerical_rank"]
            )
            alignment_metrics = alignment.to_dict()
            for key in ALIGNMENT_METRIC_KEYS:
                row[f"{name}_{key}"] = alignment_metrics[key]
            for key, value in _natural_step_diagnostics(
                alignment, arrays, alignment_metrics["reward_gradient_norm"]
            ).items():
                row[f"{name}_{key}"] = value
        rows.append(row)
    return rows


def _plot(rows: list[dict], path: Path) -> None:
    """Failed against surviving arms on the quantities that separate them."""

    fields = (
        ("stochastic_return", "stochastic return", "linear"),
        ("entropy", "policy entropy", "linear"),
        ("global_min_probability", "global min action probability", "log"),
        ("on_policy_trace", "on-policy Fisher trace", "log"),
        ("on_policy_smallest_positive_eigenvalue", "on-policy smallest eigenvalue", "log"),
        ("on_policy_positive_spectrum_condition_number", "on-policy condition number", "log"),
        ("on_policy_rank_deficit", "on-policy rank deficit", "linear"),
        ("on_policy_natural_energy", "natural energy g'F+g", "log"),
        ("on_policy_undamped_natural_step_norm", "undamped natural step norm", "log"),
        ("fixed_reference_trace", "fixed-reference Fisher trace", "log"),
        ("fixed_reference_smallest_positive_eigenvalue", "fixed-reference smallest eigenvalue", "log"),
        ("fixed_reference_natural_energy", "fixed-reference natural energy", "log"),
    )
    figure, axes = plt.subplots(4, 3, figsize=(15, 15))
    for axis, (field, title, yscale) in zip(axes.reshape(-1), fields):
        for failed, colour, label in ((True, "crimson", "failed"), (False, "steelblue", "survived")):
            drawn = False
            for seed in sorted({row["seed"] for row in rows}):
                for method in METHODS:
                    subset = [
                        row for row in rows
                        if row["seed"] == seed
                        and row["run_label"] == method
                        and bool(row["arm_failed"]) is failed
                    ]
                    if not subset:
                        continue
                    subset.sort(key=lambda row: row["update"])
                    values = [row[field] for row in subset]
                    if yscale == "log":
                        values = [
                            value if isinstance(value, (int, float)) and value > 0 else np.nan
                            for value in values
                        ]
                    axis.plot(
                        [row["update"] for row in subset],
                        values,
                        color=colour,
                        alpha=0.35,
                        linewidth=0.9,
                        label=label if not drawn else None,
                    )
                    drawn = True
        axis.set_yscale(yscale)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("update")
        axis.axvline(250, color="grey", linestyle=":", linewidth=1)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle(
        "Geometry of failed vs surviving arms, replication seeds 1001-1200 "
        "(dotted line: barrier handoff at update 250)",
        fontsize=12,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _contrast(rows: list[dict]) -> list[dict]:
    """Median of each geometric column, failed arms against surviving arms.

    Medians because the natural-energy and condition-number columns have tails
    spanning many orders of magnitude, where a mean is not a summary of
    anything. Split at the handoff so the barrier's active phase and the
    post-release phase are not averaged together.
    """

    columns = [
        key for key in rows[0]
        if key not in {"seed", "run_label", "arm_failed", "paired_arm_failed", "update"}
        and isinstance(rows[0][key], (int, float, bool))
    ]
    windows = (
        ("pre_handoff", lambda update: update < 250),
        ("post_handoff", lambda update: update >= 250),
        ("final", lambda update: update == 1000),
    )
    summary: list[dict] = []
    for window_name, predicate in windows:
        for failed in (True, False):
            group = [
                row for row in rows
                if bool(row["arm_failed"]) is failed and predicate(row["update"])
            ]
            if not group:
                continue
            entry = {
                "window": window_name,
                "arm_failed": failed,
                "checkpoint_rows": len(group),
                "runs": len({(row["seed"], row["run_label"]) for row in group}),
            }
            for column in columns:
                values = [
                    float(row[column]) for row in group
                    if isinstance(row[column], (int, float, bool))
                    and math.isfinite(float(row[column]))
                ]
                entry[f"{column}_median"] = float(np.median(values)) if values else float("nan")
            summary.append(entry)
    return summary


def run_geometry_audit(output_root: Path, *, parallel_workers: int = 10) -> dict:
    """Audit both arms of every pair containing a failure."""

    stage = _stage(output_root)
    destination = stage / DIAGNOSTIC_DIRECTORY
    destination.mkdir(parents=True, exist_ok=True)

    failures = failed_seeds(stage)
    seeds = failures["any_arm_failed"]
    if not seeds:
        raise RuntimeError("no failed seeds in the replication; nothing to audit")
    reward_set = set(failures["reward_only_failures"])
    handoff_set = set(failures["handoff_failures"])

    bank, bank_metadata = build_bank(stage, destination)
    print(
        f"reference bank: {bank.shape[0]} states, sha256 {bank_metadata['sha256'][:16]}",
        flush=True,
    )
    print(
        f"auditing {len(seeds)} failed pairs x {len(METHODS)} arms "
        f"({len(failures['reward_only_failures'])} reward-only failures, "
        f"{len(failures['handoff_failures'])} handoff failures, "
        f"{len(failures['both_arms_failed'])} both)",
        flush=True,
    )

    jobs = [
        (
            method,
            seed,
            {
                METHODS[0]: seed in reward_set,
                METHODS[1]: seed in handoff_set,
            },
        )
        for seed in seeds
        for method in METHODS
    ]
    rows: list[dict] = []
    if parallel_workers <= 1:
        for index, (method, seed, flags) in enumerate(jobs, 1):
            rows.extend(_audit_one_run(str(stage), method, seed, bank, flags))
            print(f"audited {index}/{len(jobs)}: {method}, seed={seed}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(_audit_one_run, str(stage), method, seed, bank, flags):
                    (method, seed)
                for method, seed, flags in jobs
            }
            for index, future in enumerate(as_completed(futures), 1):
                method, seed = futures[future]
                rows.extend(future.result())
                print(f"audited {index}/{len(jobs)}: {method}, seed={seed}", flush=True)

    rows.sort(key=lambda row: (row["seed"], row["run_label"], row["update"]))
    contrast = _contrast(rows)
    _write_csv(destination / "checkpoint_geometry.csv", rows)
    _write_csv(destination / "geometry_contrast.csv", contrast)
    _plot(rows, destination / "failed_seed_geometry.png")

    result = {
        "schema_version": 1,
        "complete": True,
        "stage": f"{stage.name}/{DIAGNOSTIC_DIRECTORY}",
        "post_hoc_outcome_selected": True,
        "included_in_confirmatory_statistics": False,
        "training_rerun": False,
        "failures": failures,
        "audited_pairs": len(seeds),
        "audited_runs": len(jobs),
        "checkpoint_rows": len(rows),
        "on_policy_state_count": ON_POLICY_STATE_COUNT,
        "reference_bank": bank_metadata,
        "npg_damping_reported": NPG_DAMPING,
        "target_kl_reported": TARGET_KL,
        "natural_step_is_counterfactual": (
            "these runs used Euclidean Adam; the natural-step columns report what "
            "the measured geometry would have done to an NPG step, and were not "
            "applied during training"
        ),
    }
    (destination / "geometry_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result
