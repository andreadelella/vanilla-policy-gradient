"""Entropy handoff on the 29 failed seeds only: a cheap probe before the full arm.

The full 200-seed entropy arm costs ~11 core-hours. Nearly all of its signal lives
in the 29 seeds where at least one existing arm failed, because the other 171 are
seeds every arm already survives. This runs the entropy handoff on just those 29
(~8 minutes at 10 workers) to see whether it behaves like the barrier before
committing to the full arm.

What this can and cannot conclude
---------------------------------
These seeds were **selected on outcomes**, so no failure *rate* computed here
estimates anything. 24/29 were chosen precisely because reward-only failed on
them; entropy will look terrible or wonderful on that set for reasons that are
partly selection.

What *is* valid is a **rescue rate compared against the barrier's rescue rate on
the identical seeds**, because both arms are conditioned on the same
outcome-selected set, and neither was selected using its own results. The subsets
differ in which arm did the selecting:

* :data:`SUBSET_A_SEEDS` -- the 24 where reward-only failed. Selected on a *third*
  arm, so it is neutral between entropy and the barrier. The barrier rescued 23 of
  24. This is the comparison that matters.
* :data:`SUBSET_B_SEEDS` -- the 6 where the barrier failed. Selected on the
  comparator, so it is adversarial *to the barrier* and asks the complementary
  question: does entropy rescue what the barrier could not?

Seed 1142 is in both subsets (every arm failed it), and is reported in both.

This is explicitly labelled a probe, not a preregistered test. It carries
``post_hoc_outcome_selected: true`` and declares no p-values, because an exact
McNemar on an outcome-selected subset has no null to test against. The full
:mod:`entropy_handoff` stage remains the inferential experiment.

Usage::

    python -m exploration.neural_discrete_log_barrier.run_experiment \
        --stage acrobot-entropy-probe --parallel-workers 10
"""

from __future__ import annotations

import csv
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .ablation import ABLATION_DIRECTORY_NAME, HANDOFF_FRACTION, _train_task
from .entropy_handoff import (
    ARM,
    COMPARATOR_ARMS,
    MECHANISM_UPDATE,
    RETURN_FLOOR_THRESHOLD,
    SECONDARY_ENDPOINTS,
    SELECTED_ENTROPY_COEFFICIENT,
    _assert_pairable,
    _comparator_directory,
    _still_on_floor,
    entropy_config,
)
from .reliability import _environment_step_endpoint, _write_csv
from .reliability_extension import SELECTED_BETA
from .reliability_replication import (
    HANDOFF_UPDATE,
    METHODS as REPLICATION_METHODS,
    REPLICATION_SEEDS,
    STAGE_NAME as REPLICATION_STAGE_NAME,
    UPDATES,
)
from .training import NeuralTrainingConfig, train_policy


STAGE_NAME = "entropy_probe_failed_seeds"

# Read from the replication's own pair table rather than hardcoded, so the probe
# cannot silently disagree with the stage it is derived from.
PAIRS_FILE = "replication_pairs.csv"


def _stage(output_root: Path) -> Path:
    return output_root / ABLATION_DIRECTORY_NAME / STAGE_NAME


def _replication_stage(output_root: Path) -> Path:
    return output_root / ABLATION_DIRECTORY_NAME / REPLICATION_STAGE_NAME


def failed_seed_subsets(output_root: Path) -> dict[str, list[int]]:
    """The seeds each existing arm failed, read from the replication's pair table."""

    path = _replication_stage(output_root) / PAIRS_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"the replication pair table is absent at {path}; run "
            "--stage acrobot-reliability-replication-summary first"
        )
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    reward = [int(r["seed"]) for r in rows if r["reward_only_failure"] == "True"]
    barrier = [
        int(r["seed"]) for r in rows if r["logbarrier_handoff_h25_failure"] == "True"
    ]
    return {
        "subset_a_reward_only_failed": reward,
        "subset_b_logbarrier_failed": barrier,
        "union": sorted(set(reward) | set(barrier)),
        "both_arms_failed": sorted(set(reward) & set(barrier)),
    }


def _predeclaration(subsets: dict[str, list[int]]) -> dict:
    return {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "purpose": (
            "cheap probe of the entropy handoff on outcome-selected failed seeds, "
            "run before committing to the full 200-seed arm"
        ),
        "post_hoc_outcome_selected": True,
        "inferential_status": (
            "exploratory; no failure rate computed here estimates a population "
            "quantity, and no p-value is reported, because the seeds were chosen "
            "using the comparator arms' outcomes"
        ),
        "valid_comparison": (
            "rescue rate versus the log-barrier handoff's rescue rate on the "
            "identical seeds, since both arms are conditioned on the same "
            "outcome-selected set and neither was selected on its own results"
        ),
        "arm": ARM,
        "method": "gpomdp_entropy_handoff",
        "selected_entropy_coefficient": SELECTED_ENTROPY_COEFFICIENT,
        "comparator_selected_beta": SELECTED_BETA,
        "entropy_coefficient_inherited_from_prior_calibration": True,
        "seed_source_stage": REPLICATION_STAGE_NAME,
        "subset_a_reward_only_failed": subsets["subset_a_reward_only_failed"],
        "subset_a_selected_on_a_third_arm_so_neutral_between_entropy_and_barrier": True,
        "subset_b_logbarrier_failed": subsets["subset_b_logbarrier_failed"],
        "subset_b_selected_on_the_comparator_so_adversarial_to_the_barrier": True,
        "both_arms_failed": subsets["both_arms_failed"],
        "seeds": subsets["union"],
        "run_count": len(subsets["union"]),
        "updates": UPDATES,
        "learning_rate": 3e-3,
        "handoff_fraction": HANDOFF_FRACTION,
        "handoff_update": HANDOFF_UPDATE,
        "initial_weights_shared_with_existing_arms": True,
        "full_arm_remains_the_inferential_experiment": "entropy_handoff_seeds_1001_1200",
    }


def _run_directory(stage: Path, seed: int) -> Path:
    return stage / "runs" / ARM / f"seed_{seed:04d}"


def run_entropy_probe(output_root: Path, *, parallel_workers: int = 10) -> dict:
    """Train the entropy handoff on the failed seeds and report rescue rates."""

    stage = _stage(output_root)
    stage.mkdir(parents=True, exist_ok=True)
    subsets = failed_seed_subsets(output_root)
    seeds = subsets["union"]
    declaration = _predeclaration(subsets)

    path = stage / "predeclaration.json"
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != declaration:
            raise RuntimeError(
                f"stored predeclaration at {path} disagrees with the current "
                "protocol; a frozen stage cannot be reconfigured in place"
            )
    else:
        path.write_text(json.dumps(declaration, indent=2, sort_keys=True), encoding="utf-8")

    unknown = [seed for seed in seeds if seed not in REPLICATION_SEEDS]
    if unknown:
        raise RuntimeError(f"seeds outside the replication set: {unknown}")

    tasks: list[tuple[NeuralTrainingConfig, Path]] = []
    for seed in seeds:
        _assert_pairable(seed)
        tasks.append((entropy_config(seed), _run_directory(stage, seed)))

    pending = [task for task in tasks if not (task[1] / "summary.json").exists()]
    print(
        f"entropy probe on {len(seeds)} outcome-selected failed seeds "
        f"({len(subsets['subset_a_reward_only_failed'])} reward-only failures, "
        f"{len(subsets['subset_b_logbarrier_failed'])} barrier failures, "
        f"{len(subsets['both_arms_failed'])} shared); "
        f"{len(tasks)-len(pending)} complete, {len(pending)} pending; "
        f"workers={parallel_workers}",
        flush=True,
    )

    if parallel_workers <= 1:
        for index, (config, directory) in enumerate(pending, 1):
            train_policy(config, directory)
            print(f"completed {index}/{len(pending)}: seed={config.seed}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(_train_task, config, str(directory)): config
                for config, directory in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                config = futures[future]
                future.result()
                print(f"completed {index}/{len(pending)}: seed={config.seed}", flush=True)

    return _summarize(output_root, stage, subsets, declaration)


def _summarize(
    output_root: Path,
    stage: Path,
    subsets: dict[str, list[int]],
    declaration: dict,
) -> dict:
    seeds = subsets["union"]
    arms = (ARM, *COMPARATOR_ARMS)

    def directory(arm: str, seed: int) -> Path:
        if arm == ARM:
            return _run_directory(stage, seed)
        return _comparator_directory(output_root, arm, seed)

    # Weight identity is the entire basis for reusing these seeds, so it is
    # verified on the probe's own runs rather than inherited from the full stage.
    mismatched = []
    for seed in seeds:
        identifiers = {
            json.loads((directory(arm, seed) / "summary.json").read_text(encoding="utf-8"))[
                "initial_weight_identifier"
            ]
            for arm in arms
        }
        if len(identifiers) != 1:
            mismatched.append(seed)
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} seeds have arms from different initial weights; "
            f"first: {mismatched[0]}"
        )

    endpoints = {
        arm: {
            seed: _environment_step_endpoint(directory(arm, seed), arm) for seed in seeds
        }
        for arm in arms
    }
    on_floor = {
        arm: {
            seed: _still_on_floor(directory(arm, seed), MECHANISM_UPDATE)
            for seed in seeds
        }
        for arm in arms
    }

    rows = []
    for seed in seeds:
        row: dict = {
            "seed": seed,
            "in_subset_a_reward_only_failed": seed in subsets["subset_a_reward_only_failed"],
            "in_subset_b_logbarrier_failed": seed in subsets["subset_b_logbarrier_failed"],
        }
        for arm in arms:
            row[f"{arm}_failure"] = bool(endpoints[arm][seed]["failure"])
            row[f"{arm}_on_floor_at_handoff"] = on_floor[arm][seed]
            for key in SECONDARY_ENDPOINTS:
                row[f"{arm}_{key}"] = float(endpoints[arm][seed][key])
        rows.append(row)
    _write_csv(stage / "probe_failed_seed_endpoints.csv", rows)

    def rescue(subset: list[int], rescuers: tuple[str, ...]) -> dict:
        out = {}
        for arm in rescuers:
            survived = [s for s in subset if not endpoints[arm][s]["failure"]]
            out[arm] = {
                "survived": len(survived),
                "of": len(subset),
                "rescue_rate": len(survived) / len(subset) if subset else float("nan"),
                "survived_seeds": survived,
            }
        return out

    subset_a = subsets["subset_a_reward_only_failed"]
    subset_b = subsets["subset_b_logbarrier_failed"]
    result = {
        "schema_version": 1,
        "complete": True,
        "stage": STAGE_NAME,
        "predeclaration": declaration,
        "subset_a": {
            "description": (
                "seeds reward-only failed; selected on a third arm, so neutral "
                "between entropy and the barrier"
            ),
            "seeds": subset_a,
            "rescue": rescue(subset_a, (ARM, "logbarrier_handoff_h25")),
        },
        "subset_b": {
            "description": (
                "seeds the log-barrier handoff failed; does entropy rescue what "
                "the barrier could not"
            ),
            "seeds": subset_b,
            "rescue": rescue(subset_b, (ARM, "reward_only")),
        },
        "escaped_return_floor_by_handoff": {
            arm: {
                "escaped": sum(1 for s in seeds if not on_floor[arm][s]),
                "of": len(seeds),
            }
            for arm in arms
        },
        "return_floor_threshold": RETURN_FLOOR_THRESHOLD,
        "mechanism_update": MECHANISM_UPDATE,
        "no_p_values_reported_because_seeds_were_outcome_selected": True,
    }
    (stage / "probe_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    print()
    print(f"SUBSET A -- {len(subset_a)} seeds reward-only failed (neutral selection)")
    for arm, entry in result["subset_a"]["rescue"].items():
        print(
            f"  {arm:24} rescued {entry['survived']:3}/{entry['of']} "
            f"= {entry['rescue_rate']*100:5.1f}%"
        )
    print(f"SUBSET B -- {len(subset_b)} seeds the barrier failed")
    for arm, entry in result["subset_b"]["rescue"].items():
        print(
            f"  {arm:24} rescued {entry['survived']:3}/{entry['of']} "
            f"= {entry['rescue_rate']*100:5.1f}%"
        )
    print(f"escaped the {RETURN_FLOOR_THRESHOLD} floor by update {MECHANISM_UPDATE}, "
          f"across all {len(seeds)} seeds:")
    for arm, entry in result["escaped_return_floor_by_handoff"].items():
        print(f"  {arm:24} {entry['escaped']:3}/{entry['of']}")
    return result
