"""Paired 200-seed Acrobot reliability replication on fresh seeds 1001..1200.

The 200 pairs already on disk (501..700, split across ``conf_20`` and
``reliability_extension_60_total``) establish the failure-rate contrast. This
runs it again on seeds no prior stage has touched, so the replication is an
independent sample rather than an extension of the same one.

What "paired" means here, and why it is asserted rather than assumed
--------------------------------------------------------------------
Both arms of a seed are built from the same :func:`baseline_config`, so they get
the same architecture, learning rate, horizon, episode budget, return
convention, and -- via ``torch.manual_seed(config.seed)`` in
:func:`build_seeded_policy` -- the same initial weights and the same environment
reset seeds. The barrier is the only difference.

That already held for seeds 501..700, but only by construction: nothing checked
it. A future edit to ``baseline_config`` or to the ``replace`` call could make
the arms differ in something else and the comparison would quietly stop being
paired. So this module checks it twice:

* :func:`_assert_pairable` runs *before* any training and compares the two
  serialized configs field by field. Anything outside
  :data:`PAIRED_FREE_FIELDS` differing is a design error, and it is cheap to
  catch for all 200 seeds up front rather than after eleven core-hours.
* :func:`verify_realized_pairing` runs *after* training and compares the
  ``initial_weight_identifier`` recorded in each arm's ``summary.json``. That is
  a SHA-256 over the actual initial parameter tensors, so it confirms the two
  arms really started from the same point rather than merely being configured to.

No mechanism audit
------------------
:mod:`reliability_extension` also reconstructs per-state policy diagnostics at
the handoff against a frozen reference-state bank. That bank
(``state_banks/acrobot_reference_states.npz``) is not present in this
repository, so the audit would fail on its first read. The primary endpoint --
the paired catastrophic failure rate -- does not depend on it, and the
secondary endpoints are read from each run's own archive, so this stage stops at
the outcome analysis.

Usage::

    python -m exploration.neural_discrete_log_barrier.run_experiment \
        --stage acrobot-reliability-replication --parallel-workers 10
    python -m exploration.neural_discrete_log_barrier.run_experiment \
        --stage acrobot-reliability-replication-summary
"""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path

import numpy as np

from vpg.stats import mean_confidence_interval

from .ablation import ABLATION_DIRECTORY_NAME, HANDOFF_FRACTION, _train_task
from .baseline import baseline_config
from .reliability import (
    FAILURE_RETURN_THRESHOLD,
    FAILURE_TERMINATION_THRESHOLD,
    _environment_step_endpoint,
    _summarize_confirmation,
    _wilson_interval,
    _write_csv,
)
from .reliability_extension import SELECTED_BETA, _exact_mcnemar_p
from .training import NeuralTrainingConfig, train_policy


REPLICATION_SEEDS = tuple(range(1001, 1201))
METHODS = ("reward_only", "logbarrier_handoff_h25")
HANDOFF_UPDATE = 250
UPDATES = 1000
LEARNING_RATE = 3e-3
STAGE_NAME = "reliability_replication_seeds_1001_1200"

# The only configuration fields the two arms of a pair may differ in. Every
# other field -- architecture, seed, optimizer, learning rate, horizon, episode
# budget, collector, return convention -- must match exactly, because that is
# what makes the contrast attributable to the barrier and nothing else.
PAIRED_FREE_FIELDS = frozenset({
    "method",
    "beta",
    "handoff_fraction",
    "handoff_update",
})

SECONDARY_ENDPOINTS = (
    "final_stochastic_return",
    "final_stochastic_termination_rate",
    "environment_step_return_auc",
)


def _predeclaration() -> dict:
    return {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "seeds": list(REPLICATION_SEEDS),
        "paired_seed_count": len(REPLICATION_SEEDS),
        "seeds_disjoint_from_prior_stages": True,
        "methods": list(METHODS),
        "pairing": (
            "both arms of a seed share initial weights, environment reset seeds, "
            "and every configuration field except "
            + ", ".join(sorted(PAIRED_FREE_FIELDS))
        ),
        "paired_free_fields": sorted(PAIRED_FREE_FIELDS),
        "learning_rate": LEARNING_RATE,
        "selected_beta": SELECTED_BETA,
        "updates": UPDATES,
        "episodes_per_update": 8,
        "training_episodes_per_seed": UPDATES * 8,
        "architecture": [8, 8],
        "optimizer": "Adam",
        "gamma": 0.99,
        "center_returns": True,
        "normalize_returns": False,
        "horizon": 500,
        "evaluation_episodes_per_checkpoint": 32,
        "collector_mode": "complete_episodes_by_update",
        "handoff_fraction": HANDOFF_FRACTION,
        "handoff_update": HANDOFF_UPDATE,
        "primary_endpoint": "paired catastrophic failure rate",
        "failure_definition": (
            f"final stochastic return < {FAILURE_RETURN_THRESHOLD} OR final "
            f"stochastic termination rate < {FAILURE_TERMINATION_THRESHOLD}"
        ),
        "primary_test": "two-sided exact McNemar on discordant pairs",
        "secondary_endpoints": list(SECONDARY_ENDPOINTS),
        "mechanism_audit_included": False,
        "mechanism_audit_omitted_because": (
            "the frozen reference-state bank is absent from this repository"
        ),
        "beta_inherited_from_prior_calibration": True,
        "outcomes_used_for_configuration": False,
    }


def _stage(output_root: Path) -> Path:
    return output_root / ABLATION_DIRECTORY_NAME / STAGE_NAME


def _ensure_predeclaration(stage: Path) -> dict:
    """Write the frozen protocol once, then refuse any later disagreement."""

    declared = _predeclaration()
    path = stage / "predeclaration.json"
    stage.mkdir(parents=True, exist_ok=True)
    if path.exists():
        stored = json.loads(path.read_text(encoding="utf-8"))
        if stored != declared:
            raise RuntimeError(
                f"stored predeclaration at {path} disagrees with the current "
                "protocol; a frozen stage cannot be reconfigured in place"
            )
    else:
        path.write_text(json.dumps(declared, indent=2, sort_keys=True), encoding="utf-8")
    return declared


def paired_configs(seed: int) -> dict[str, NeuralTrainingConfig]:
    """The two arms of one seed, differing only in the barrier.

    Both start from ``baseline_config``, so the reward-only arm *is* the
    baseline protocol and the handoff arm is that same protocol with beta
    switched on until the handoff.
    """

    reward_only = baseline_config(seed, learning_rate=LEARNING_RATE, updates=UPDATES)
    handoff = replace(
        reward_only,
        method="gpomdp_logbarrier_handoff",
        beta=SELECTED_BETA,
        handoff_fraction=HANDOFF_FRACTION,
    )
    return {"reward_only": reward_only, "logbarrier_handoff_h25": handoff}


def _assert_pairable(configs: dict[str, NeuralTrainingConfig]) -> None:
    """Fail before training if the arms differ anywhere they must not."""

    left, right = (configs[method].to_dict() for method in METHODS)
    differing = {
        field
        for field in set(left) | set(right)
        if left.get(field) != right.get(field)
    }
    unexpected = differing - PAIRED_FREE_FIELDS
    if unexpected:
        raise RuntimeError(
            f"arms of seed {left['seed']} differ in {sorted(unexpected)}, which "
            "breaks the pairing; only "
            f"{sorted(PAIRED_FREE_FIELDS)} may differ"
        )
    if differing != PAIRED_FREE_FIELDS:
        raise RuntimeError(
            f"arms of seed {left['seed']} are identical in "
            f"{sorted(PAIRED_FREE_FIELDS - differing)}; the handoff arm is not "
            "actually applying the barrier"
        )


def _run_directory(stage: Path, method: str, seed: int) -> Path:
    return stage / "runs" / method / f"seed_{seed:04d}"


def run_replication(output_root: Path, *, parallel_workers: int = 10) -> dict:
    """Train both arms of all 200 pairs, resuming over any already complete.

    Both arms go into a single worker pool. Each run is fully determined by its
    own config and seed, so the scheduling order cannot affect a result, and one
    pool means the pairing verification covers everything in one pass.
    """

    stage = _stage(output_root)
    declaration = _ensure_predeclaration(stage)

    tasks: list[tuple[NeuralTrainingConfig, Path]] = []
    for seed in REPLICATION_SEEDS:
        configs = paired_configs(seed)
        _assert_pairable(configs)
        for method in METHODS:
            tasks.append((configs[method], _run_directory(stage, method, seed)))

    pending = [task for task in tasks if not (task[1] / "summary.json").exists()]
    print(
        f"{len(tasks)-len(pending)}/{len(tasks)} runs complete "
        f"({len(REPLICATION_SEEDS)} pairs x {len(METHODS)} arms); "
        f"{len(pending)} pending; workers={parallel_workers}",
        flush=True,
    )

    if parallel_workers <= 1:
        for index, (config, path) in enumerate(pending, 1):
            train_policy(config, path)
            print(
                f"completed {index}/{len(pending)}: {config.method}, seed={config.seed}",
                flush=True,
            )
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
                    f"completed {index}/{len(pending)}: {config.method}, seed={config.seed}",
                    flush=True,
                )

    pairing = verify_realized_pairing(stage)
    result = {
        "schema_version": 1,
        "complete": True,
        "stage": STAGE_NAME,
        "run_count": len(tasks),
        "parallel_workers": parallel_workers,
        "realized_pairing": pairing,
        "predeclaration": declaration,
    }
    (stage / "replication_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def verify_realized_pairing(stage: Path) -> dict:
    """Confirm each pair's two arms actually started from the same weights.

    ``initial_weight_identifier`` is a SHA-256 over the initial parameter
    tensors, recorded by ``train_policy`` at construction time. Comparing it
    across arms checks the executed runs, not just the configs they were asked
    for.
    """

    mismatched: list[int] = []
    missing: list[str] = []
    for seed in REPLICATION_SEEDS:
        identifiers = []
        for method in METHODS:
            path = _run_directory(stage, method, seed) / "summary.json"
            if not path.exists():
                missing.append(str(path))
                continue
            identifiers.append(
                json.loads(path.read_text(encoding="utf-8"))["initial_weight_identifier"]
            )
        if len(identifiers) == len(METHODS) and len(set(identifiers)) != 1:
            mismatched.append(seed)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} runs are missing a summary; first: {missing[0]}"
        )
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} pairs started from different initial weights; "
            f"first: seed {mismatched[0]}. The comparison is not paired."
        )
    return {
        "pairs_checked": len(REPLICATION_SEEDS),
        "initial_weights_identical_within_every_pair": True,
        "mismatched_pairs": [],
    }


def summarize_replication(output_root: Path) -> dict:
    """Failure-rate contrast, exact McNemar, and paired secondary endpoints."""

    stage = _stage(output_root)
    declaration = _ensure_predeclaration(stage)
    pairing = verify_realized_pairing(stage)

    endpoints: list[dict] = []
    for seed in REPLICATION_SEEDS:
        for method in METHODS:
            endpoints.append(
                _environment_step_endpoint(_run_directory(stage, method, seed), method)
            )
    summaries, paired = _summarize_confirmation(endpoints)

    by_method = {
        method: {row["seed"]: row for row in endpoints if row["run_label"] == method}
        for method in METHODS
    }
    reward, handoff = (by_method[method] for method in METHODS)

    pairs: list[dict] = []
    for seed in REPLICATION_SEEDS:
        row = {"seed": seed}
        for method in METHODS:
            endpoint = by_method[method][seed]
            row[f"{method}_failure"] = bool(endpoint["failure"])
            for key in SECONDARY_ENDPOINTS:
                row[f"{method}_{key}"] = float(endpoint[key])
        row["discordant"] = (
            row[f"{METHODS[0]}_failure"] != row[f"{METHODS[1]}_failure"]
        )
        pairs.append(row)

    # a is the direction the preregistration predicted: the barrier arm survives
    # a seed on which the reward-only arm collapses.
    a = sum(1 for seed in REPLICATION_SEEDS if reward[seed]["failure"] and not handoff[seed]["failure"])
    b = sum(1 for seed in REPLICATION_SEEDS if handoff[seed]["failure"] and not reward[seed]["failure"])
    both = sum(1 for seed in REPLICATION_SEEDS if reward[seed]["failure"] and handoff[seed]["failure"])

    failure_rates = {}
    for method in METHODS:
        failures = sum(1 for seed in REPLICATION_SEEDS if by_method[method][seed]["failure"])
        low, high = _wilson_interval(failures, len(REPLICATION_SEEDS))
        failure_rates[method] = {
            "failures": failures,
            "rate": failures / len(REPLICATION_SEEDS),
            "wilson95_low": low,
            "wilson95_high": high,
        }

    paired_secondary = {}
    for key in SECONDARY_ENDPOINTS:
        differences = np.asarray([
            handoff[seed][key] - reward[seed][key] for seed in REPLICATION_SEEDS
        ])
        mean, low, high = mean_confidence_interval(differences)
        paired_secondary[key] = {
            "mean_handoff_minus_reward_only": float(mean),
            "ci95_low": float(low),
            "ci95_high": float(high),
            "excludes_zero": bool(low > 0.0 or high < 0.0),
        }

    _write_csv(stage / "replication_seed_endpoints.csv", endpoints)
    _write_csv(stage / "replication_method_summaries.csv", summaries)
    _write_csv(stage / "replication_paired_differences.csv", paired)
    # One row per seed with both arms side by side, so the discordant pairs
    # behind the McNemar count are inspectable rather than just tallied.
    _write_csv(stage / "replication_pairs.csv", pairs)

    result = {
        "schema_version": 1,
        "complete": True,
        "stage": STAGE_NAME,
        "predeclaration": declaration,
        "realized_pairing": pairing,
        "paired_seed_count": len(REPLICATION_SEEDS),
        "failure_rates": failure_rates,
        "discordance": {
            "reward_fails_handoff_survives": a,
            "handoff_fails_reward_survives": b,
            "both_fail": both,
            "neither_fails": len(REPLICATION_SEEDS) - a - b - both,
            "exact_two_sided_mcnemar_p": _exact_mcnemar_p(a, b),
        },
        "paired_secondary": paired_secondary,
        "outcomes_used_to_change_configuration": False,
    }
    (stage / "replication_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"{len(REPLICATION_SEEDS)} paired seeds {REPLICATION_SEEDS[0]}..{REPLICATION_SEEDS[-1]}")
    for method in METHODS:
        rates = failure_rates[method]
        print(
            f"  {method:24} {rates['failures']:3}/{len(REPLICATION_SEEDS)} "
            f"= {rates['rate']*100:5.2f}%  Wilson95 "
            f"[{rates['wilson95_low']*100:.2f}%, {rates['wilson95_high']*100:.2f}%]"
        )
    print(
        f"  discordant a={a} b={b} both={both} "
        f"neither={len(REPLICATION_SEEDS)-a-b-both}  "
        f"exact McNemar p={result['discordance']['exact_two_sided_mcnemar_p']:.6g}"
    )
    for key, entry in paired_secondary.items():
        verdict = "SIG" if entry["excludes_zero"] else "ns"
        print(
            f"  paired {key:36} {entry['mean_handoff_minus_reward_only']:+10.4f} "
            f"[{entry['ci95_low']:+.4f}, {entry['ci95_high']:+.4f}]  {verdict}"
        )
    return result
