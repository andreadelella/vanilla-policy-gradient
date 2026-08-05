"""Third arm on seeds 1001..1200: an entropy handoff at the barrier's coefficient.

The replication established, on these same 200 paired seeds, that a log-barrier
handoff cuts the catastrophic failure rate from 12.0% to 3.0% (exact McNemar
p = 9.1e-4). The geometry audit then established *why* not: failing runs are not
saturated. They sit at a minimum action probability of 0.10-0.26 for their entire
life and never leave the -500 return floor, so with ``center_returns=True`` every
advantage is ~0 and nothing is learned. The barrier's benefit is that it keeps the
policy exploratory long enough to find the goal.

An entropy bonus targets exactly that. So this arm asks whether the *specific
functional form* matters, or whether any comparably-scaled push away from
determinism would do -- with the coefficient set by the same rule that set beta.

Why this expects equivalence rather than superiority
-----------------------------------------------------
In logit space both regularizers are bounded, because
``d/dz_a [(1/K) sum_b log pi_b] = 1/K - pi_a``. The barrier does not have a
divergent gradient there; what it has is a *monotone* one that saturates at
``beta/K = 182.1`` and never yields. The entropy gradient is **non-monotone**: it
peaks at 80.8 near ``pi = 0.119`` and then decays to zero, abandoning an action
once it is already dying. Restoring force on a dying action, at the two calibrated
coefficients:

===========  ===========  ==========  =====
min prob     beta*dB/dz   c*dH/dz     ratio
===========  ===========  ==========  =====
0.30         18.2         19.1        0.96
0.20         72.9         65.3        1.12
0.105        124.8        80.2        1.56
0.05         154.8        63.0        2.46
0.01         176.7        22.7        7.77
===========  ===========  ==========  =====

The barrier is decisively stronger only below ``pi ~ 0.05``, which is a regime
Acrobot failures never enter. In the band the failures actually occupy the two
agree to within a factor of 1.6, by construction, because that is what matching
the gradient norm at initialization bought. Hence the preregistered expectation:
a strong benefit over reward-only, and no detectable difference from the barrier.

Power, and what a null here can and cannot mean
-----------------------------------------------
Against reward-only the test is well powered (0.92 at a barrier-sized effect,
0.70 even if entropy only reaches 8%). Against the barrier it is not: at n=200,
distinguishing entropy's 3% from 4% has power 0.02, from 6% power 0.28, and only
reaches 0.70 at 8%. So the barrier contrast is declared as a **discrimination
test against the strong null "entropy does not help at all"** -- it can rule out
entropy being much worse, and it cannot certify equivalence. That is stated here,
before the run, so a null is not later read as a positive finding.

Reusing seeds 1001..1200
------------------------
``build_seeded_policy`` seeds on ``config.seed``, so this arm starts from
bit-identical initial weights and environment reset seeds as the two arms already
on disk. That yields a three-way paired comparison at n=200 for 200 runs instead
of 600.

The disclosure this requires: the two comparator arms' outcomes are already
observed. This is not outcome-based selection -- the seed set is all 200, not a
subset chosen after looking -- and this arm's own endpoints are unobserved, so its
tests are valid. The predeclaration records it as
``comparator_arm_outcomes_known_before_this_stage`` regardless.

Usage::

    python -m exploration.neural_discrete_log_barrier.run_experiment \
        --stage acrobot-entropy-handoff --parallel-workers 10
    python -m exploration.neural_discrete_log_barrier.run_experiment \
        --stage acrobot-entropy-handoff-summary
"""

from __future__ import annotations

import csv
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
    _wilson_interval,
    _write_csv,
)
from .reliability_extension import SELECTED_BETA, _exact_mcnemar_p
from .reliability_replication import (
    REPLICATION_SEEDS,
    STAGE_NAME as REPLICATION_STAGE_NAME,
    UPDATES,
    LEARNING_RATE,
    HANDOFF_UPDATE,
    paired_configs,
)
from .training import NeuralTrainingConfig, train_policy


ARM = "entropy_handoff_h25"
STAGE_NAME = "entropy_handoff_seeds_1001_1200"

# Frozen, not recomputed. ``ablation.calibrate_episode_regularizers`` selects this
# by the same rule as beta -- 0.3 * the reward gradient norm at initialization,
# median over seeds 221..225 x 5 updates, outcomes never consulted. Recomputing it
# reproduces this value to 8e-9 relative, and beta to 1e-7, so pinning the literal
# costs nothing and guarantees the coefficient cannot drift between stages. This
# is the value the ``conf_20`` entropy_fixed runs actually used.
SELECTED_ENTROPY_COEFFICIENT = 588.8195483317652

# The two arms already on disk, which this one is compared against.
COMPARATOR_ARMS = ("reward_only", "logbarrier_handoff_h25")

SECONDARY_ENDPOINTS = (
    "final_stochastic_return",
    "final_stochastic_termination_rate",
    "environment_step_return_auc",
)

# Runs whose evaluation return never clears this are still at the -500 floor and
# have therefore received no learning signal. 29 of 30 failed runs never exceeded
# it at any checkpoint; 0 of 28 survivors stayed below it.
RETURN_FLOOR_THRESHOLD = -450.0
MECHANISM_UPDATE = HANDOFF_UPDATE

# Only these may differ between this arm and reward-only. Identical in spirit to
# the replication's guard, but the free set is the entropy fields rather than the
# barrier ones -- so a copy-paste that left beta set would be caught here.
PAIRED_FREE_FIELDS = frozenset({
    "method",
    "entropy_coefficient",
    "handoff_fraction",
    "handoff_update",
})


def _predeclaration() -> dict:
    return {
        "schema_version": 1,
        "stage": STAGE_NAME,
        "arm": ARM,
        "seeds": list(REPLICATION_SEEDS),
        "paired_seed_count": len(REPLICATION_SEEDS),
        "seeds_shared_with_stage": REPLICATION_STAGE_NAME,
        "comparator_arms": list(COMPARATOR_ARMS),
        "comparator_arm_outcomes_known_before_this_stage": True,
        "comparator_arm_outcomes_did_not_select_the_seed_set": True,
        "this_arm_outcomes_known_before_this_stage": False,
        "method": "gpomdp_entropy_handoff",
        "selected_entropy_coefficient": SELECTED_ENTROPY_COEFFICIENT,
        "entropy_coefficient_selection_rule": (
            "0.3 / median unscaled entropy-to-reward gradient norm ratio over "
            "calibration seeds 221..225 x 5 updates; the same rule and the same "
            "target ratio that selected beta"
        ),
        "entropy_coefficient_inherited_from_prior_calibration": True,
        "comparator_selected_beta": SELECTED_BETA,
        "learning_rate": LEARNING_RATE,
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
        "handoff_schedule_identical_to_barrier": True,
        "primary_endpoint": "paired catastrophic failure rate versus reward_only",
        "failure_definition": (
            f"final stochastic return < {FAILURE_RETURN_THRESHOLD} OR final "
            f"stochastic termination rate < {FAILURE_TERMINATION_THRESHOLD}"
        ),
        "primary_test": "two-sided exact McNemar on discordant pairs",
        "secondary_test": (
            "two-sided exact McNemar versus logbarrier_handoff_h25, declared "
            "underpowered for small differences"
        ),
        "secondary_test_power_at_n200": {
            "entropy_true_rate_0.04": 0.02,
            "entropy_true_rate_0.05": 0.12,
            "entropy_true_rate_0.06": 0.28,
            "entropy_true_rate_0.08": 0.70,
            "entropy_true_rate_0.12": 0.99,
            "note": (
                "simulated against the observed barrier rate of 3.0%; a null "
                "rules out entropy being much worse and cannot certify equivalence"
            ),
        },
        "secondary_endpoints": list(SECONDARY_ENDPOINTS),
        "mechanism_endpoint": (
            f"fraction of runs whose best evaluation return through update "
            f"{MECHANISM_UPDATE} is still <= {RETURN_FLOOR_THRESHOLD}"
        ),
        "predicted_direction": (
            "strong benefit over reward_only; no detectable difference from the "
            "log-barrier handoff, because in logit space both restoring forces "
            "are bounded and agree within a factor of 1.6 over the min-probability "
            "band of 0.10-0.26 that the failures actually occupy"
        ),
        "falsifier": (
            "entropy failing near the reward_only rate of 12% while its runs stay "
            "at a min probability of 0.10-0.26 would show matched force in the "
            "operative band is not sufficient, and that the barrier's benefit is "
            "not its restoring force"
        ),
        "outcomes_used_for_configuration": False,
    }


def _stage(output_root: Path) -> Path:
    return output_root / ABLATION_DIRECTORY_NAME / STAGE_NAME


def _replication_stage(output_root: Path) -> Path:
    return output_root / ABLATION_DIRECTORY_NAME / REPLICATION_STAGE_NAME


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


def entropy_config(seed: int) -> NeuralTrainingConfig:
    """The entropy-handoff arm of one seed.

    Built from the *same* ``baseline_config`` the replication's reward-only arm
    uses, so the pairing is structural rather than reconstructed.
    """

    reward_only = baseline_config(seed, learning_rate=LEARNING_RATE, updates=UPDATES)
    return replace(
        reward_only,
        method="gpomdp_entropy_handoff",
        entropy_coefficient=SELECTED_ENTROPY_COEFFICIENT,
        handoff_fraction=HANDOFF_FRACTION,
    )


def _assert_pairable(seed: int) -> None:
    """Fail before training if this arm differs from reward-only anywhere it must not."""

    reward_only = paired_configs(seed)["reward_only"].to_dict()
    entropy = entropy_config(seed).to_dict()
    differing = {
        field
        for field in set(reward_only) | set(entropy)
        if reward_only.get(field) != entropy.get(field)
    }
    unexpected = differing - PAIRED_FREE_FIELDS
    if unexpected:
        raise RuntimeError(
            f"entropy arm of seed {seed} differs from reward_only in "
            f"{sorted(unexpected)}, which breaks the pairing; only "
            f"{sorted(PAIRED_FREE_FIELDS)} may differ"
        )
    if differing != PAIRED_FREE_FIELDS:
        raise RuntimeError(
            f"entropy arm of seed {seed} is identical to reward_only in "
            f"{sorted(PAIRED_FREE_FIELDS - differing)}; the entropy bonus is not "
            "actually being applied"
        )
    if entropy["beta"] != 0.0:
        raise RuntimeError(
            f"entropy arm of seed {seed} has beta={entropy['beta']}; the barrier "
            "must be off in this arm or the contrast is confounded"
        )


def _run_directory(stage: Path, seed: int) -> Path:
    return stage / "runs" / ARM / f"seed_{seed:04d}"


def _comparator_directory(output_root: Path, arm: str, seed: int) -> Path:
    return _replication_stage(output_root) / "runs" / arm / f"seed_{seed:04d}"


def run_entropy_handoff(output_root: Path, *, parallel_workers: int = 10) -> dict:
    """Train the entropy-handoff arm on all 200 seeds, resuming over any complete."""

    stage = _stage(output_root)
    declaration = _ensure_predeclaration(stage)

    missing = [
        str(_comparator_directory(output_root, arm, seed))
        for arm in COMPARATOR_ARMS
        for seed in REPLICATION_SEEDS
        if not (_comparator_directory(output_root, arm, seed) / "summary.json").exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} comparator runs are absent, so this arm would have "
            f"nothing to be paired against; first: {missing[0]}"
        )

    tasks: list[tuple[NeuralTrainingConfig, Path]] = []
    for seed in REPLICATION_SEEDS:
        _assert_pairable(seed)
        tasks.append((entropy_config(seed), _run_directory(stage, seed)))

    pending = [task for task in tasks if not (task[1] / "summary.json").exists()]
    print(
        f"{len(tasks)-len(pending)}/{len(tasks)} entropy-handoff runs complete; "
        f"{len(pending)} pending; workers={parallel_workers}",
        flush=True,
    )

    if parallel_workers <= 1:
        for index, (config, path) in enumerate(pending, 1):
            train_policy(config, path)
            print(f"completed {index}/{len(pending)}: seed={config.seed}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = {
                executor.submit(_train_task, config, str(path)): config
                for config, path in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                config = futures[future]
                future.result()
                print(f"completed {index}/{len(pending)}: seed={config.seed}", flush=True)

    pairing = verify_realized_pairing(output_root)
    result = {
        "schema_version": 1,
        "complete": True,
        "stage": STAGE_NAME,
        "run_count": len(tasks),
        "parallel_workers": parallel_workers,
        "realized_pairing": pairing,
        "predeclaration": declaration,
    }
    (stage / "entropy_handoff_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def verify_realized_pairing(output_root: Path) -> dict:
    """Confirm all three arms of every seed started from the same weights.

    The whole reason this arm can reuse seeds 1001..1200 is that
    ``build_seeded_policy`` keys initial weights on the seed alone. That is a
    claim about executed runs, so it is checked against the SHA-256 each run
    recorded rather than assumed from the configs.
    """

    stage = _stage(output_root)
    mismatched: list[int] = []
    missing: list[str] = []
    for seed in REPLICATION_SEEDS:
        paths = [_run_directory(stage, seed) / "summary.json"] + [
            _comparator_directory(output_root, arm, seed) / "summary.json"
            for arm in COMPARATOR_ARMS
        ]
        identifiers = []
        for path in paths:
            if not path.exists():
                missing.append(str(path))
                continue
            identifiers.append(
                json.loads(path.read_text(encoding="utf-8"))["initial_weight_identifier"]
            )
        if len(identifiers) == len(paths) and len(set(identifiers)) != 1:
            mismatched.append(seed)
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} runs are missing a summary; first: {missing[0]}"
        )
    if mismatched:
        raise RuntimeError(
            f"{len(mismatched)} seeds have arms starting from different initial "
            f"weights; first: seed {mismatched[0]}. The comparison is not paired."
        )
    return {
        "seeds_checked": len(REPLICATION_SEEDS),
        "arms_checked": [ARM, *COMPARATOR_ARMS],
        "initial_weights_identical_across_all_three_arms": True,
        "mismatched_seeds": [],
    }


def _still_on_floor(run_directory: Path, through_update: int) -> bool:
    """Whether a run's best evaluation return through an update is still at the floor.

    Uses the best rather than the last checkpoint, so a single unlucky evaluation
    cannot make an escaping run look stuck.
    """

    with (run_directory / "checkpoint_behavior.csv").open(newline="", encoding="utf-8") as handle:
        best = max(
            (
                float(row["stochastic_return"])
                for row in csv.DictReader(handle)
                if int(row["update"]) <= through_update
            ),
            default=float("-inf"),
        )
    return best <= RETURN_FLOOR_THRESHOLD


def summarize_entropy_handoff(output_root: Path) -> dict:
    """Three-way paired failure rates, both McNemar tests, and the floor endpoint."""

    stage = _stage(output_root)
    declaration = _ensure_predeclaration(stage)
    pairing = verify_realized_pairing(output_root)

    arms = (ARM, *COMPARATOR_ARMS)

    def directory(arm: str, seed: int) -> Path:
        if arm == ARM:
            return _run_directory(stage, seed)
        return _comparator_directory(output_root, arm, seed)

    endpoints: list[dict] = []
    for arm in arms:
        for seed in REPLICATION_SEEDS:
            endpoints.append(_environment_step_endpoint(directory(arm, seed), arm))
    by_arm = {
        arm: {row["seed"]: row for row in endpoints if row["run_label"] == arm}
        for arm in arms
    }

    failure_rates = {}
    for arm in arms:
        failures = sum(1 for seed in REPLICATION_SEEDS if by_arm[arm][seed]["failure"])
        low, high = _wilson_interval(failures, len(REPLICATION_SEEDS))
        failure_rates[arm] = {
            "failures": failures,
            "rate": failures / len(REPLICATION_SEEDS),
            "wilson95_low": low,
            "wilson95_high": high,
            "failed_seeds": [
                seed for seed in REPLICATION_SEEDS if by_arm[arm][seed]["failure"]
            ],
        }

    def contrast(treatment: str, reference: str) -> dict:
        a = sum(
            1 for seed in REPLICATION_SEEDS
            if by_arm[reference][seed]["failure"] and not by_arm[treatment][seed]["failure"]
        )
        b = sum(
            1 for seed in REPLICATION_SEEDS
            if by_arm[treatment][seed]["failure"] and not by_arm[reference][seed]["failure"]
        )
        both = sum(
            1 for seed in REPLICATION_SEEDS
            if by_arm[reference][seed]["failure"] and by_arm[treatment][seed]["failure"]
        )
        secondary = {}
        for key in SECONDARY_ENDPOINTS:
            differences = np.asarray([
                by_arm[treatment][seed][key] - by_arm[reference][seed][key]
                for seed in REPLICATION_SEEDS
            ])
            mean, low, high = mean_confidence_interval(differences)
            secondary[key] = {
                "mean_treatment_minus_reference": float(mean),
                "ci95_low": float(low),
                "ci95_high": float(high),
                "excludes_zero": bool(low > 0.0 or high < 0.0),
            }
        return {
            "treatment": treatment,
            "reference": reference,
            "reference_fails_treatment_survives": a,
            "treatment_fails_reference_survives": b,
            "both_fail": both,
            "neither_fails": len(REPLICATION_SEEDS) - a - b - both,
            "exact_two_sided_mcnemar_p": _exact_mcnemar_p(a, b),
            "paired_secondary": secondary,
        }

    contrasts = {
        "versus_reward_only": contrast(ARM, "reward_only"),
        "versus_logbarrier_handoff": contrast(ARM, "logbarrier_handoff_h25"),
    }

    floor = {}
    for arm in arms:
        stuck = [
            seed for seed in REPLICATION_SEEDS
            if _still_on_floor(directory(arm, seed), MECHANISM_UPDATE)
        ]
        low, high = _wilson_interval(len(stuck), len(REPLICATION_SEEDS))
        floor[arm] = {
            "runs_still_on_floor": len(stuck),
            "fraction": len(stuck) / len(REPLICATION_SEEDS),
            "wilson95_low": low,
            "wilson95_high": high,
        }

    rows: list[dict] = []
    for seed in REPLICATION_SEEDS:
        row: dict = {"seed": seed}
        for arm in arms:
            row[f"{arm}_failure"] = bool(by_arm[arm][seed]["failure"])
            row[f"{arm}_on_floor_at_handoff"] = _still_on_floor(
                directory(arm, seed), MECHANISM_UPDATE
            )
            for key in SECONDARY_ENDPOINTS:
                row[f"{arm}_{key}"] = float(by_arm[arm][seed][key])
        rows.append(row)

    _write_csv(stage / "entropy_seed_endpoints.csv", endpoints)
    _write_csv(stage / "entropy_three_arm_pairs.csv", rows)

    result = {
        "schema_version": 1,
        "complete": True,
        "stage": STAGE_NAME,
        "predeclaration": declaration,
        "realized_pairing": pairing,
        "paired_seed_count": len(REPLICATION_SEEDS),
        "failure_rates": failure_rates,
        "contrasts": contrasts,
        "still_on_return_floor_at_handoff": floor,
        "outcomes_used_to_change_configuration": False,
    }
    (stage / "entropy_handoff_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"{len(REPLICATION_SEEDS)} paired seeds, three arms sharing initial weights")
    for arm in arms:
        rates = failure_rates[arm]
        print(
            f"  {arm:24} {rates['failures']:3}/{len(REPLICATION_SEEDS)} "
            f"= {rates['rate']*100:5.2f}%  Wilson95 "
            f"[{rates['wilson95_low']*100:.2f}%, {rates['wilson95_high']*100:.2f}%]"
        )
    for name, entry in contrasts.items():
        print(
            f"  {name:28} a={entry['reference_fails_treatment_survives']:3} "
            f"b={entry['treatment_fails_reference_survives']:3} "
            f"both={entry['both_fail']:3}  exact McNemar "
            f"p={entry['exact_two_sided_mcnemar_p']:.6g}"
        )
    print(f"  still on the -500 floor at update {MECHANISM_UPDATE}:")
    for arm in arms:
        entry = floor[arm]
        print(
            f"    {arm:24} {entry['runs_still_on_floor']:3}/{len(REPLICATION_SEEDS)} "
            f"= {entry['fraction']*100:5.2f}%"
        )
    return result
