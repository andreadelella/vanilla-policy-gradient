"""Entropy handoff on the sampled two-state trap, against the barrier handoff.

On Acrobot the entropy handoff matched the log-barrier handoff, and on the 29
outcome-selected failed seeds it slightly exceeded it (rescuing 24/24 where the
barrier rescued 23/24). The logit-space reason was that both restoring forces are
bounded there and agree within a factor of 1.6 over the minimum-probability band
of 0.10-0.26 that Acrobot failures actually occupy.

This MDP is the venue where that agreement should break, which is why it is worth
running. The adverse initialization starts at

    pi0 = (0.8668, 0.0159, 0.1173)     pi1 = (0.0159, 0.8668, 0.1173)

so the good action in state 1 begins at probability **0.0159** -- an order of
magnitude deeper than anything Acrobot reached, and on the far side of the entropy
gradient's non-monotone peak near ``p = 0.119``. The barrier's per-state force
``1 - 3p`` is monotone and keeps pushing as ``p -> 0``; the entropy force carries a
factor of ``p`` and therefore *vanishes* there, abandoning an action once it is
nearly dead. At the adverse point, on the coordinate that lifts the good action,
the barrier's scaled force is 0.1587 against entropy's unscaled 0.0294 -- 5.4x.

Prediction, recorded before the run: **entropy underperforms the barrier here**,
and the gap should be largest on the adverse initialization and small or absent on
uniform. A null would falsify the shape argument the Acrobot result rests on.

Calibrating the coefficient
---------------------------
"As we did in logbarrier" cannot be transplanted literally. Acrobot's beta came
from ``0.3 x the reward gradient norm at initialization``; this factorial's
``beta = 0.2`` is a **hardcoded default that was never gradient-calibrated** (for
reference, the Acrobot rule would imply ``beta = 0.0378`` here, ~5x smaller). Two
candidate rules therefore disagree, and the choice matters:

* **Matched-force (used here, primary).** Set ``c`` so the entropy force has the
  same norm as ``beta * grad_barrier`` at the adverse initialization:
  ``c = 0.4685``. This holds the *initial push* equal and varies only the shape,
  which is the hypothesis under test, and it leaves the existing barrier arms
  usable as comparators without re-running them.
* **Acrobot-rule (recorded, secondary).** ``0.3 x |grad_reward|`` at the adverse
  init gives ``c = 0.0885``. Reported in the calibration artifact for
  transparency, but it would confound shape with a 5x scale change against the
  barrier arms already on disk.

Both are computed at the **adverse** point because at the uniform initialization
both gradients are exactly zero, so no ratio is defined there. The same ``c`` is
then used for both initializations, so the two grids differ only in start point.

Existing results are untouched: the entropy methods live outside ``METHODS``, so
the canonical 36-cell factorial and every artifact from it is unchanged.

Usage::

    python -m exploration.npg_logbarrier_factorial.run_sampled_entropy \
        --updates 2000 --seeds 100 --workers 10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from exploration.tabular_mdp.model import (
    DTYPE,
    TwoStepTrap,
    as_phi,
    probabilities_from_reduced_logits,
)

from .run_experiment import DEFAULT_ROOT, _require_validation
from .sampled_two_state import (
    ENTROPY_METHODS,
    INITIALIZATIONS,
    SampledFactorialConfig,
    _summaries,
    run_one,
)
from . import sampled_two_state_parallel as parallel_driver


# The barrier strength the entropy coefficient is matched against. This is the
# factorial's own default, so the comparator arms already on disk used it.
COMPARATOR_BETA = SampledFactorialConfig.beta

# Both regularizer gradients vanish identically at the uniform initialization, so
# any ratio between them is 0/0 there. The adverse point is the only one of the two
# where a matched-force rule is defined, and it is also the regime under test.
CALIBRATION_INITIALIZATION = "adverse"

ACROBOT_TARGET_RATIO = 0.3


def _install_damped_step() -> None:
    """Point sampled_two_state at the damped step, in this process only.

    ``run_one`` calls ``target_kl_natural_step`` as a module global, so rebinding
    that name is enough -- but it only affects the calling process, so worker
    processes must call this from their initializer too.
    """

    from . import sampled_two_state as module
    from .natural_step_damped import damped_target_kl_natural_step

    module.target_kl_natural_step = damped_target_kl_natural_step


def _worker_initializer() -> None:
    """Worker setup for the damped entropy grid. Must be module-level to pickle."""

    import torch

    torch.set_num_threads(1)
    _install_damped_step()


def _exact_gradients(initialization: str) -> dict[str, torch.Tensor]:
    """Population barrier, entropy, and reward gradients at an initialization.

    Exact rather than sampled: a coefficient chosen from one noisy batch would not
    be reproducible, and the population quantity is available in closed form here.
    """

    phi = as_phi(INITIALIZATIONS[initialization]).clone().requires_grad_(True)
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    # Mean over the two pooled states, matching how both sampled estimators
    # aggregate before visitation weighting.
    barrier = (torch.log(pi0).mean() + torch.log(pi1).mean()) / 2
    entropy = (
        -(pi0 * torch.log(pi0)).sum() - (pi1 * torch.log(pi1)).sum()
    ) / 2
    reward = TwoStepTrap().exact_return(phi)
    barrier_gradient, = torch.autograd.grad(barrier, phi, retain_graph=True)
    entropy_gradient, = torch.autograd.grad(entropy, phi, retain_graph=True)
    reward_gradient, = torch.autograd.grad(reward, phi)
    return {
        "barrier": barrier_gradient.detach(),
        "entropy": entropy_gradient.detach(),
        "reward": reward_gradient.detach(),
    }


def calibrate_entropy_coefficient(output: Path) -> dict:
    """Match the entropy force to ``beta * grad_barrier`` at the adverse start."""

    gradients = _exact_gradients(CALIBRATION_INITIALIZATION)
    barrier_force = float((COMPARATOR_BETA * gradients["barrier"]).norm())
    entropy_norm = float(gradients["entropy"].norm())
    reward_norm = float(gradients["reward"].norm())
    if entropy_norm <= 0.0:
        raise RuntimeError("entropy gradient vanishes at the calibration point")

    matched = barrier_force / entropy_norm
    acrobot_rule = ACROBOT_TARGET_RATIO * reward_norm / entropy_norm
    result = {
        "schema_version": 1,
        "calibration_initialization": CALIBRATION_INITIALIZATION,
        "calibration_is_exact_not_sampled": True,
        "calibrated_at_adverse_because_both_gradients_vanish_at_uniform": True,
        "comparator_beta": COMPARATOR_BETA,
        "comparator_beta_was_a_hardcoded_default_not_gradient_calibrated": True,
        "barrier_force_norm": barrier_force,
        "unscaled_entropy_gradient_norm": entropy_norm,
        "reward_gradient_norm": reward_norm,
        "selected_entropy_coefficient": matched,
        "selection_rule": (
            "entropy coefficient such that |c * grad_entropy| equals "
            "|beta * grad_barrier| at the adverse initialization, holding the "
            "initial push equal so only the force shape differs"
        ),
        "alternative_acrobot_rule_coefficient": acrobot_rule,
        "alternative_acrobot_rule": (
            f"{ACROBOT_TARGET_RATIO} x the reward gradient norm, the rule that set "
            "Acrobot's beta; recorded but not used, because it would confound shape "
            "with a scale change against the barrier arms already on disk"
        ),
        "beta_implied_by_acrobot_rule_for_reference": (
            ACROBOT_TARGET_RATIO * reward_norm / float(gradients["barrier"].norm())
        ),
        "good_action_initial_probability": float(
            probabilities_from_reduced_logits(
                as_phi(INITIALIZATIONS[CALIBRATION_INITIALIZATION])
            )[1][0]
        ),
        "predicted_direction": (
            "entropy underperforms the barrier, most visibly on the adverse "
            "initialization, because the entropy force carries a factor of p and "
            "vanishes near the boundary where this MDP starts"
        ),
        "outcomes_used_for_selection": False,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "entropy_calibration.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    return result


def _register(updates: int, handoff: int, seeds: int, coefficient: float) -> str:
    """Register an entropy-only preset, overriding the driver's cell builder.

    The driver enumerates ``METHODS``; entropy lives outside it so the canonical
    factorial stays byte-identical. Rather than widen that constant, this swaps in
    a cell builder for the entropy grid only.
    """

    name = f"entropy_u{updates}"
    batch_sizes = (4, 32, 128)
    parallel_driver.PRESETS[name] = {
        "n_seeds": seeds,
        "updates": updates,
        "handoff": handoff,
        "batch_sizes": batch_sizes,
    }

    original = parallel_driver._cell_configs

    def cell_configs(preset: str):
        if preset != name:
            return original(preset)
        return [
            SampledFactorialConfig(
                method,
                initialization,
                n,
                seeds,
                updates=updates,
                handoff_update=handoff,
                record_interval=10,
                entropy_coefficient=coefficient,
            )
            for initialization in INITIALIZATIONS
            for n in batch_sizes
            for method in ENTROPY_METHODS
        ]

    parallel_driver._cell_configs = cell_configs
    return name


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--handoff", type=int, default=None)
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed-chunk", type=int, default=25)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--natural-step",
        choices=("damped", "undamped"),
        default="damped",
        help="Which scaling the NPG arm uses. Damped matches damped_u<updates>, "
             "which is the comparator worth having.",
    )
    args = parser.parse_args(argv)

    handoff = args.handoff if args.handoff is not None else args.updates // 2
    if not 0 < handoff < args.updates:
        raise SystemExit("handoff must lie strictly inside the horizon")

    _require_validation(args.root)
    output = args.output or (
        args.root / "sampled_two_state" / f"entropy_{args.natural_step}_u{args.updates}"
    )

    if args.natural_step == "damped":
        # Same swap run_sampled_damped performs, so the NPG entropy arm is
        # comparable with damped_u<updates> rather than with the undamped default.
        # Both the parent (for workers==1) and every worker need it.
        _install_damped_step()
        parallel_driver._initializer = _worker_initializer

    calibration = calibrate_entropy_coefficient(output)
    coefficient = float(calibration["selected_entropy_coefficient"])
    print(
        f"entropy handoff on the sampled two-state trap; coefficient "
        f"c={coefficient:.6f} matched to beta={COMPARATOR_BETA} at the adverse "
        f"start (good action p={calibration['good_action_initial_probability']:.4f}); "
        f"{len(INITIALIZATIONS)}x3x{len(ENTROPY_METHODS)} cells, {args.seeds} seeds, "
        f"{args.updates} updates, handoff at {handoff}, {args.natural_step} scaling",
        flush=True,
    )

    name = _register(args.updates, handoff, args.seeds, coefficient)
    result = parallel_driver.run_sampled_factorial_parallel(
        output,
        preset=name,
        workers=args.workers,
        seed_chunk=args.seed_chunk,
    )
    result["natural_step"] = (
        "damped_identity_regularized"
        if args.natural_step == "damped"
        else "undamped_quadratic_form"
    )
    result["handoff_update"] = handoff
    result["entropy_coefficient"] = coefficient
    result["entropy_calibration"] = calibration
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in result.items() if k != "entropy_calibration"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
