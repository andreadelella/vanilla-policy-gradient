"""Run the sampled two-state factorial with the ORIGINAL undamped KL scaling.

Companion to :mod:`run_sampled_damped`. That module scales the natural step under
``F + lambda I``; this one leaves :mod:`natural_step` untouched, so the scaling
form is measured under the bare ``F``:

    scale = sqrt(2 * target_kl / d^T F d)

On a saturating softmax ``F -> 0`` while that form sits in a denominator, so the
step inflates as the signal dies. Running it at matched horizons documents the
pathology instead of describing it, and makes the damped/undamped pair a
controlled comparison: same seeds, same estimator, one line of arithmetic apart.

No step function is swapped -- this is the repository's default code path. The
module exists only to register short-horizon presets and to keep the output in a
directory named for the arithmetic used, next to its damped counterpart.

The full six-method factorial runs even though the contrast of interest is the
three NPG arms. ``_summaries`` builds its per-cell table over every entry of
``METHODS`` and its paired-comparison table references the Euclidean arms by
name, so an NPG-only grid would divide by zero on the empty PG groups. Keeping
them is also a free consistency check: the PG arms never call the step function,
so their endpoints must match the damped run seed for seed. ``--verify-pg``
checks exactly that.

Usage::

    python -m exploration.npg_logbarrier_factorial.run_sampled_undamped \
        --updates 2000 --workers 10
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .run_experiment import DEFAULT_ROOT, _require_validation
from .sampled_two_state import INITIALIZATIONS, METHODS
from . import sampled_two_state_parallel as parallel_driver

# Columns that must agree exactly between the damped and undamped runs for any
# Euclidean cell. Excludes nothing: a PG row is fully determined by its seed.
PG_ENDPOINT_COLUMNS = (
    "finite", "invalid_solve_count", "final_return", "final_q",
    "final_pi1_good", "near_optimal_basin", "first_delta_safe_positive_update",
    "zero_s1_batch_fraction",
)


def _register(updates: int, handoff: int, seeds: int) -> str:
    name = f"undamped_u{updates}"
    parallel_driver.PRESETS[name] = {
        "n_seeds": seeds,
        "updates": updates,
        "handoff": handoff,
        "batch_sizes": (4, 32, 128),
    }
    return name


def _verify_pg(undamped: Path, damped: Path) -> int:
    """Confirm the Euclidean arms are unaffected by the step-scaling change."""
    def load(directory: Path) -> dict[tuple, dict]:
        rows = csv.DictReader((directory / "sampled_endpoints.csv").open(newline="", encoding="utf-8"))
        return {
            (row["method"], row["initialization"], row["n_trajectories"], row["seed"]): row
            for row in rows if "_pg_" in row["method"]
        }

    if not (damped / "sampled_endpoints.csv").exists():
        print(f"no damped counterpart at {damped}; skipping PG verification")
        return 0
    left, right = load(undamped), load(damped)
    shared = sorted(set(left) & set(right))
    if not shared:
        print("no shared Euclidean cells to verify")
        return 1
    mismatches = [
        (key, column, left[key][column], right[key][column])
        for key in shared
        for column in PG_ENDPOINT_COLUMNS
        if left[key][column] != right[key][column]
    ]
    print(f"PG verification: {len(shared)} shared Euclidean runs, {len(mismatches)} mismatched fields")
    for key, column, a, b in mismatches[:5]:
        print(f"  {key} {column}: undamped={a} damped={b}")
    if mismatches:
        print("PG ARMS DIFFER -- the step change leaked into the Euclidean path")
        return 1
    print("PG arms identical: the damped/undamped contrast is confined to the NPG arms")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument(
        "--handoff",
        type=int,
        default=None,
        help="Update at which the barrier hands off to the reward. Default: half "
             "the horizon, matching run_sampled_damped.",
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed-chunk", type=int, default=25)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--verify-pg",
        action="store_true",
        help="After the run, diff the Euclidean endpoints against damped_u<updates>.",
    )
    args = parser.parse_args(argv)

    handoff = args.handoff if args.handoff is not None else args.updates // 2
    if not 0 < handoff < args.updates:
        raise SystemExit("handoff must lie strictly inside the horizon")

    _require_validation(args.root)
    output = args.output or (
        args.root / "sampled_two_state" / f"undamped_u{args.updates}"
    )
    name = _register(args.updates, handoff, args.seeds)

    print(
        f"UNDAMPED natural step (original natural_step.py); "
        f"{len(INITIALIZATIONS)}x3x{len(METHODS)} cells, {args.seeds} seeds, "
        f"{args.updates} updates, handoff at {handoff}",
        flush=True,
    )
    result = parallel_driver.run_sampled_factorial_parallel(
        output,
        preset=name,
        workers=args.workers,
        seed_chunk=args.seed_chunk,
    )
    result["natural_step"] = "undamped_quadratic_form"
    result["handoff_update"] = handoff
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))

    if args.verify_pg:
        damped = args.root / "sampled_two_state" / f"damped_u{args.updates}"
        return _verify_pg(output, damped)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
