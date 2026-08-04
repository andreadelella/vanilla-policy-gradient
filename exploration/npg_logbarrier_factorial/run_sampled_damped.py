"""Run the sampled two-state factorial with damped KL scaling and a short horizon.

Two changes from the ``full`` preset, both motivated by the observed dynamics:

* the natural step scales under ``F + lambda I`` instead of ``F``
  (:mod:`natural_step_damped`), which bounds the step as the metric degenerates;
* the horizon is short. The interesting geometry is over quickly -- the uniform
  init reaches the near-optimal basin around update 38 and its Fisher is
  numerically dead by 75, while the adverse init's outcome is settled by update
  20. A 4000-update horizon spends ~99% of the run integrating noise through a
  collapsed metric, which is what produced ``|phi| ~ 1e6``.

Nothing in ``sampled_two_state.py`` or ``natural_step.py`` is modified. The step
function is swapped inside the worker processes, so ``run_one`` and every
diagnostic column are the originals.

Usage::

    python -m exploration.npg_logbarrier_factorial.run_sampled_damped \
        --updates 100 --workers 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import sampled_two_state
from .natural_step_damped import damped_target_kl_natural_step
from .run_experiment import DEFAULT_ROOT, _require_validation
from .sampled_two_state import INITIALIZATIONS, METHODS
from . import sampled_two_state_parallel as parallel_driver


def _install_damped_step() -> None:
    """Point sampled_two_state at the damped step, in this process only.

    ``run_one`` calls ``target_kl_natural_step`` as a module global, so rebinding
    that name is enough. Worker processes each call this from their initializer.
    """
    sampled_two_state.target_kl_natural_step = damped_target_kl_natural_step


def _worker_initializer() -> None:
    import torch

    torch.set_num_threads(1)
    _install_damped_step()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument(
        "--handoff",
        type=int,
        default=None,
        help="Update at which the barrier hands off to the reward. Default: half "
             "the horizon, the same fraction the full preset uses (2000 of 4000).",
    )
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed-chunk", type=int, default=25)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    handoff = args.handoff if args.handoff is not None else args.updates // 2
    if not 0 < handoff < args.updates:
        raise SystemExit("handoff must lie strictly inside the horizon")

    _require_validation(args.root)
    output = args.output or (
        args.root / "sampled_two_state" / f"damped_u{args.updates}"
    )

    # Register the short horizon as a preset the parallel driver can plan over,
    # then swap in the damped step for the parent and every worker.
    name = f"damped_u{args.updates}"
    parallel_driver.PRESETS[name] = {
        "n_seeds": args.seeds,
        "updates": args.updates,
        "handoff": handoff,
        "batch_sizes": (4, 32, 128),
    }
    parallel_driver._initializer = _worker_initializer
    _install_damped_step()

    print(
        f"damped natural step; {len(INITIALIZATIONS)}x3x{len(METHODS)} cells, "
        f"{args.seeds} seeds, {args.updates} updates, handoff at {handoff}",
        flush=True,
    )
    result = parallel_driver.run_sampled_factorial_parallel(
        output,
        preset=name,
        workers=args.workers,
        seed_chunk=args.seed_chunk,
    )
    result["natural_step"] = "damped_quadratic_form"
    result["handoff_update"] = handoff
    (output / "manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
