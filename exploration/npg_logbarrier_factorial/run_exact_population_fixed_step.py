"""CLI for the exact-population fixed-step NPG experiment.

Usage::

    python -m exploration.npg_logbarrier_factorial.run_exact_population_fixed_step

This runner is exact and deterministic. It never collects trajectories, never
samples, and never touches the Acrobot or sampled experiment outputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

from .exact_population_fixed_step import (
    DAMPING_CONTROLS,
    PRIMARY_DAMPING,
    PRIMARY_METHOD,
    FixedStepConfig,
    run_exact_population_fixed_step,
)
from .run_experiment import DEFAULT_ROOT


DEFAULT_OUTPUT = DEFAULT_ROOT / "exact_population_fixed_step"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--eta", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--handoff", type=int, default=500)
    parser.add_argument("--record-interval", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    base = FixedStepConfig(
        method=PRIMARY_METHOD,
        initialization="adverse",
        damping=PRIMARY_DAMPING,
        updates=args.updates,
        alpha=args.alpha,
        eta=args.eta,
        beta=args.beta,
        handoff_update=args.handoff,
        record_interval=args.record_interval,
    )
    base.validate()
    for damping in DAMPING_CONTROLS:
        replace(base, damping=damping).validate()

    manifest = run_exact_population_fixed_step(args.output, base=base)

    print(json.dumps(manifest["primary_endpoints"], indent=2, sort_keys=True))
    print(
        "worst direction relative error: "
        f"{manifest['worst_direction_relative_error']:.3e}"
    )
    print(
        "worst log-odds absolute error:  "
        f"{manifest['worst_log_odds_absolute_error']:.3e}"
    )
    print(f"artifacts: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
