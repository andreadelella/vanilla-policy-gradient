"""Command-line entry point for the exact finite-MDP experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .reporting import plot_behavior, plot_explained_trace, plot_spectra, write_csv
from .training import METHODS, ExactTrainingConfig, train
from .verify import verification_results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("results/log_barrier/exact_mdp"))
    parser.add_argument("--initialization", choices=("uniform", "adverse"), default="adverse")
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--updates", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    results = [
        train(ExactTrainingConfig(method, args.initialization, args.alpha, args.beta, args.updates, args.gamma))
        for method in METHODS
    ]
    write_csv(output / "trajectories.csv", (row for result in results for row in result.trajectory))
    write_csv(output / "fisher_spectra.csv", (row for result in results for row in result.spectra))
    endpoints = [{**result.trajectory[-1], "checkpoint_count": len(result.spectra) // 2} for result in results]
    write_csv(output / "endpoints.csv", endpoints)
    plot_behavior(results, output / "behavior.png")
    plot_spectra(results, output / "fisher_geometry.png")
    plot_explained_trace(results, output / "fisher_explained_trace.png")
    configuration = {
        "schema_version": 1,
        "methods": list(METHODS),
        "initialization": args.initialization,
        "alpha": args.alpha,
        "beta": args.beta,
        "updates": args.updates,
        "gamma": args.gamma,
        "dtype": "float64",
        "occupancy": "normalized discounted occupancy including absorbing terminal state",
        "barrier": "0.5 * logdet of the exact reduced Fisher",
    }
    (output / "config.json").write_text(json.dumps(configuration, indent=2) + "\n", encoding="utf-8")
    (output / "verification.json").write_text(
        json.dumps(verification_results(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Saved exact MDP experiment to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
