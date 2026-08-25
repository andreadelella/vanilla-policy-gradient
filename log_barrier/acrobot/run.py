"""CLI for the Acrobot reward-only versus categorical-barrier experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .reporting import plot_fisher, plot_training
from fisher_log_barrier import SCORE_BACKENDS

from .training import AcrobotConfig, METHODS, POLICY_PARAMETERIZATIONS, train


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results/log_barrier/acrobot/new_runs"))
    parser.add_argument("--seeds", default="1")
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=["reward_only", "log_barrier"],
    )
    parser.add_argument("--updates", type=int, default=1000)
    parser.add_argument("--episodes-per-update", type=int, default=8)
    parser.add_argument("--fisher-episodes-per-update", type=int, default=256)
    parser.add_argument("--fisher-parallel-envs", type=int, default=16)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[8, 8])
    parser.add_argument(
        "--policy-parameterization",
        choices=POLICY_PARAMETERIZATIONS,
        default="auto",
    )
    parser.add_argument("--learning-rate", type=float, default=0.003)
    parser.add_argument("--beta", type=float, default=546.4135158976487)
    parser.add_argument("--fisher-mu", type=float, default=1e-10)
    parser.add_argument("--fisher-beta", type=float, default=1.0)
    parser.add_argument(
        "--fisher-score-backend",
        choices=SCORE_BACKENDS,
        default="vmap",
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value.strip()) for value in args.seeds.split(",") if value.strip()]
    all_training, all_spectra, configurations = [], [], []
    for seed in seeds:
        for method in args.methods:
            config = AcrobotConfig(
                method=method,
                seed=seed,
                updates=args.updates,
                episodes_per_update=args.episodes_per_update,
                fisher_episodes_per_update=args.fisher_episodes_per_update,
                fisher_parallel_envs=args.fisher_parallel_envs,
                hidden_sizes=tuple(args.hidden_sizes),
                learning_rate=args.learning_rate,
                beta=args.beta,
                fisher_mu=args.fisher_mu,
                fisher_beta=args.fisher_beta,
                fisher_score_backend=args.fisher_score_backend,
                policy_parameterization=args.policy_parameterization,
                device=args.device,
            )
            run_directory = args.output / config.run_id
            training_rows, spectrum_rows = train(config, run_directory / "checkpoints")
            all_training.extend(training_rows)
            all_spectra.extend(spectrum_rows)
            configurations.append(config.to_dict())
            _write_csv(run_directory / "training.csv", training_rows)
            _write_csv(run_directory / "fisher_spectra.csv", spectrum_rows)
    _write_csv(args.output / "training.csv", all_training)
    _write_csv(args.output / "fisher_spectra.csv", all_spectra)
    (args.output / "config.json").write_text(json.dumps(configurations, indent=2, sort_keys=True), encoding="utf-8")
    plot_training(all_training, args.output / "training.png")
    plot_fisher(all_spectra, args.output / "fisher_spectra.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
