import argparse
import json
import os
import time
import traceback

from vpg.tune import run_search


_DEFAULT_ENVS = ["Swimmer-v5", "Hopper-v5", "HalfCheetah-v5"]
_DEFAULT_ALGORITHMS = ["gpomdp", "npg"]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Runs tune.py's Optuna search over every (env, algorithm) combination in one go "
            "-- e.g. Swimmer/Hopper/HalfCheetah x GPOMDP/NPG -- and writes a combined summary. "
            "Per-trial search settings below mirror tune.py's own flags; see tune.py for details."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--envs", type=str, nargs="+", default=_DEFAULT_ENVS,
        help="Environment IDs to sweep over.",
    )
    parser.add_argument(
        "--algorithms", type=str, nargs="+", default=_DEFAULT_ALGORITHMS, choices=["gpomdp", "npg"],
        help="Algorithms to sweep over.",
    )
    parser.add_argument(
        "--storage", type=str, default="sqlite:///runs/optuna/studies.db",
        help=(
            "Shared Optuna storage URL for every combo's study (each gets its own study name, "
            "'<algorithm>_<env_id>'). Persistent by default (unlike tune.py's own in-memory "
            "default) so an interrupted sweep can be resumed combo-by-combo."
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default=os.path.join("runs", "optuna"),
        help="Base directory; each combo's best_params.json goes to <output_dir>/<env_id>/<algorithm>/.",
    )

    parser.add_argument("--seed", type=int, default=23, help="Base seed, forwarded to every combo.")
    parser.add_argument(
        "--n_trials", type=int, default=15,
        help=(
            "Grid points per combination. The default runs the 15-point screening phase; "
            "see tune.py for the complete grids."
        ),
    )
    parser.add_argument(
        "--n_iterations", type=int, default=2000,
        help=(
            "Policy-update cycles per seed; every cycle collects "
            "n_envs * n_trajectories complete trajectories."
        ),
    )
    parser.add_argument("--n_tuning_seeds", type=int, default=1, help="Seeds per grid point.")
    parser.add_argument("--n_envs", type=int, default=8, help="Parallel environments per trial.")
    parser.add_argument("--n_trajectories", type=int, default=1, help="Episodes per env per iteration.")
    parser.add_argument("--horizon", type=int, default=1000, help="Max episode length (MuJoCo default).")
    parser.add_argument("--hidden_sizes", type=str, default="32,32", help="Fixed hidden layer sizes.")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    combos = [(env_id, algorithm) for env_id in args.envs for algorithm in args.algorithms]
    print(f"Sweeping {len(combos)} combinations: {combos}\n")

    summary = {}
    for i, (env_id, algorithm) in enumerate(combos, 1):
        print(f"\n{'=' * 70}")
        print(f"[{i}/{len(combos)}] {algorithm.upper()} on {env_id}")
        print(f"{'=' * 70}")

        combo_args = argparse.Namespace(
            algorithm=algorithm,
            env_id=env_id,
            seed=args.seed,
            n_trials=args.n_trials,
            n_iterations=args.n_iterations,
            n_tuning_seeds=args.n_tuning_seeds,
            n_envs=args.n_envs,
            n_trajectories=args.n_trajectories,
            horizon=args.horizon,
            hidden_sizes=args.hidden_sizes,
            study_name=f"{algorithm}_{env_id}",
            storage=args.storage,
            output_dir=os.path.join(args.output_dir, env_id, algorithm),
        )

        t0 = time.perf_counter()
        try:
            result = run_search(combo_args)
            result["status"] = "ok"
        except Exception:
            # One combo's failure (e.g. a MuJoCo/env-specific issue) shouldn't abort the rest
            # of an overnight sweep -- record it and move on to the next combination.
            traceback.print_exc()
            result = {"algorithm": algorithm, "env_id": env_id, "status": "failed"}
        result["wall_time_s"] = time.perf_counter() - t0

        summary.setdefault(env_id, {})[algorithm] = result

    summary_path = os.path.join(args.output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'=' * 70}")
    print("Sweep complete")
    print(f"{'=' * 70}")
    for env_id, per_algo in summary.items():
        for algorithm, result in per_algo.items():
            if result.get("status") == "ok":
                print(
                    f"{env_id:16s} {algorithm:8s} best_value={result['best_value']:.2f}  "
                    f"({result['wall_time_s']:.0f}s)  params={result['best_params']}"
                )
            else:
                print(f"{env_id:16s} {algorithm:8s} FAILED  ({result['wall_time_s']:.0f}s)")
    print(f"\nSaved combined summary to: {summary_path}")


if __name__ == "__main__":
    main()
