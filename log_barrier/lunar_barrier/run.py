"""Resumable smoke selection and paired reliability study for LunarLander."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from .statistics import (
    paired_reliability_summary,
    rank_configurations,
    select_barrier_configuration,
)
from .training import LunarLanderConfig, train_and_evaluate


def _parse_seeds(specification: str) -> list[int]:
    seeds = [int(value.strip()) for value in specification.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("seeds must be a non-empty list of unique integers")
    return seeds


def _result_path(output: Path, config: LunarLanderConfig) -> Path:
    identity = config.to_dict()
    identity.pop("seed")
    identity.pop("method")
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    return output / "runs" / f"config_{digest}" / f"seed_{config.seed}__{config.method}.json"


def _run_one(config_dict: dict, path_string: str) -> dict:
    config_dict = dict(config_dict)
    config_dict["hidden_sizes"] = tuple(config_dict["hidden_sizes"])
    config_dict.pop("optimizer", None)
    config = LunarLanderConfig(**config_dict)
    result = train_and_evaluate(config)
    path = Path(path_string)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)
    return result


def _execute(configurations: list[LunarLanderConfig], output: Path, workers: int) -> list[dict]:
    results: list[dict] = []
    pending = []
    for config in configurations:
        path = _result_path(output, config)
        if path.is_file():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result.get("config") != config.to_dict():
                raise ValueError(
                    f"saved run configuration does not match this invocation: {path}. "
                    "Use a different --output or restore the original arguments."
                )
            results.append(result)
        else:
            pending.append((config, path))
    print(f"Loaded {len(results)} completed runs; {len(pending)} runs remain.")
    if workers == 1:
        for index, (config, path) in enumerate(pending, 1):
            print(f"[{index}/{len(pending)}] seed={config.seed} method={config.method} lr={config.learning_rate:g} beta={config.beta:g}")
            results.append(_run_one(config.to_dict(), str(path)))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_one, config.to_dict(), str(path)): config
                for config, path in pending
            }
            for index, future in enumerate(as_completed(futures), 1):
                config = futures[future]
                results.append(future.result())
                print(f"[{index}/{len(pending)}] completed seed={config.seed} method={config.method}")
    return results


def _write_aggregate(path: Path, results: list[dict]) -> None:
    rows = []
    for result in results:
        config = result["config"]
        rows.append(
            {
                "seed": result["seed"],
                "method": result["method"],
                "learning_rate": config["learning_rate"],
                "beta": config["beta"],
                "gamma": config["gamma"],
                "episodes_per_update": config["episodes_per_update"],
                "center_returns": config["center_returns"],
                "normalize_returns": config["normalize_returns"],
                "handoff_fraction": config["handoff_fraction"],
                "updates": config["updates"],
                "evaluation_episodes": config["evaluation_episodes"],
                "environment_steps": result["environment_steps"],
                "final_training_mean": result["final_training_mean"],
                "stochastic_evaluation_mean": result["stochastic_evaluation_mean"],
                "deterministic_evaluation_mean": result["deterministic_evaluation_mean"],
                "mean_min_probability": result["mean_min_probability"],
            }
        )
    rows.sort(
        key=lambda row: (
            row["seed"],
            row["learning_rate"],
            row["gamma"],
            row["episodes_per_update"],
            row["beta"],
            row["handoff_fraction"],
            row["method"],
        )
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--updates", type=int, required=True)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1)


def _return_settings(mode: str) -> tuple[bool, bool]:
    if mode == "centered":
        return True, False
    if mode == "centered_normalized":
        return True, True
    if mode == "raw":
        return False, False
    raise ValueError(f"unknown return mode: {mode}")


def _config(args, method: str, seed: int, settings: dict) -> LunarLanderConfig:
    center_returns, normalize_returns = _return_settings(settings["return_mode"])
    return LunarLanderConfig(
        method=method,
        seed=seed,
        learning_rate=float(settings["learning_rate"]),
        beta=float(settings.get("beta", 0.0)),
        hidden_sizes=(8, 8),
        gamma=float(settings["gamma"]),
        updates=args.updates,
        episodes_per_update=int(settings["episodes_per_update"]),
        horizon=args.horizon,
        center_returns=center_returns,
        normalize_returns=normalize_returns,
        evaluation_episodes=32,
        handoff_fraction=float(settings.get("handoff_fraction", 0.25)),
        device=args.device,
    )


def _smoke(args) -> int:
    seeds = _parse_seeds(args.seeds)
    base_settings = [
        {
            "learning_rate": learning_rate,
            "gamma": gamma,
            "episodes_per_update": episodes_per_update,
            "return_mode": return_mode,
        }
        for learning_rate in args.learning_rates
        for gamma in args.gammas
        for episodes_per_update in args.episodes_per_update_options
        for return_mode in args.return_modes
    ]
    base_configurations = [
        _config(args, "reward_only", seed, settings)
        for settings in base_settings
        for seed in seeds
    ]
    for learning_rate in args.learning_rates:
        if learning_rate <= 0.0:
            raise ValueError("learning rates must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    print(f"Stage 1: {len(base_settings)} GPOMDP configurations")
    base_results = _execute(base_configurations, args.output, args.workers)
    base_fields = (
        "learning_rate",
        "gamma",
        "episodes_per_update",
        "center_returns",
        "normalize_returns",
    )
    base_ranking = rank_configurations(
        base_results, base_fields, args.lower_quantile
    )
    selected_base = base_ranking[0]
    selected_settings = {
        key: selected_base[key]
        for key in ("learning_rate", "gamma", "episodes_per_update")
    }
    selected_settings["return_mode"] = (
        "centered_normalized"
        if selected_base["center_returns"] and selected_base["normalize_returns"]
        else "centered"
        if selected_base["center_returns"]
        else "raw"
    )
    selected_baselines = [
        result
        for result in base_results
        if all(result["config"][field] == selected_base[field] for field in base_fields)
    ]
    barrier_settings = [
        {
            **selected_settings,
            "beta": beta,
            "handoff_fraction": handoff_fraction,
        }
        for beta in args.betas
        for handoff_fraction in args.handoff_fractions
    ]
    barrier_configurations = [
        _config(args, "log_barrier", seed, settings)
        for settings in barrier_settings
        for seed in seeds
    ]
    print(f"Stage 2: {len(barrier_settings)} barrier configurations")
    barrier_results = _execute(barrier_configurations, args.output, args.workers)
    barrier_selection = select_barrier_configuration(
        selected_baselines, barrier_results, args.lower_quantile
    )
    selected = {
        key: selected_base[key] for key in base_fields
    }
    selected.update(
        beta=barrier_selection["selected"]["beta"],
        handoff_fraction=barrier_selection["selected"]["handoff_fraction"],
    )
    selection = {
        "selected": selected,
        "base_ranking": base_ranking,
        "barrier_ranking": barrier_selection["ranking"],
        "lower_quantile": args.lower_quantile,
    }
    selection["smoke_seeds"] = seeds
    selection["fixed"] = {
        "hidden_sizes": [8, 8],
        "evaluation_episodes": 32,
    }
    selection["smoke_budget"] = {
        "updates": args.updates,
        "horizon": args.horizon,
    }
    retained_results = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.output / "runs").rglob("*.json"))
    ]
    selection["retained_run_count"] = len(retained_results)
    _write_aggregate(args.output / "smoke_results.csv", retained_results)
    (args.output / "selection.json").write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("Selected configuration:")
    print(json.dumps(selection["selected"], indent=2, sort_keys=True))
    print(f"Saved selection to {args.output / 'selection.json'}")
    return 0


def _reliability(args) -> int:
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    smoke_seeds = set(selection.get("smoke_seeds", []))
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    overlap = smoke_seeds.intersection(seeds)
    if overlap:
        raise ValueError(f"confirmatory seeds overlap smoke seeds: {sorted(overlap)}")
    selected = selection["selected"]
    return_mode = (
        "centered_normalized"
        if selected["center_returns"] and selected["normalize_returns"]
        else "centered"
        if selected["center_returns"]
        else "raw"
    )
    settings = {
        "learning_rate": selected["learning_rate"],
        "gamma": selected["gamma"],
        "episodes_per_update": selected["episodes_per_update"],
        "return_mode": return_mode,
        "beta": selected["beta"],
        "handoff_fraction": selected["handoff_fraction"],
    }
    configurations = [
        _config(
            args,
            method,
            seed,
            {**settings, "beta": 0.0 if method == "reward_only" else settings["beta"]},
        )
        for seed in seeds
        for method in ("reward_only", "log_barrier")
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    predeclaration = {
        "schema_version": 1,
        "stage": "paired_reliability",
        "outcomes_used_for_configuration": False,
        "selection_file": str(args.selection),
        "locked_selection": selected,
        "fixed": {"hidden_sizes": [8, 8], "evaluation_episodes": 32},
        "updates": args.updates,
        "horizon": args.horizon,
        "paired_seed_count": len(seeds),
        "seeds": seeds,
        "smoke_seeds": sorted(smoke_seeds),
        "seeds_disjoint_from_smoke": True,
        "methods": ["reward_only", "log_barrier"],
        "primary_endpoint": "paired catastrophic failure rate",
        "catastrophic_failure_definition": (
            f"mean stochastic return across 32 episodes < {args.failure_threshold}"
        ),
        "primary_test": "two-sided exact McNemar on discordant pairs",
        "secondary_endpoints": [
            "mean stochastic evaluation return",
            "solved rate at mean return >= 200",
            "paired win rate",
        ],
        "pairing": (
            "both arms share policy initialization, training environment seeds, "
            "evaluation environment seeds, and evaluation action uniforms"
        ),
    }
    (args.output / "predeclaration.json").write_text(
        json.dumps(predeclaration, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    results = _execute(configurations, args.output, args.workers)
    summary = paired_reliability_summary(
        results,
        args.bootstrap_samples,
        args.bootstrap_seed,
        args.failure_threshold,
    )
    summary["locked_selection"] = selected
    summary["selection_file"] = str(args.selection)
    _write_aggregate(args.output / "paired_results.csv", results)
    (args.output / "reliability_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    interval = summary["paired_difference_log_barrier_minus_reward_only"]["bootstrap_95_percent_ci"]
    mean = summary["paired_difference_log_barrier_minus_reward_only"]["mean"]
    print(f"Paired mean difference: {mean:.3f} (bootstrap 95% CI [{interval[0]:.3f}, {interval[1]:.3f}])")
    print(f"Saved reliability summary to {args.output / 'reliability_summary.json'}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="phase", required=True)
    smoke = subparsers.add_parser("smoke", help="screen hyperparameters on a small seed set")
    _common(smoke)
    smoke.add_argument("--output", type=Path, default=Path("results/log_barrier/lunar_barrier/smoke"))
    smoke.add_argument("--seeds", default="101,102,103,104,105")
    smoke.add_argument("--learning-rates", type=float, nargs="+", default=[0.001, 0.003])
    smoke.add_argument("--gammas", type=float, nargs="+", default=[0.97, 0.99])
    smoke.add_argument(
        "--episodes-per-update-options", type=int, nargs="+", default=[4, 8]
    )
    smoke.add_argument(
        "--return-modes",
        nargs="+",
        choices=("centered", "centered_normalized", "raw"),
        default=["centered", "centered_normalized"],
    )
    smoke.add_argument(
        "--betas", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0, 40.0]
    )
    smoke.add_argument(
        "--handoff-fractions", type=float, nargs="+", default=[0.05, 0.10, 0.25]
    )
    smoke.add_argument("--lower-quantile", type=float, default=0.25)
    smoke.set_defaults(handler=_smoke)

    reliability = subparsers.add_parser("reliability", help="run the locked paired-seed study")
    _common(reliability)
    reliability.add_argument("--selection", type=Path, required=True)
    reliability.add_argument("--output", type=Path, default=Path("results/log_barrier/lunar_barrier/reliability"))
    reliability.add_argument("--seed-start", type=int, default=10_000)
    reliability.add_argument("--n-seeds", type=int, default=200)
    reliability.add_argument("--bootstrap-samples", type=int, default=10_000)
    reliability.add_argument("--bootstrap-seed", type=int, default=20260821)
    reliability.add_argument("--failure-threshold", type=float, default=-100.0)
    reliability.set_defaults(handler=_reliability)
    args = parser.parse_args(argv)
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.phase == "smoke" and not 0.0 <= args.lower_quantile <= 1.0:
        parser.error("--lower-quantile must be between zero and one")
    if args.phase == "reliability" and args.n_seeds < 2:
        parser.error("--n-seeds must be at least two")
    if args.phase == "reliability" and args.bootstrap_samples < 1:
        parser.error("--bootstrap-samples must be positive")
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
