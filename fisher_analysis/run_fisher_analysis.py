"""Command-line orchestration for fixed-policy Fisher-spectrum experiments."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import numpy as np
import torch

from fisher_analysis.fisher import (
    SCORE_BATCH_SIZE,
    analyze_fisher,
    assert_policy_unchanged,
    compute_empirical_fisher,
    parameter_layout,
    state_dict_snapshot,
)
from fisher_analysis.plotting import plot_spectra
from fisher_analysis.rollout import (
    collect_fixed_policy_batch,
    make_env_factory,
    multiprocessing_context,
    policy_seed,
)
from vpg.policy import build_policy


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "results" / "hopper_width_sweep"
)

def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty CSV file: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.widths or any(width <= 0 for width in args.widths):
        raise ValueError("--widths must contain positive integers")
    if len(set(args.widths)) != len(args.widths):
        raise ValueError("--widths must not contain duplicates")
    if args.depth <= 0:
        raise ValueError("--depth must be positive")
    for name in ("iterations", "n_envs", "trajectories_per_env"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.horizon < 0:
        raise ValueError("--horizon must be non-negative")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze undamped empirical Fisher spectra of fixed policies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-id", default="Hopper-v5")
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--trajectories-per-env", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def _experiment_config(args, output_dir: Path, vector_context) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "experiment": "fixed_policy_undamped_empirical_fisher",
        "env_id": args.env_id,
        "widths": list(args.widths),
        "depth": args.depth,
        "iterations": args.iterations,
        "n_envs": args.n_envs,
        "trajectories_per_env": args.trajectories_per_env,
        "horizon": args.horizon,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "dtype": "float64",
        "damping": 0.0,
        "init_log_std": -0.5,
        "learn_std": True,
        "score_batch_size": SCORE_BATCH_SIZE,
        "vectorization": "gymnasium.vector.AsyncVectorEnv",
        "multiprocessing_context": vector_context or "platform_default",
        "rank_definition": (
            "eigenvalues > P * eps(float64) * max(abs(eigenvalues))"
        ),
        "effective_rank_definition": (
            "exp(entropy(positive_eigenvalues / positive_trace))"
        ),
        "stable_rank_definition": (
            "sum(positive_eigenvalues^2) / largest_eigenvalue^2"
        ),
        "seed_schedule": {
            "policy": "(seed + 10007 * width) mod (2^63 - 1)",
            "environment": "(seed + 1000003 + flattened episode index) mod 2^32",
            "action": (
                "(seed + 2000003 + 100003 * width + iteration) mod (2^63 - 1)"
            ),
        },
        "policy_seeds": {
            str(width): policy_seed(args.seed, width) for width in args.widths
        },
    }


def _save_spectrum(
    output_dir: Path,
    width: int,
    layout,
    fisher_tensor,
    eigenvalues,
    metrics,
    sample_counts,
    rank_tolerance,
    psd_tolerance,
) -> None:
    fisher = fisher_tensor.detach().cpu().numpy()
    trace = float(metrics["trace"])
    normalized = eigenvalues / trace if trace > 0 else np.zeros_like(eigenvalues)
    cumulative_values = eigenvalues.copy()
    near_zero = np.abs(cumulative_values) <= psd_tolerance
    cumulative_values[near_zero] = np.maximum(cumulative_values[near_zero], 0.0)
    cumulative = np.cumsum(cumulative_values)
    if cumulative.size and cumulative[-1] > 0:
        cumulative /= cumulative[-1]

    np.savez_compressed(
        output_dir / f"fisher_width_{width}.npz",
        fisher=fisher,
        eigenvalues=eigenvalues,
        trace_normalized_eigenvalues=normalized,
        cumulative_explained_trace=cumulative,
        parameter_names=np.asarray([item["name"] for item in layout]),
        parameter_offsets=np.asarray(
            [[item["start"], item["stop"]] for item in layout],
            dtype=np.int64,
        ),
        parameter_shapes_json=np.asarray(
            [json.dumps(item["shape"]) for item in layout]
        ),
        parameter_layout_json=np.asarray(json.dumps(layout)),
        sample_counts=np.asarray(sample_counts, dtype=np.int64),
        total_sample_count=np.asarray(int(sum(sample_counts)), dtype=np.int64),
        rank_tolerance=np.asarray(rank_tolerance, dtype=np.float64),
        psd_tolerance=np.asarray(psd_tolerance, dtype=np.float64),
    )


def run_analysis(args: argparse.Namespace) -> list[dict[str, Any]]:
    _validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    vector_context = multiprocessing_context()
    config = _experiment_config(args, output_dir, vector_context)
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    summary_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    parameter_dimensions: dict[str, int] = {}

    vector_env_kwargs: dict[str, Any] = {
        "autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP,
    }
    if vector_context is not None:
        vector_env_kwargs["context"] = vector_context
    parallel_envs = gym.vector.AsyncVectorEnv(
        [make_env_factory(args.env_id) for _ in range(args.n_envs)],
        **vector_env_kwargs,
    )
    probe_env = gym.make(args.env_id)
    try:
        for width in args.widths:
            torch.manual_seed(policy_seed(args.seed, width))
            policy_config = {
                "hidden_sizes": [width] * args.depth,
                "policy": "mlp",
                "init_log_std": -0.5,
                "learn_std": True,
            }
            policy = build_policy(policy_config, probe_env).to(
                device="cpu",
                dtype=torch.float64,
            )
            policy.eval()
            initial_state = state_dict_snapshot(policy)
            layout = parameter_layout(policy)
            dimension = sum(item["numel"] for item in layout)
            parameter_dimensions[str(width)] = dimension
            torch.save(initial_state, checkpoint_dir / f"policy_width_{width}.pt")

            width_states: list[np.ndarray] = []
            width_actions: list[np.ndarray] = []
            sample_counts: list[int] = []
            print(
                f"Width {width}: P={dimension}, collecting "
                f"{args.iterations} seeded rollout batches"
            )
            for iteration in range(args.iterations):
                start = time.perf_counter()
                batch = collect_fixed_policy_batch(
                    policy,
                    parallel_envs,
                    iteration=iteration,
                    trajectories_per_env=args.trajectories_per_env,
                    horizon=args.horizon,
                    base_seed=args.seed,
                    width=width,
                )
                elapsed = time.perf_counter() - start
                assert_policy_unchanged(policy, initial_state)
                width_states.append(batch.states)
                width_actions.append(batch.actions)
                sample_counts.append(batch.states.shape[0])
                iteration_rows.append(
                    {
                        "width": width,
                        "iteration": iteration,
                        "trajectory_count": int(batch.episode_returns.size),
                        "sample_count": int(batch.states.shape[0]),
                        "invalid_sample_count": batch.invalid_sample_count,
                        "mean_return": float(np.mean(batch.episode_returns)),
                        "std_return": float(np.std(batch.episode_returns)),
                        "min_return": float(np.min(batch.episode_returns)),
                        "max_return": float(np.max(batch.episode_returns)),
                        "mean_length": float(np.mean(batch.episode_lengths)),
                        "min_length": int(np.min(batch.episode_lengths)),
                        "max_length": int(np.max(batch.episode_lengths)),
                        "rollout_seconds": elapsed,
                    }
                )
                print(
                    f"  iteration {iteration + 1:02d}/{args.iterations}: "
                    f"{batch.states.shape[0]} samples, "
                    f"mean return {np.mean(batch.episode_returns):.2f}"
                )

            states = np.concatenate(width_states, axis=0)
            actions = np.concatenate(width_actions, axis=0)
            print(f"  accumulating undamped Fisher from {states.shape[0]} scores")
            fisher_tensor = compute_empirical_fisher(policy, states, actions)
            assert_policy_unchanged(policy, initial_state)
            eigenvalues, metrics, rank_tolerance, psd_tolerance = analyze_fisher(
                fisher_tensor,
                sample_count=states.shape[0],
            )
            _save_spectrum(
                output_dir,
                width,
                layout,
                fisher_tensor,
                eigenvalues,
                metrics,
                sample_counts,
                rank_tolerance,
                psd_tolerance,
            )

            summary_rows.append(
                {
                    "width": width,
                    "depth": args.depth,
                    **metrics,
                    "rank_tolerance": rank_tolerance,
                    "psd_tolerance": psd_tolerance,
                }
            )
            spectra.append(
                {
                    "width": width,
                    "eigenvalues": eigenvalues,
                    "trace": float(metrics["trace"]),
                    "rank_tolerance": rank_tolerance,
                    "psd_tolerance": psd_tolerance,
                }
            )
            print(
                f"  trace={metrics['trace']:.6g}, "
                f"rank={metrics['numerical_rank']}/{dimension}, "
                f"condition={metrics['positive_condition_number']:.3e}"
            )
    finally:
        probe_env.close()
        parallel_envs.close()

    config["parameter_dimensions"] = parameter_dimensions
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    _write_csv(output_dir / "iteration_stats.csv", iteration_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    plot_spectra(spectra, output_dir)
    print(f"Saved Fisher analysis to {output_dir}")
    return summary_rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_analysis(args)
    except (ValueError, AssertionError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
