"""Recompute trajectory Fisher checkpoint spectra without retraining.

The saved training artifacts use float32 Fisher matrices.  This command loads
the policy checkpoints, collects fresh seeded trajectories, and recomputes the
trajectory Fisher entirely in float64.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from fisher_analysis.fisher import (
    analyze_fisher,
    assert_policy_unchanged,
    parameter_layout,
    state_dict_snapshot,
)
from fisher_log_barrier.loss1 import compute_trajectory_fisher
from vpg.data_collection import collect_parallel_trajectories
from vpg.policy import build_policy


DEFAULT_RUN_DIR = Path(
    "results/fisher_log_barrier/swimmer/16x16/seed_24_beta1pct"
)
CHECKPOINT_PATTERNS = (
    re.compile(r"update_(\d+)\.pt$"),
    re.compile(r"snapshot_iter_(\d+)\.pt$"),
)


class ActionRepeat(gym.Wrapper):
    """Hold one policy action for several primitive environment steps."""

    def __init__(self, env: gym.Env, repeat: int) -> None:
        super().__init__(env)
        self.repeat = repeat

    def step(self, action):
        total_reward = 0.0
        for _ in range(self.repeat):
            observation, reward, terminated, truncated, info = self.env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return observation, total_reward, terminated, truncated, info


def _make_env(config: dict[str, Any], horizon: int):
    def thunk():
        env = gym.make(config["env_id"], max_episode_steps=horizon)
        action_repeat = int(config.get("action_repeat", 1))
        return ActionRepeat(env, action_repeat) if action_repeat > 1 else env

    return thunk


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Recompute checkpoint trajectory-Fisher spectra in float64.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--updates", type=int, nargs="*")
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--trajectories-per-env", type=int)
    parser.add_argument("--horizon", type=int)
    parser.add_argument("--diagnostic-seed", type=int, default=1_000_024)
    parser.add_argument("--score-backend", choices=("vmap", "loop"), default="vmap")
    parser.add_argument("--force", action="store_true")
    return parser


def _checkpoint_paths(run_dir: Path) -> dict[int, Path]:
    paths: dict[int, Path] = {}
    for path in (run_dir / "checkpoints").glob("*.pt"):
        match = next(
            (
                pattern.match(path.name)
                for pattern in CHECKPOINT_PATTERNS
                if pattern.match(path.name)
            ),
            None,
        )
        if match:
            paths[int(match.group(1))] = path
    if not paths:
        raise FileNotFoundError(f"no checkpoints found under {run_dir / 'checkpoints'}")
    return paths


def _write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_summary(output_dir: Path) -> None:
    rows = []
    for path in sorted((output_dir / "metadata").glob("update_*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    if not rows:
        return
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _load_policy(probe_env, config: dict[str, Any], checkpoint_path: Path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("policy_state_dict", checkpoint)
    policy = build_policy(config, probe_env).to(device="cpu", dtype=torch.float64)
    policy.load_state_dict(state_dict)
    policy.eval()
    return policy, checkpoint


def _spectral_metadata(
    *,
    update: int,
    checkpoint_path: Path,
    trajectories,
    fisher: torch.Tensor,
    scores: torch.Tensor,
    eigenvalues: np.ndarray,
    metrics: dict[str, float | int],
    rank_tolerance: float,
    psd_tolerance: float,
    elapsed_seconds: float,
) -> dict[str, Any]:
    episode_returns = np.asarray(
        [sum(trajectory.rewards) for trajectory in trajectories],
        dtype=np.float64,
    )
    episode_lengths = np.asarray(
        [len(trajectory.rewards) for trajectory in trajectories],
        dtype=np.int64,
    )
    positive = eigenvalues[eigenvalues > rank_tolerance]
    logdet = float(np.log(positive).sum()) if positive.size else float("nan")
    dimension = int(fisher.shape[0])
    return {
        "update": update,
        "checkpoint": str(checkpoint_path),
        "trajectory_count": len(trajectories),
        "transition_count": int(episode_lengths.sum()),
        "parameter_count": dimension,
        "fisher_dtype": str(fisher.dtype),
        "score_dtype": str(scores.dtype),
        "numerical_rank": int(metrics["numerical_rank"]),
        "minimum_eigenvalue": float(eigenvalues[-1]),
        "maximum_eigenvalue": float(eigenvalues[0]),
        "positive_condition_number": float(metrics["positive_condition_number"]),
        "logdet_positive_spectrum": logdet,
        "normalized_logdet_positive_spectrum": logdet / dimension,
        "effective_rank": float(metrics["effective_rank"]),
        "stable_rank": float(metrics["stable_rank"]),
        "components_90": int(metrics["components_90"]),
        "components_95": int(metrics["components_95"]),
        "components_99": int(metrics["components_99"]),
        "rank_tolerance": float(rank_tolerance),
        "psd_tolerance": float(psd_tolerance),
        "mean_return": float(episode_returns.mean()),
        "std_return": float(episode_returns.std()),
        "mean_episode_length": float(episode_lengths.mean()),
        "elapsed_seconds": elapsed_seconds,
    }


def run(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_dir / "float64_spectra"
    )
    fisher_dir = output_dir / "fishers"
    metadata_dir = output_dir / "metadata"
    fisher_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_paths = _checkpoint_paths(run_dir)
    updates = sorted(checkpoint_paths) if args.updates is None else sorted(set(args.updates))
    missing = [update for update in updates if update not in checkpoint_paths]
    if missing:
        raise ValueError(f"requested checkpoint updates do not exist: {missing}")
    if args.n_envs <= 0:
        raise ValueError("--n-envs must be positive")
    trajectories_per_env = args.trajectories_per_env
    if trajectories_per_env is None:
        total = int(config["fisher_trajectory_count"])
        if total % args.n_envs:
            raise ValueError("fisher trajectory count is not divisible by --n-envs")
        trajectories_per_env = total // args.n_envs
    if trajectories_per_env <= 0:
        raise ValueError("--trajectories-per-env must be positive")
    horizon = int(config["horizon"] if args.horizon is None else args.horizon)
    if horizon <= 0:
        raise ValueError("--horizon must be positive")

    recompute_config = {
        "schema_version": 1,
        "object": "trajectory_score_empirical_fisher",
        "source_run": str(run_dir),
        "source_config": config,
        "updates": updates,
        "n_envs": args.n_envs,
        "trajectories_per_env": trajectories_per_env,
        "trajectory_count": args.n_envs * trajectories_per_env,
        "horizon": horizon,
        "action_repeat": int(config.get("action_repeat", 1)),
        "diagnostic_seed": args.diagnostic_seed,
        "score_backend": args.score_backend,
        "policy_dtype": "torch.float64",
        "score_dtype": "torch.float64",
        "fisher_dtype": "torch.float64",
        "action_storage": "raw Gaussian samples",
        "environment_actions": "clipped to action-space bounds",
        "common_random_numbers_across_checkpoints": True,
    }
    _write_json(output_dir / "config.json", recompute_config)

    torch.set_num_threads(int(config.get("torch_threads", torch.get_num_threads())))
    probe_env = gym.make(config["env_id"])
    try:
        for update in updates:
            output_path = fisher_dir / f"update_{update:06d}.npz"
            metadata_path = metadata_dir / f"update_{update:06d}.json"
            if output_path.exists() and metadata_path.exists() and not args.force:
                print(f"Update {update:04d}: already complete; skipping", flush=True)
                continue

            print(
                f"Update {update:04d}: collecting "
                f"{args.n_envs * trajectories_per_env} trajectories",
                flush=True,
            )
            started = time.perf_counter()
            policy, checkpoint = _load_policy(
                probe_env,
                config,
                checkpoint_paths[update],
            )
            initial_state = state_dict_snapshot(policy)
            layout = parameter_layout(policy)
            torch.manual_seed(args.diagnostic_seed)
            np.random.seed(args.diagnostic_seed % (2**32))
            envs = gym.vector.AsyncVectorEnv(
                [_make_env(config, horizon) for _ in range(args.n_envs)]
            )
            try:
                reset_seeds = [
                    [
                        args.diagnostic_seed
                        + trajectory_index * args.n_envs
                        + worker
                        for worker in range(args.n_envs)
                    ]
                    for trajectory_index in range(trajectories_per_env)
                ]
                trajectories = collect_parallel_trajectories(
                    envs,
                    policy,
                    n_trajectories_per_env=trajectories_per_env,
                    clip_actions=bool(config["clip_actions"]),
                    device=torch.device("cpu"),
                    reset_seeds=reset_seeds,
                )
            finally:
                envs.close()

            fisher, scores = compute_trajectory_fisher(
                policy,
                trajectories,
                score_backend=args.score_backend,
                device="cpu",
            )
            assert_policy_unchanged(policy, initial_state)
            eigenvalues, metrics, rank_tolerance, psd_tolerance = analyze_fisher(
                fisher,
                sample_count=len(trajectories),
            )
            elapsed_seconds = time.perf_counter() - started
            metadata = _spectral_metadata(
                update=update,
                checkpoint_path=checkpoint_paths[update],
                trajectories=trajectories,
                fisher=fisher,
                scores=scores,
                eigenvalues=eigenvalues,
                metrics=metrics,
                rank_tolerance=rank_tolerance,
                psd_tolerance=psd_tolerance,
                elapsed_seconds=elapsed_seconds,
            )
            np.savez_compressed(
                output_path,
                fisher=fisher.cpu().numpy(),
                trajectory_scores=scores.cpu().numpy(),
                eigenvalues=eigenvalues,
                parameter_layout_json=np.asarray(json.dumps(layout)),
                policy_update=np.asarray(
                    int(checkpoint.get("policy_update", update)),
                    dtype=np.int64,
                ),
                trajectory_count=np.asarray(len(trajectories), dtype=np.int64),
                transition_count=np.asarray(
                    sum(len(trajectory.rewards) for trajectory in trajectories),
                    dtype=np.int64,
                ),
                diagnostic_seed=np.asarray(args.diagnostic_seed, dtype=np.int64),
                rank_tolerance=np.asarray(rank_tolerance, dtype=np.float64),
                psd_tolerance=np.asarray(psd_tolerance, dtype=np.float64),
            )
            _write_json(metadata_path, metadata)
            _write_summary(output_dir)
            print(
                f"  lambda_min={metadata['minimum_eigenvalue']:.6e}, "
                f"condition={metadata['positive_condition_number']:.6e}, "
                f"rank={metadata['numerical_rank']}/{metadata['parameter_count']}, "
                f"elapsed={elapsed_seconds:.1f}s",
                flush=True,
            )
    finally:
        probe_env.close()
    _write_summary(output_dir)


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
