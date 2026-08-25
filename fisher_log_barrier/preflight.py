"""Feasibility preflight for the literal Acrobot trajectory Fisher barrier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from vpg.data_collection import collect_parallel_trajectories
from vpg.train import make_env

from .loss1 import SCORE_BACKENDS, FisherLogDetDomainError, estimate_trajectory_fisher_inverse
from .policy import ReferenceMLPSoftmaxPolicy


def run_preflight(
    *,
    seed: int = 1,
    episodes_per_update: int = 256,
    parallel_envs: int = 16,
    hidden_sizes: tuple[int, ...] = (8, 8),
    horizon: int = 500,
    mu: float = 1e-10,
    score_backend: str = "vmap",
    device: str = "cpu",
) -> dict:
    """Collect one normal batch and report strict barrier feasibility."""

    if episodes_per_update < 1:
        raise ValueError("episodes_per_update must be positive")
    if parallel_envs < 1 or episodes_per_update % parallel_envs != 0:
        raise ValueError("episodes_per_update must be divisible by positive parallel_envs")
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch_device = torch.device(device)
    probe = gym.make("Acrobot-v1")
    try:
        policy = ReferenceMLPSoftmaxPolicy(
            state_dim=probe.observation_space.shape[0],
            action_dim=probe.action_space.n,
            hidden_sizes=hidden_sizes,
        ).to(torch_device)
    finally:
        probe.close()
    envs = gym.vector.SyncVectorEnv(
        [
            make_env("Acrobot-v1", seed + worker, horizon)
            for worker in range(parallel_envs)
        ]
    )
    try:
        trajectories = collect_parallel_trajectories(
            envs,
            policy,
            episodes_per_update // parallel_envs,
            device=torch_device,
        )
    finally:
        envs.close()

    base = {
        "schema_version": 1,
        "object": "trajectory_score_fisher_logdet_loss1",
        "env_id": "Acrobot-v1",
        "seed": seed,
        "hidden_sizes": list(hidden_sizes),
        "horizon": horizon,
        "episodes_per_update": episodes_per_update,
        "parallel_envs": parallel_envs,
        "policy_parameterization": "reference",
        "mu": mu,
        "score_backend": score_backend,
    }
    try:
        estimate = estimate_trajectory_fisher_inverse(
            policy,
            trajectories,
            mu=mu,
            score_backend=score_backend,
            device=torch_device,
        )
    except FisherLogDetDomainError as error:
        return {
            **base,
            **error.to_dict(),
            "feasible": False,
            "training_can_proceed": False,
        }
    return {
        **base,
        **estimate.to_dict(),
        "feasible": True,
        "training_can_proceed": True,
        "failure_reason": None,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/fisher_log_barrier/acrobot/preflight.json"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--episodes-per-update", type=int, default=256)
    parser.add_argument("--parallel-envs", type=int, default=16)
    parser.add_argument("--hidden-sizes", type=int, nargs="+", default=[8, 8])
    parser.add_argument("--horizon", type=int, default=500)
    parser.add_argument("--mu", type=float, default=1e-10)
    parser.add_argument("--score-backend", choices=SCORE_BACKENDS, default="vmap")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    result = run_preflight(
        seed=args.seed,
        episodes_per_update=args.episodes_per_update,
        parallel_envs=args.parallel_envs,
        hidden_sizes=tuple(args.hidden_sizes),
        horizon=args.horizon,
        mu=args.mu,
        score_backend=args.score_backend,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["feasible"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
