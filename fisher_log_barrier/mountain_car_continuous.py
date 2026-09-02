"""Train fixed or gradient-clipped Fisher regularization on MountainCarContinuous."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from fisher_log_barrier.loss1 import trajectory_fisher_logdet_surrogate
from vpg.data_collection import collect_parallel_trajectories
from vpg.gpomdp import compute_gpomdp_loss
from vpg.policy import GaussianPolicy


ENV_ID = "MountainCarContinuous-v0"


@dataclass(frozen=True)
class MountainCarContinuousConfig:
    seed: int = 101
    hidden_sizes: tuple[int, ...] = (2, 2)
    updates: int = 250
    reward_trajectory_count: int = 32
    fisher_trajectory_count: int = 256
    workers: int = 16
    horizon: int = 500
    gamma: float = 1.0
    learning_rate: float = 0.02
    fisher_beta: float = 0.1
    fisher_mu: float = 0.0
    clip_ratio: float | None = None
    checkpoint_interval: int = 50
    torch_threads: int = 8

    @property
    def reward_trajectories_per_worker(self) -> int:
        return self.reward_trajectory_count // self.workers

    @property
    def fisher_trajectories_per_worker(self) -> int:
        return self.fisher_trajectory_count // self.workers

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["hidden_sizes"] = list(self.hidden_sizes)
        value["env_id"] = ENV_ID
        value["mode"] = "fixed" if self.clip_ratio is None else "clipped"
        return value


def _make_env(config: MountainCarContinuousConfig):
    def thunk():
        return gym.make(ENV_ID, max_episode_steps=config.horizon)

    return thunk


def _build_policy(config: MountainCarContinuousConfig) -> GaussianPolicy:
    return GaussianPolicy(
        state_dim=2,
        action_dim=1,
        hidden_sizes=config.hidden_sizes,
        init_log_std=-0.5,
        learn_std=True,
    )


def _seed_batches(
    config: MountainCarContinuousConfig,
    update: int,
    stream: int,
    trajectories_per_worker: int,
) -> list[list[int]]:
    base = config.seed + update * 10_007 + stream * 1_000_003
    return [
        [
            (base + trajectory * config.workers + worker) % (2**32)
            for worker in range(config.workers)
        ]
        for trajectory in range(trajectories_per_worker)
    ]


def _collect(
    envs,
    policy: GaussianPolicy,
    config: MountainCarContinuousConfig,
    update: int,
    stream: int,
    trajectories_per_worker: int,
):
    torch.manual_seed(config.seed + update * 1009 + stream * 100_003)
    return collect_parallel_trajectories(
        envs,
        policy,
        trajectories_per_worker,
        clip_actions=True,
        device="cpu",
        reset_seeds=_seed_batches(
            config,
            update,
            stream,
            trajectories_per_worker,
        ),
    )


def _flatten(gradients: tuple[torch.Tensor, ...]) -> torch.Tensor:
    return torch.cat([gradient.reshape(-1) for gradient in gradients])


def _set_policy_gradients(
    parameters: tuple[torch.nn.Parameter, ...],
    reward_loss: torch.Tensor,
    fisher_surrogate: torch.Tensor,
    beta: float,
    clip_ratio: float | None,
) -> dict[str, float | bool]:
    reward_gradients = torch.autograd.grad(
        reward_loss,
        parameters,
        retain_graph=True,
    )
    barrier_gradients = torch.autograd.grad(-fisher_surrogate, parameters)

    reward_norm = float(torch.linalg.vector_norm(_flatten(reward_gradients)))
    barrier_norm = float(
        torch.linalg.vector_norm(beta * _flatten(barrier_gradients))
    )
    scale = 1.0
    if clip_ratio is not None:
        scale = min(1.0, clip_ratio * reward_norm / barrier_norm)

    effective_beta = beta * scale
    applied_barrier_gradients = tuple(
        effective_beta * gradient for gradient in barrier_gradients
    )
    combined_gradients = tuple(
        reward_gradient + barrier_gradient
        for reward_gradient, barrier_gradient in zip(
            reward_gradients,
            applied_barrier_gradients,
            strict=True,
        )
    )
    for parameter, gradient in zip(parameters, combined_gradients, strict=True):
        parameter.grad = gradient

    applied_barrier_norm = barrier_norm * scale
    return {
        "effective_beta": effective_beta,
        "clip_scale": scale,
        "clipped": scale < 1.0,
        "reward_gradient_norm": reward_norm,
        "barrier_gradient_norm": barrier_norm,
        "applied_barrier_gradient_norm": applied_barrier_norm,
        "barrier_to_reward_ratio": applied_barrier_norm / reward_norm,
        "total_gradient_norm": float(
            torch.linalg.vector_norm(_flatten(combined_gradients))
        ),
    }


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True) + "\n")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def train(
    config: MountainCarContinuousConfig,
    output: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=False)
    _write_json(output / "config.json", config.to_dict())

    torch.set_num_threads(config.torch_threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    policy = _build_policy(config)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    reward_envs = gym.vector.AsyncVectorEnv(
        [_make_env(config) for _ in range(config.workers)]
    )
    fisher_envs = gym.vector.AsyncVectorEnv(
        [_make_env(config) for _ in range(config.workers)]
    )
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    training_path = output / "training.jsonl"
    rows = []
    started = time.perf_counter()

    try:
        for update in range(1, config.updates + 1):
            step_started = time.perf_counter()
            reward_trajectories = _collect(
                reward_envs,
                policy,
                config,
                update,
                0,
                config.reward_trajectories_per_worker,
            )
            fisher_trajectories = _collect(
                fisher_envs,
                policy,
                config,
                update,
                1,
                config.fisher_trajectories_per_worker,
            )

            reward_loss = compute_gpomdp_loss(
                policy,
                reward_trajectories,
                gamma=config.gamma,
                center_returns=True,
                normalize_returns=False,
                device="cpu",
            )
            fisher_surrogate, fisher = trajectory_fisher_logdet_surrogate(
                policy,
                reward_trajectories,
                mu=config.fisher_mu,
                fisher_trajectories=fisher_trajectories,
                score_backend="vmap",
                device="cpu",
            )

            optimizer.zero_grad()
            gradient = _set_policy_gradients(
                tuple(policy.parameters()),
                reward_loss,
                fisher_surrogate,
                config.fisher_beta,
                config.clip_ratio,
            )
            optimizer.step()

            returns = np.asarray(
                [sum(trajectory.rewards) for trajectory in reward_trajectories]
            )
            fisher_loss = -gradient["effective_beta"] * float(
                fisher_surrogate.detach()
            )
            row = {
                "update": update,
                "mean_return": float(returns.mean()),
                "reward_loss": float(reward_loss.detach()),
                "fisher_loss": fisher_loss,
                "total_loss": float(reward_loss.detach()) + fisher_loss,
                "beta": config.fisher_beta,
                "clip_ratio": config.clip_ratio,
                **gradient,
                "fisher_minimum_eigenvalue": fisher.minimum_eigenvalue,
                "fisher_maximum_eigenvalue": fisher.maximum_eigenvalue,
                "fisher_condition_number": (
                    fisher.maximum_eigenvalue / fisher.minimum_eigenvalue
                ),
                "fisher_effective_rank": fisher.effective_rank,
                "fisher_components_90": fisher.components_90,
                "fisher_numerical_rank": fisher.numerical_rank,
                "fisher_trace": fisher.trace,
                "fisher_logdet": fisher.logdet_margin,
                "elapsed_seconds": time.perf_counter() - step_started,
            }
            rows.append(row)
            _append_jsonl(training_path, row)

            if update % config.checkpoint_interval == 0 or update == config.updates:
                torch.save(
                    policy.state_dict(),
                    checkpoints / f"update_{update:04d}.pt",
                )
            if update == 1 or update % 10 == 0:
                print(
                    f"{update:03d}/{config.updates} "
                    f"return={row['mean_return']:.2f} "
                    f"ratio={row['barrier_to_reward_ratio']:.3f} "
                    f"lambda_min={row['fisher_minimum_eigenvalue']:.2e}",
                    flush=True,
                )
    finally:
        reward_envs.close()
        fisher_envs.close()

    result = {
        "config": config.to_dict(),
        "elapsed_seconds": time.perf_counter() - started,
        "final_return": rows[-1]["mean_return"],
        "last_10_mean_return": float(
            np.mean([row["mean_return"] for row in rows[-10:]])
        ),
    }
    torch.save(policy.state_dict(), output / "policy_final.pt")
    _write_json(output / "result.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--max-gradient-ratio", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    config = MountainCarContinuousConfig(
        seed=args.seed,
        fisher_beta=args.beta,
        learning_rate=args.learning_rate,
        clip_ratio=args.max_gradient_ratio,
    )
    print(json.dumps(train(config, args.output), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
