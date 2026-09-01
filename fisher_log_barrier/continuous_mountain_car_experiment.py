"""Single-seed Fisher log-barrier experiment on MountainCarContinuous-v0."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from fisher_log_barrier.loss1 import (
    FisherLogDetDomainError,
    trajectory_fisher_logdet_surrogate,
)
from vpg.data_collection import collect_parallel_trajectories
from vpg.gpomdp import compute_gpomdp_loss
from vpg.policy import build_policy
from vpg.train import make_env


@dataclass(frozen=True)
class ContinuousMountainCarConfig:
    seed: int = 101
    fisher_beta: float = 1.0 / 5000.0
    updates: int = 250
    env_id: str = "MountainCarContinuous-v0"
    workers: int = 16
    reward_trajectories_per_worker: int = 2
    fisher_trajectories_per_worker: int = 16
    fisher_seed_offset: int = 100_000
    horizon: int = 500
    action_repeat: int = 5
    hidden_sizes: tuple[int, ...] = (2, 2)
    learning_rate: float = 0.02
    gamma: float = 1.0
    fisher_mu: float = 0.0
    target_fisher_gradient_ratio: float | None = None
    score_backend: str = "vmap"
    checkpoint_interval: int = 40
    posthoc_fisher_trajectory_count: int = 256
    solve_return: float = 90.0
    torch_threads: int = 8

    @property
    def reward_trajectory_count(self) -> int:
        return self.workers * self.reward_trajectories_per_worker

    @property
    def training_fisher_trajectory_count(self) -> int:
        return self.workers * self.fisher_trajectories_per_worker

    def validate(self) -> None:
        if self.updates < 1:
            raise ValueError("updates must be positive")
        if min(
            self.workers,
            self.reward_trajectories_per_worker,
            self.fisher_trajectories_per_worker,
        ) < 1:
            raise ValueError("trajectory batch dimensions must be positive")
        if self.action_repeat < 1 or self.horizon < 1:
            raise ValueError("action_repeat and horizon must be positive")
        if not np.isfinite(self.fisher_beta) or self.fisher_beta < 0.0:
            raise ValueError("fisher_beta must be finite and non-negative")
        if self.target_fisher_gradient_ratio is not None and (
            not np.isfinite(self.target_fisher_gradient_ratio)
            or self.target_fisher_gradient_ratio < 0.0
        ):
            raise ValueError("target_fisher_gradient_ratio must be finite and non-negative")

    def to_dict(self) -> dict:
        result = asdict(self)
        result.update(
            {
                "schema_version": 2,
                "hidden_sizes": list(self.hidden_sizes),
                "optimizer": "Adam",
                "center_returns": True,
                "normalize_returns": False,
                "clip_actions": True,
                "init_log_std": -0.5,
                "learn_std": True,
                "policy": "mlp",
                "reward_trajectory_count": self.reward_trajectory_count,
                "training_fisher_trajectory_count": self.training_fisher_trajectory_count,
                "fisher_trajectory_count": self.posthoc_fisher_trajectory_count,
                "same_batch_fisher": False,
                "use_npg": False,
            }
        )
        return result


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, separators=(",", ":"), sort_keys=True) + "\n")


def _gradients(loss: torch.Tensor, parameters, *, retain_graph: bool):
    values = torch.autograd.grad(
        loss,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return tuple(
        torch.zeros_like(parameter) if gradient is None else gradient.detach()
        for parameter, gradient in zip(parameters, values, strict=True)
    )


def _gradient_norm(gradients) -> float:
    return float(torch.sqrt(sum(gradient.square().sum() for gradient in gradients)))


def gradient_balanced_beta(
    reward_gradient_norm: float,
    unscaled_fisher_gradient_norm: float,
    target_ratio: float,
) -> float:
    """Scale the Fisher gradient to a target fraction of the reward gradient."""
    if target_ratio < 0.0 or not np.isfinite(target_ratio):
        raise ValueError("target_ratio must be finite and non-negative")
    if reward_gradient_norm <= 0.0 or unscaled_fisher_gradient_norm <= 0.0:
        return 0.0
    return target_ratio * reward_gradient_norm / unscaled_fisher_gradient_norm


def _checkpoint(policy: torch.nn.Module, directory: Path, update: int) -> None:
    torch.save(policy.state_dict(), directory / f"update_{update:04d}.pt")


def run(config: ContinuousMountainCarConfig, output: Path) -> dict:
    config.validate()
    output.mkdir(parents=True, exist_ok=False)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir()
    _write_json(output / "config.json", config.to_dict())

    torch.set_num_threads(config.torch_threads)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    probe_env = gym.make(config.env_id)
    try:
        policy = build_policy(config.to_dict(), probe_env).to("cpu")
    finally:
        probe_env.close()
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                config.env_id,
                config.seed + worker,
                config.horizon,
                action_repeat=config.action_repeat,
            )
            for worker in range(config.workers)
        ]
    )
    fisher_envs = gym.vector.AsyncVectorEnv(
        [
            make_env(
                config.env_id,
                config.seed + config.fisher_seed_offset + worker,
                config.horizon,
                action_repeat=config.action_repeat,
            )
            for worker in range(config.workers)
        ]
    )

    diagnostics_path = output / "diagnostics.jsonl"
    parameters = tuple(policy.parameters())
    returns: list[float] = []
    completed_updates = 0
    started = time.perf_counter()
    status = "complete"
    failure: dict = {}
    try:
        for update in range(1, config.updates + 1):
            trajectories = collect_parallel_trajectories(
                envs,
                policy,
                n_trajectories_per_env=config.reward_trajectories_per_worker,
                clip_actions=True,
                device=torch.device("cpu"),
            )
            episode_returns = np.asarray(
                [sum(trajectory.rewards) for trajectory in trajectories],
                dtype=np.float64,
            )
            reward_loss = compute_gpomdp_loss(
                policy,
                trajectories,
                gamma=config.gamma,
                center_returns=True,
                normalize_returns=False,
                entropy_coeff=0.0,
                returns_implementation="recursive",
                device=torch.device("cpu"),
            )
            fisher_trajectories = collect_parallel_trajectories(
                fisher_envs,
                policy,
                n_trajectories_per_env=config.fisher_trajectories_per_worker,
                clip_actions=True,
                device=torch.device("cpu"),
            )
            fisher_surrogate, fisher_diagnostics = trajectory_fisher_logdet_surrogate(
                policy,
                trajectories,
                mu=config.fisher_mu,
                fisher_trajectories=fisher_trajectories,
                score_backend=config.score_backend,
                device=torch.device("cpu"),
            )
            reward_gradients = _gradients(reward_loss, parameters, retain_graph=True)
            unscaled_fisher_gradients = _gradients(
                -fisher_surrogate,
                parameters,
                retain_graph=False,
            )
            reward_gradient_norm = _gradient_norm(reward_gradients)
            unscaled_fisher_gradient_norm = _gradient_norm(unscaled_fisher_gradients)
            effective_beta = (
                config.fisher_beta
                if config.target_fisher_gradient_ratio is None
                else gradient_balanced_beta(
                    reward_gradient_norm,
                    unscaled_fisher_gradient_norm,
                    config.target_fisher_gradient_ratio,
                )
            )
            applied_fisher_gradients = tuple(
                effective_beta * gradient for gradient in unscaled_fisher_gradients
            )
            combined_gradients = tuple(
                reward_gradient + fisher_gradient
                for reward_gradient, fisher_gradient in zip(
                    reward_gradients,
                    applied_fisher_gradients,
                    strict=True,
                )
            )
            fisher_gradient_norm = _gradient_norm(applied_fisher_gradients)
            gradient_norm = _gradient_norm(combined_gradients)
            gradient_cosine = (
                float(
                    sum(
                        (reward_gradient * fisher_gradient).sum()
                        for reward_gradient, fisher_gradient in zip(
                            reward_gradients,
                            applied_fisher_gradients,
                            strict=True,
                        )
                    )
                    / (reward_gradient_norm * fisher_gradient_norm)
                )
                if reward_gradient_norm > 0.0 and fisher_gradient_norm > 0.0
                else None
            )
            fisher_loss = -effective_beta * fisher_surrogate
            total_loss = reward_loss + fisher_loss

            optimizer.zero_grad()
            for parameter, gradient in zip(parameters, combined_gradients, strict=True):
                parameter.grad = gradient
            optimizer.step()

            mean_return = float(episode_returns.mean())
            returns.append(mean_return)
            completed_updates = update
            row = {
                "update": update,
                "return": mean_return,
                "return_std": float(episode_returns.std()),
                "reward_loss": float(reward_loss.detach()),
                "fisher_surrogate": float(fisher_surrogate.detach()),
                "fisher_loss": float(fisher_loss.detach()),
                "total_loss": float(total_loss.detach()),
                "effective_fisher_beta": effective_beta,
                "target_fisher_gradient_ratio": config.target_fisher_gradient_ratio,
                "reward_gradient_norm": reward_gradient_norm,
                "unscaled_fisher_gradient_norm": unscaled_fisher_gradient_norm,
                "fisher_gradient_norm": fisher_gradient_norm,
                "achieved_fisher_gradient_ratio": (
                    fisher_gradient_norm / reward_gradient_norm
                    if reward_gradient_norm > 0.0
                    else None
                ),
                "gradient_cosine": gradient_cosine,
                "gradient_norm": gradient_norm,
                "parameter_count": fisher_diagnostics.parameter_count,
                "fisher_rank": fisher_diagnostics.rank,
                "lambda_min": fisher_diagnostics.minimum_eigenvalue,
                "lambda_max": fisher_diagnostics.maximum_eigenvalue,
                "logdet": fisher_diagnostics.logdet_margin,
                "elapsed_seconds": time.perf_counter() - started,
            }
            _append_jsonl(diagnostics_path, row)

            if update % config.checkpoint_interval == 0 or update == config.updates:
                _checkpoint(policy, checkpoint_dir, update)
            if update == 1 or update % 25 == 0:
                print(
                    f"update={update:03d} return={mean_return:.2f} "
                    f"beta={effective_beta:.3e} "
                    f"reward_grad={reward_gradient_norm:.3f} "
                    f"fisher_grad={fisher_gradient_norm:.3f} "
                    f"rank={fisher_diagnostics.rank}/{fisher_diagnostics.parameter_count}",
                    flush=True,
                )
    except FisherLogDetDomainError as error:
        status = "failed"
        failure = {
            "failure_update": completed_updates + 1,
            "failure_type": "strict_domain_failure",
            **error.to_dict(),
        }
    finally:
        envs.close()
        fisher_envs.close()

    elapsed = time.perf_counter() - started
    first_solved = next(
        (index for index, value in enumerate(returns, start=1) if value >= config.solve_return),
        None,
    )
    result = {
        "status": status,
        "config": config.to_dict(),
        "completed_updates": completed_updates,
        "elapsed_seconds": elapsed,
        "final_training_return": returns[-1] if returns else None,
        "last_10_mean_return": float(np.mean(returns[-10:])) if returns else None,
        "maximum_training_return": max(returns) if returns else None,
        "first_solved_update": first_solved,
        **failure,
    }
    _write_json(output / "result.json", result)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--beta", type=float, default=1.0 / 5000.0)
    parser.add_argument(
        "--target-gradient-ratio",
        type=float,
        help="adapt beta each update so the applied Fisher gradient has this norm ratio",
    )
    parser.add_argument("--updates", type=int, default=250)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "results/fisher_log_barrier/mountaincar_continuous/"
            "beta_1_over_5000/seed_101"
        ),
    )
    args = parser.parse_args(argv)
    result = run(
        ContinuousMountainCarConfig(
            seed=args.seed,
            fisher_beta=args.beta,
            target_fisher_gradient_ratio=args.target_gradient_ratio,
            updates=args.updates,
        ),
        args.output,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
