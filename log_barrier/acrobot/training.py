"""Acrobot GPOMDP training with a fixed categorical log barrier."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch

from vpg.data_collection import collect_parallel_trajectories
from vpg.gpomdp import compute_gpomdp_loss, trajectories_to_tensors
from vpg.policy import build_policy
from vpg.train import make_env

from .barrier import categorical_log_barrier
from .fisher import empirical_policy_fisher_spectrum


METHODS = ("reward_only", "log_barrier")


@dataclass(frozen=True)
class AcrobotConfig:
    method: str
    seed: int
    env_id: str = "Acrobot-v1"
    hidden_sizes: tuple[int, ...] = (8, 8)
    learning_rate: float = 0.003
    gamma: float = 0.99
    updates: int = 1000
    episodes_per_update: int = 8
    horizon: int = 500
    center_returns: bool = True
    normalize_returns: bool = False
    beta: float = 546.4135158976487
    device: str = "cpu"

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method: {self.method}")
        if self.updates < 1 or self.episodes_per_update < 1:
            raise ValueError("updates and episodes_per_update must be positive")
        if self.learning_rate <= 0.0 or self.beta < 0.0:
            raise ValueError("learning_rate must be positive and beta non-negative")

    @property
    def run_id(self) -> str:
        return f"seed_{self.seed}__{self.method}"

    def to_dict(self) -> dict:
        result = asdict(self)
        result["hidden_sizes"] = list(self.hidden_sizes)
        result["optimizer"] = "Adam"
        result["collector"] = "complete episodes per update"
        return result


def checkpoint_updates(updates: int) -> tuple[int, ...]:
    return tuple(sorted({int(round(updates * index / 10.0)) for index in range(11)}))


def regularization_coefficient(config: AcrobotConfig) -> float:
    """Return the fixed coefficient used by the selected method."""

    return 0.0 if config.method == "reward_only" else config.beta


def _valid_states(trajectories, device: torch.device) -> torch.Tensor:
    states, _, _, mask = trajectories_to_tensors(trajectories, device=device)
    return states.reshape(-1, states.shape[-1])[mask.reshape(-1).bool()].detach()


def _training_row(config: AcrobotConfig, update: int, trajectories, beta: float, barrier) -> dict:
    returns = np.asarray([sum(trajectory.rewards) for trajectory in trajectories], dtype=np.float64)
    lengths = np.asarray([len(trajectory.rewards) for trajectory in trajectories], dtype=np.float64)
    row = {
        "run_id": config.run_id,
        "seed": config.seed,
        "method": config.method,
        "update": update,
        "environment_steps": int(lengths.sum()),
        "mean_batch_return": float(returns.mean()),
        "mean_episode_length": float(lengths.mean()),
        "active_beta": beta,
    }
    if barrier is not None:
        row.update(barrier.to_dict())
    return row


def _spectrum_row(config: AcrobotConfig, update: int, policy, states: torch.Tensor) -> tuple[dict, torch.Tensor]:
    spectrum = empirical_policy_fisher_spectrum(policy, states)
    row = {
        "run_id": config.run_id,
        "seed": config.seed,
        "method": config.method,
        "update": update,
        **spectrum.metrics.to_dict(),
    }
    row.update({f"eigenvalue_{index + 1}": float(value) for index, value in enumerate(spectrum.eigenvalues)})
    return row, spectrum.eigenvalues


def train(config: AcrobotConfig, checkpoint_dir=None) -> tuple[list[dict], list[dict]]:
    """Train one seed. No code or defaults in :mod:`vpg` are modified."""

    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    probe_env = gym.make(config.env_id)
    policy = build_policy({"hidden_sizes": config.hidden_sizes, "policy": "mlp"}, probe_env).to(device)
    probe_env.close()
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    envs = gym.vector.AsyncVectorEnv([
        make_env(config.env_id, config.seed + worker, config.horizon)
        for worker in range(config.episodes_per_update)
    ])
    checkpoints = set(checkpoint_updates(config.updates))
    training_rows: list[dict] = []
    spectrum_rows: list[dict] = []
    cumulative_steps = 0

    try:
        for update in range(config.updates + 1):
            trajectories = collect_parallel_trajectories(envs, policy, 1, device=device)
            states = _valid_states(trajectories, device)
            if update in checkpoints:
                spectrum_row, _ = _spectrum_row(config, update, policy, states)
                spectrum_row["environment_steps"] = cumulative_steps
                spectrum_rows.append(spectrum_row)
                if checkpoint_dir is not None:
                    checkpoint_dir.mkdir(parents=True, exist_ok=True)
                    torch.save(policy.state_dict(), checkpoint_dir / f"update_{update:04d}.pt")
            if update == config.updates:
                break

            beta = regularization_coefficient(config)
            reward_loss = compute_gpomdp_loss(
                policy,
                trajectories,
                gamma=config.gamma,
                center_returns=config.center_returns,
                normalize_returns=config.normalize_returns,
                device=device,
            )
            barrier_diagnostics = None
            loss = reward_loss
            if beta > 0.0:
                barrier, barrier_diagnostics = categorical_log_barrier(policy(states))
                loss = loss - beta * barrier
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cumulative_steps += sum(len(trajectory.rewards) for trajectory in trajectories)
            row = _training_row(config, update + 1, trajectories, beta, barrier_diagnostics)
            row["environment_steps"] = cumulative_steps
            training_rows.append(row)
    finally:
        envs.close()
    return training_rows, spectrum_rows
