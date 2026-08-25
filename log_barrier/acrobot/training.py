"""Acrobot GPOMDP training with distinct action and Fisher log barriers."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch

from fisher_log_barrier import (
    SCORE_BACKENDS,
    ReferenceMLPSoftmaxPolicy,
    trajectory_fisher_logdet_surrogate,
)
from vpg.data_collection import collect_parallel_trajectories
from vpg.gpomdp import compute_gpomdp_loss, trajectories_to_tensors
from vpg.policy import build_policy
from vpg.train import make_env

from .barrier import categorical_log_barrier
from .fisher import empirical_policy_fisher_spectrum


METHODS = ("reward_only", "log_barrier", "fisher_logdet")
POLICY_PARAMETERIZATIONS = ("auto", "standard", "reference")


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
    fisher_episodes_per_update: int = 256
    fisher_parallel_envs: int = 16
    horizon: int = 500
    center_returns: bool = True
    normalize_returns: bool = False
    beta: float = 546.4135158976487
    fisher_mu: float = 1e-10
    fisher_beta: float = 1.0
    fisher_score_backend: str = "vmap"
    policy_parameterization: str = "auto"
    device: str = "cpu"

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method: {self.method}")
        if self.updates < 1 or self.episodes_per_update < 1:
            raise ValueError("updates and episodes_per_update must be positive")
        if self.fisher_episodes_per_update < 1 or self.fisher_parallel_envs < 1:
            raise ValueError("Fisher trajectory and worker counts must be positive")
        if self.fisher_episodes_per_update % self.fisher_parallel_envs != 0:
            raise ValueError("fisher_episodes_per_update must be divisible by fisher_parallel_envs")
        if self.learning_rate <= 0.0 or self.beta < 0.0 or self.fisher_beta < 0.0:
            raise ValueError(
                "learning_rate must be positive and barrier coefficients non-negative"
            )
        if not np.isfinite(self.fisher_mu) or self.fisher_mu < 0.0:
            raise ValueError("fisher_mu must be finite and non-negative")
        if self.fisher_score_backend not in SCORE_BACKENDS:
            raise ValueError(f"fisher_score_backend must be one of {SCORE_BACKENDS}")
        if self.policy_parameterization not in POLICY_PARAMETERIZATIONS:
            raise ValueError(
                "policy_parameterization must be auto, standard, or reference"
            )
        if self.method == "fisher_logdet" and self.effective_policy_parameterization != "reference":
            raise ValueError(
                "fisher_logdet requires the identifiable reference-logit policy"
            )

    @property
    def effective_policy_parameterization(self) -> str:
        if self.policy_parameterization == "auto":
            return "reference" if self.method == "fisher_logdet" else "standard"
        return self.policy_parameterization

    @property
    def run_id(self) -> str:
        suffix = (
            "__reference_policy"
            if self.method != "fisher_logdet"
            and self.effective_policy_parameterization == "reference"
            else ""
        )
        return f"seed_{self.seed}__{self.method}{suffix}"

    def to_dict(self) -> dict:
        result = asdict(self)
        result["hidden_sizes"] = list(self.hidden_sizes)
        result["optimizer"] = "Adam"
        result["collector"] = "complete episodes per update"
        result["effective_policy_parameterization"] = self.effective_policy_parameterization
        return result


def checkpoint_updates(updates: int) -> tuple[int, ...]:
    return tuple(sorted({int(round(updates * index / 10.0)) for index in range(11)}))


def regularization_coefficient(config: AcrobotConfig) -> float:
    """Return only the existing categorical action-barrier coefficient."""

    return config.beta if config.method == "log_barrier" else 0.0


def _valid_states(trajectories, device: torch.device) -> torch.Tensor:
    states, _, _, mask = trajectories_to_tensors(trajectories, device=device)
    return states.reshape(-1, states.shape[-1])[mask.reshape(-1).bool()].detach()


def _build_policy(config: AcrobotConfig, env) -> torch.nn.Module:
    if config.effective_policy_parameterization == "reference":
        if not isinstance(env.action_space, gym.spaces.Discrete):
            raise ValueError("reference-logit policies require a discrete action space")
        return ReferenceMLPSoftmaxPolicy(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.n,
            hidden_sizes=config.hidden_sizes,
        )
    return build_policy(
        {"hidden_sizes": config.hidden_sizes, "policy": "mlp"},
        env,
    )


def _training_row(
    config: AcrobotConfig,
    update: int,
    trajectories,
    beta: float,
    barrier,
    fisher_beta: float,
    fisher_diagnostics,
) -> dict:
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
        "active_fisher_beta": fisher_beta,
    }
    if barrier is not None:
        row.update(barrier.to_dict())
    if fisher_diagnostics is not None:
        row.update(
            {
                f"fisher_logdet_{key}": value
                for key, value in fisher_diagnostics.to_dict().items()
            }
        )
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
    policy = _build_policy(config, probe_env).to(device)
    probe_env.close()
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    envs = gym.vector.AsyncVectorEnv([
        make_env(config.env_id, config.seed + worker, config.horizon)
        for worker in range(config.episodes_per_update)
    ])
    fisher_envs = None
    if config.method == "fisher_logdet" and config.fisher_beta > 0.0:
        fisher_envs = gym.vector.SyncVectorEnv([
            make_env(config.env_id, config.seed + 100_000 + worker, config.horizon)
            for worker in range(config.fisher_parallel_envs)
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
            fisher_beta = config.fisher_beta if config.method == "fisher_logdet" else 0.0
            reward_loss = compute_gpomdp_loss(
                policy,
                trajectories,
                gamma=config.gamma,
                center_returns=config.center_returns,
                normalize_returns=config.normalize_returns,
                device=device,
            )
            barrier_diagnostics = None
            fisher_diagnostics = None
            fisher_trajectories = []
            loss = reward_loss
            if beta > 0.0:
                barrier, barrier_diagnostics = categorical_log_barrier(policy(states))
                loss = loss - beta * barrier
            elif fisher_beta > 0.0:
                fisher_trajectories = collect_parallel_trajectories(
                    fisher_envs,
                    policy,
                    config.fisher_episodes_per_update // config.fisher_parallel_envs,
                    device=device,
                )
                fisher_surrogate, fisher_diagnostics = trajectory_fisher_logdet_surrogate(
                    policy,
                    trajectories,
                    mu=config.fisher_mu,
                    fisher_trajectories=fisher_trajectories,
                    score_backend=config.fisher_score_backend,
                    device=device,
                )
                loss = loss - fisher_beta * fisher_surrogate
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            gradient_steps = sum(len(trajectory.rewards) for trajectory in trajectories)
            fisher_steps = (
                sum(len(trajectory.rewards) for trajectory in fisher_trajectories)
                if fisher_beta > 0.0
                else 0
            )
            cumulative_steps += gradient_steps + fisher_steps
            row = _training_row(
                config,
                update + 1,
                trajectories,
                beta,
                barrier_diagnostics,
                fisher_beta,
                fisher_diagnostics,
            )
            row["reward_loss"] = float(reward_loss.detach().cpu())
            row["total_loss"] = float(loss.detach().cpu())
            row["gradient_environment_steps"] = gradient_steps
            row["fisher_environment_steps"] = fisher_steps
            row["environment_steps"] = cumulative_steps
            training_rows.append(row)
    finally:
        envs.close()
        if fisher_envs is not None:
            fisher_envs.close()
    return training_rows, spectrum_rows
