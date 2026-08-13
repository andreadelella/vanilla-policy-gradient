"""Deterministic exact-gradient training for the finite MDP experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .geometry import barrier_gradient, fisher_metrics, joint_fisher, policy_fisher
from .model import DTYPE, ThreeStateChain, chain_probabilities, discounted_occupancy


METHODS = ("reward_only", "policy_fisher_logdet", "joint_fisher_logdet")
INITIALIZATIONS = {
    "uniform": (0.0,) * 6,
    "adverse": (2.0, -2.0, 2.0, -2.0, -2.0, 2.0),
}


@dataclass(frozen=True)
class ExactTrainingConfig:
    method: str
    initialization: str = "adverse"
    alpha: float = 0.05
    beta: float = 0.1
    updates: int = 2000
    gamma: float = 0.99

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method: {self.method}")
        if self.initialization not in INITIALIZATIONS:
            raise ValueError(f"unknown initialization: {self.initialization}")
        if self.alpha <= 0.0 or self.updates < 1:
            raise ValueError("alpha and updates must be positive")
        if self.beta < 0.0:
            raise ValueError("beta must be non-negative")
        if not 0.0 < self.gamma < 1.0:
            raise ValueError("gamma must lie strictly between zero and one")

    @property
    def effective_beta(self) -> float:
        return 0.0 if self.method == "reward_only" else self.beta

    @property
    def run_id(self) -> str:
        beta = str(self.effective_beta).replace(".", "p")
        return f"{self.initialization}__{self.method}__beta_{beta}"


@dataclass(frozen=True)
class ExactTrainingResult:
    config: ExactTrainingConfig
    trajectory: tuple[dict, ...]
    spectra: tuple[dict, ...]


def checkpoint_updates(updates: int) -> tuple[int, ...]:
    return tuple(sorted({int(round(updates * fraction / 10.0)) for fraction in range(11)}))


def _regularizer_gradient(phi: torch.Tensor, config: ExactTrainingConfig) -> torch.Tensor:
    if config.method == "reward_only":
        return torch.zeros_like(phi)
    fisher_name = "policy" if config.method == "policy_fisher_logdet" else "joint"
    return barrier_gradient(phi, fisher_name, config.gamma)


def _behavior_row(
    phi: torch.Tensor,
    config: ExactTrainingConfig,
    update: int,
    reward_gradient: torch.Tensor,
    regularizer_gradient: torch.Tensor,
) -> dict:
    mdp = ThreeStateChain()
    pi = chain_probabilities(phi)
    occupancy = discounted_occupancy(phi, config.gamma)
    value, v1, v2 = mdp.values(phi)
    applied = config.effective_beta * regularizer_gradient
    total = reward_gradient + applied
    return {
        "run_id": config.run_id,
        "method": config.method,
        "initialization": config.initialization,
        "update": update,
        "return": float(value),
        "V1": float(v1),
        "V2": float(v2),
        "q0": float(pi[0, 1]),
        "q1": float(pi[1, 1]),
        "p_good_s2": float(pi[2, 0]),
        **{f"d{state}": float(occupancy[state]) for state in range(4)},
        **{f"pi{state}_a{action}": float(pi[state, action]) for state in range(3) for action in range(3)},
        **{f"phi_{index}": float(phi[index]) for index in range(6)},
        "reward_gradient_norm": float(torch.linalg.vector_norm(reward_gradient)),
        "barrier_gradient_norm": float(torch.linalg.vector_norm(regularizer_gradient)),
        "applied_barrier_gradient_norm": float(torch.linalg.vector_norm(applied)),
        "total_gradient_norm": float(torch.linalg.vector_norm(total)),
        "finite": bool(torch.isfinite(total).all()),
    }


def _spectrum_rows(phi: torch.Tensor, config: ExactTrainingConfig, update: int) -> tuple[dict, dict]:
    rows = []
    for name, matrix in (
        ("policy", policy_fisher(phi, config.gamma)),
        ("joint", joint_fisher(phi, config.gamma)),
    ):
        metrics = fisher_metrics(matrix)
        rows.append(
            {
                "run_id": config.run_id,
                "method": config.method,
                "initialization": config.initialization,
                "update": update,
                "fisher": name,
                **metrics.to_dict(),
            }
        )
    return rows[0], rows[1]


def train(config: ExactTrainingConfig) -> ExactTrainingResult:
    """Run exact Euclidean gradient ascent and retain every behavioral update."""

    config.validate()
    phi = torch.tensor(INITIALIZATIONS[config.initialization], dtype=DTYPE)
    mdp = ThreeStateChain()
    checkpoints = set(checkpoint_updates(config.updates))
    trajectory: list[dict] = []
    spectra: list[dict] = []

    for update in range(config.updates + 1):
        reward_gradient = mdp.reward_gradient(phi)
        regularizer_gradient = _regularizer_gradient(phi, config)
        row = _behavior_row(phi, config, update, reward_gradient, regularizer_gradient)
        trajectory.append(row)
        if update in checkpoints:
            spectra.extend(_spectrum_rows(phi, config, update))
        if update == config.updates:
            break
        total = reward_gradient + config.effective_beta * regularizer_gradient
        phi = phi + config.alpha * total
        if not bool(torch.isfinite(phi).all()):
            raise FloatingPointError(f"non-finite parameters in {config.run_id}")

    return ExactTrainingResult(config, tuple(trajectory), tuple(spectra))


def config_dict(config: ExactTrainingConfig) -> dict:
    return asdict(config)

