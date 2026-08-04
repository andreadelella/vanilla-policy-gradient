"""Deterministic exact-gradient training for the two-step trap."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .geometry import barrier_gradients, geometry_snapshot, gradient_for_method
from .model import DTYPE, TwoStepTrap, as_phi


class MethodName(str, Enum):
    REWARD_ONLY = "reward_only"
    DETACHED_CONDITIONAL = "detached_conditional"
    COMPLETE_WEIGHTED = "complete_weighted"
    UNIFORM_ACTION = "uniform_action"
    VISITATION_ONLY = "visitation_only"
    FULL_POOLED_FISHER = "full_pooled_fisher"


ALL_METHODS = tuple(method.value for method in MethodName)


@dataclass(frozen=True)
class ExactTrainingConfig:
    method: str
    alpha: float = 0.05
    beta: float = 0.1
    updates: int = 2000
    label: str = ""

    def __post_init__(self) -> None:
        if self.method not in ALL_METHODS:
            raise ValueError(f"unknown method: {self.method}")
        if not np.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not np.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if self.updates < 1:
            raise ValueError("updates must be positive")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExactTrainingResult:
    config: ExactTrainingConfig
    steps: np.ndarray
    phi: np.ndarray
    metrics: dict[str, np.ndarray]
    finite: np.ndarray

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp.npz")
        arrays: dict[str, Any] = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "config_json": np.asarray(json.dumps(self.config.to_dict(), sort_keys=True)),
            "steps": self.steps,
            "phi": self.phi,
            "finite": self.finite,
        }
        arrays.update({f"metric__{key}": value for key, value in self.metrics.items()})
        np.savez_compressed(temporary, **arrays)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path, expected: ExactTrainingConfig) -> "ExactTrainingResult":
        with np.load(path, allow_pickle=False) as archive:
            config = json.loads(str(archive["config_json"].item()))
            if config != expected.to_dict():
                raise ValueError(f"result has incompatible configuration: {path}")
            metrics = {
                key.removeprefix("metric__"): archive[key].copy()
                for key in archive.files
                if key.startswith("metric__")
            }
            return cls(
                expected,
                archive["steps"].copy(),
                archive["phi"].copy(),
                metrics,
                archive["finite"].copy(),
            )


def _norm(vector: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(vector, dim=-1)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    left_norm, right_norm = _norm(left), _norm(right)
    defined = (left_norm > 0) & (right_norm > 0)
    values = torch.full_like(left_norm, torch.nan)
    values[defined] = (left[defined] * right[defined]).sum(dim=-1) / (
        left_norm[defined] * right_norm[defined]
    )
    return values, defined


def _snapshot_metrics(
    phi: torch.Tensor,
    mdp: TwoStepTrap,
    method: str,
    beta: float,
) -> dict[str, torch.Tensor]:
    snapshot = dict(geometry_snapshot(phi).values)
    reward = mdp.exact_reward_gradient(phi)
    gradients = barrier_gradients(phi)
    regularizer = gradient_for_method(phi, method)
    scaled = beta * regularizer
    cosine, cosine_defined = _cosine(reward, regularizer)
    uniform_scaled = beta * gradients.uniform_action
    visit_scaled = beta * gradients.visitation_only
    cond_scaled = beta * gradients.detached_conditional
    state_scaled = beta * gradients.weighted_state_term
    cos_uniform, defined_uniform = _cosine(reward, gradients.uniform_action)
    cos_visit, defined_visit = _cosine(reward, gradients.visitation_only)
    cos_cond, defined_cond = _cosine(reward, gradients.detached_conditional)
    cos_state, defined_state = _cosine(reward, gradients.weighted_state_term)
    snapshot.update(
        {
            "return": mdp.exact_return(phi),
            "grad_reward_norm": _norm(reward),
            "grad_regularizer_scaled_norm": _norm(scaled),
            "cosine_reward_regularizer": cosine,
            "cosine_reward_regularizer_defined": cosine_defined,
            "grad_uniform_scaled_norm": _norm(uniform_scaled),
            "grad_visit_scaled_norm": _norm(visit_scaled),
            "cosine_reward_uniform": cos_uniform,
            "cosine_reward_uniform_defined": defined_uniform,
            "cosine_reward_visit": cos_visit,
            "cosine_reward_visit_defined": defined_visit,
            "grad_conditional_scaled_norm": _norm(cond_scaled),
            "grad_weight_state_scaled_norm": _norm(state_scaled),
            "cosine_reward_conditional": cos_cond,
            "cosine_reward_conditional_defined": defined_cond,
            "cosine_reward_weight_state": cos_state,
            "cosine_reward_weight_state_defined": defined_state,
        }
    )
    return snapshot


def train_exact(
    config: ExactTrainingConfig,
    initial_phi,
    *,
    mdp: TwoStepTrap | None = None,
) -> ExactTrainingResult:
    """Run explicit-Euler ascent using exact analytical gradients only."""
    mdp = mdp or TwoStepTrap()
    phi = as_phi(initial_phi)
    if phi.ndim == 1:
        phi = phi.unsqueeze(0)
    batch = phi.shape[0]
    first = _snapshot_metrics(phi, mdp, config.method, config.beta)
    metrics = {
        key: np.full((config.updates + 1, batch), np.nan, dtype=np.bool_ if value.dtype == torch.bool else np.float64)
        for key, value in first.items()
    }
    phi_history = np.full((config.updates + 1, batch, 4), np.nan, dtype=np.float64)
    finite = np.ones(batch, dtype=np.bool_)

    def record(index: int, values: dict[str, torch.Tensor]) -> None:
        phi_history[index] = phi.detach().numpy()
        for key, value in values.items():
            metrics[key][index] = value.detach().numpy()

    record(0, first)
    for step in range(1, config.updates + 1):
        reward_gradient = mdp.exact_reward_gradient(phi)
        regularizer_gradient = gradient_for_method(phi, config.method)
        candidate = phi + config.alpha * (reward_gradient + config.beta * regularizer_gradient)
        is_finite = torch.isfinite(candidate).all(dim=-1)
        finite &= is_finite.numpy()
        phi = torch.where(is_finite[:, None], candidate, torch.full_like(candidate, torch.nan))
        if bool(is_finite.any()):
            values = _snapshot_metrics(phi, mdp, config.method, config.beta)
            record(step, values)
        else:
            phi_history[step] = phi.numpy()
    return ExactTrainingResult(config, np.arange(config.updates + 1), phi_history, metrics, finite)


def magnitude_matched_betas(initial_phi, *, mdp: TwoStepTrap | None = None) -> dict[str, float]:
    """Match every regularizer's initial contribution to ||grad J||."""
    mdp = mdp or TwoStepTrap()
    phi = as_phi(initial_phi)
    target = float(_norm(mdp.exact_reward_gradient(phi)).item())
    result: dict[str, float] = {}
    for method in ALL_METHODS:
        if method == MethodName.REWARD_ONLY.value:
            continue
        norm = float(_norm(gradient_for_method(phi, method)).item())
        if norm == 0.0:
            raise ValueError(f"cannot magnitude-match zero gradient for {method}")
        result[method] = target / norm
    return result
