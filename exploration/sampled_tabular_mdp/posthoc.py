"""Exact post-hoc quantities for stored two-state handoff policies."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exploration.tabular_mdp.geometry import (
    barrier_gradients,
    enumerated_reduced_fisher,
    reduced_categorical_fisher,
)
from exploration.tabular_mdp.model import (
    DTYPE,
    TwoStepTrap,
    probabilities_from_reduced_logits,
    transition_pool_weights,
)


FISHER_NAMES = ("f0", "f1", "f_pool", "f_ref")
NUMERICAL_RANK_RELATIVE_THRESHOLD = 1e-12


@dataclass(frozen=True)
class StoredTrainingArchive:
    path: Path
    config: dict[str, Any]
    steps: np.ndarray
    phi: np.ndarray
    metrics: dict[str, np.ndarray]
    finite: np.ndarray
    sha256: str
    size: int
    mtime_ns: int

    def index_for_step(self, step: int) -> int:
        matches = np.flatnonzero(self.steps == step)
        if len(matches) != 1:
            raise ValueError(f"archive {self.path} does not store checkpoint {step}")
        return int(matches[0])


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def policy_hash(phi: np.ndarray) -> str:
    value = np.ascontiguousarray(phi, dtype="<f8")
    return hashlib.sha256(value.tobytes()).hexdigest()


def load_training_archive(path: str | Path) -> StoredTrainingArchive:
    source = Path(path)
    stat = source.stat()
    digest = sha256_file(source)
    with np.load(source, allow_pickle=False) as archive:
        config = json.loads(str(archive["config_json"].item()))
        metrics = {
            key.removeprefix("metric__"): archive[key].copy()
            for key in archive.files
            if key.startswith("metric__")
        }
        return StoredTrainingArchive(
            path=source,
            config=config,
            steps=archive["steps"].copy(),
            phi=archive["phi"].copy(),
            metrics=metrics,
            finite=archive["finite"].copy(),
            sha256=digest,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )


def fixed_reference_fisher(phi: torch.Tensor) -> torch.Tensor:
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    f0 = reduced_categorical_fisher(pi0)
    f1 = reduced_categorical_fisher(pi1)
    result = torch.zeros(phi.shape[:-1] + (4, 4), dtype=DTYPE)
    result[..., :2, :2] = 0.5 * f0
    result[..., 2:, 2:] = 0.5 * f1
    return result


def _block_fisher(phi: torch.Tensor) -> tuple[torch.Tensor, ...]:
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    mu0, mu1 = transition_pool_weights(phi)
    f0 = reduced_categorical_fisher(pi0)
    f1 = reduced_categorical_fisher(pi1)
    f_pool = torch.zeros(phi.shape[:-1] + (4, 4), dtype=DTYPE)
    f_pool[..., :2, :2] = mu0[..., None, None] * f0
    f_pool[..., 2:, 2:] = mu1[..., None, None] * f1
    return f0, f1, f_pool, fixed_reference_fisher(phi)


def _fisher_metrics(
    name: str,
    fisher: torch.Tensor,
    gradient: torch.Tensor,
) -> dict[str, torch.Tensor]:
    eigenvalues, eigenvectors = torch.linalg.eigh(fisher)
    eigenvalues = eigenvalues.flip(dims=(-1,))
    eigenvectors = eigenvectors.flip(dims=(-1,))
    projection = torch.einsum("...ji,...j->...i", eigenvectors, gradient)
    sign, logdet = torch.linalg.slogdet(fisher)
    threshold = NUMERICAL_RANK_RELATIVE_THRESHOLD * torch.maximum(
        torch.ones_like(eigenvalues[..., 0]), eigenvalues[..., 0]
    )
    result: dict[str, torch.Tensor] = {
        f"{name}_lambda_max": eigenvalues[..., 0],
        f"{name}_lambda_min": eigenvalues[..., -1],
        f"{name}_trace": eigenvalues.sum(dim=-1),
        f"{name}_slogdet_sign": sign,
        f"{name}_logdet": logdet,
        f"{name}_condition": eigenvalues[..., 0] / eigenvalues[..., -1],
        f"{name}_numerical_rank": (eigenvalues > threshold[..., None]).sum(dim=-1).to(DTYPE),
    }
    natural_energy = projection.square() / eigenvalues
    natural_total = natural_energy.sum(dim=-1)
    cumulative = torch.cumsum(natural_energy, dim=-1) / natural_total[..., None]
    for index in range(eigenvalues.shape[-1]):
        result[f"{name}_eigenvalue_{index + 1}"] = eigenvalues[..., index]
        result[f"{name}_reward_projection_{index + 1}"] = projection[..., index]
        result[f"{name}_reward_projection_sq_{index + 1}"] = projection[..., index].square()
        result[f"{name}_natural_energy_{index + 1}"] = natural_energy[..., index]
        result[f"{name}_natural_energy_cumulative_{index + 1}"] = cumulative[..., index]
    return result


def exact_policy_metrics(phi) -> dict[str, torch.Tensor]:
    """Compute exact behavioral, vector-field, and Fisher quantities.

    Reduced-coordinate convention: phi=(logit s0-a0, logit s0-a1,
    logit s1-a0, logit s1-a1), with action a2 fixed to zero at both states.
    The explore contrast is c=(-1,1,0,0).
    """
    value = torch.as_tensor(phi, dtype=DTYPE, device="cpu")
    if value.ndim == 1:
        value = value.unsqueeze(0)
    if value.ndim != 2 or value.shape[-1] != 4:
        raise ValueError("phi must have shape (4,) or (batch, 4)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("phi must be finite")

    mdp = TwoStepTrap()
    pi0, pi1 = probabilities_from_reduced_logits(value)
    mu0, mu1 = transition_pool_weights(value)
    q = pi0[:, 1]
    p_good = pi1[:, 0]
    v1 = pi1[:, 0] + 0.2 * pi1[:, 1]
    delta_safe = v1 - 0.5
    reward_gradient = mdp.exact_reward_gradient(value)
    reward_norm = torch.linalg.vector_norm(reward_gradient, dim=-1)
    conditional = barrier_gradients(value).detached_conditional
    beta_conditional = 0.2 * conditional
    barrier_norm = torch.linalg.vector_norm(beta_conditional, dim=-1)
    cosine_defined = (reward_norm > 0) & (barrier_norm > 0)
    cosine = torch.full_like(reward_norm, torch.nan)
    cosine[cosine_defined] = (
        (reward_gradient[cosine_defined] * conditional[cosine_defined]).sum(dim=-1)
        / (
            reward_norm[cosine_defined]
            * torch.linalg.vector_norm(conditional[cosine_defined], dim=-1)
        )
    )
    ratio = torch.full_like(reward_norm, torch.nan)
    ratio[barrier_norm > 0] = reward_norm[barrier_norm > 0] / barrier_norm[barrier_norm > 0]
    entropy0 = -(pi0 * torch.log(pi0)).sum(dim=-1)
    entropy1 = -(pi1 * torch.log(pi1)).sum(dim=-1)

    result: dict[str, torch.Tensor] = {
        **{f"phi_{index}": value[:, index] for index in range(4)},
        **{f"pi0_a{index}": pi0[:, index] for index in range(3)},
        **{f"pi1_a{index}": pi1[:, index] for index in range(3)},
        "q": q,
        "p_good": p_good,
        "v1": v1,
        "delta_safe": delta_safe,
        "q0_a0": torch.full_like(q, 0.5),
        "q0_a1": v1,
        "q0_a2": torch.zeros_like(q),
        "population_return": mdp.exact_return(value),
        "min_pi0": pi0.min(dim=-1).values,
        "min_pi1": pi1.min(dim=-1).values,
        "entropy0": entropy0,
        "entropy1": entropy1,
        "mu0": mu0,
        "mu1": mu1,
        "reward_gradient_norm": reward_norm,
        "explore_logit_reward_gradient": reward_gradient[:, 1],
        "d_explore_j": -reward_gradient[:, 0] + reward_gradient[:, 1],
        "beta_conditional_gradient_norm": barrier_norm,
        "reward_to_barrier_norm_ratio": ratio,
        "reward_barrier_cosine": cosine,
        "reward_barrier_cosine_defined": cosine_defined,
    }
    for index in range(4):
        result[f"reward_gradient_{index}"] = reward_gradient[:, index]
        result[f"conditional_gradient_{index}"] = conditional[:, index]

    f0, f1, f_pool, f_ref = _block_fisher(value)
    for name, fisher, gradient in (
        ("f0", f0, reward_gradient[:, :2]),
        ("f1", f1, reward_gradient[:, 2:]),
        ("f_pool", f_pool, reward_gradient),
        ("f_ref", f_ref, reward_gradient),
    ):
        result.update(_fisher_metrics(name, fisher, gradient))
    return result


def tensor_metrics_to_numpy(metrics: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
    return {key: value.detach().cpu().numpy() for key, value in metrics.items()}


def exact_reward_continuation(
    phi,
    *,
    start_update: int,
    final_update: int = 4000,
    alpha: float = 0.05,
    record_interval: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic explicit-Euler continuation under the exact reward field."""
    value = torch.as_tensor(phi, dtype=DTYPE, device="cpu").detach().clone()
    if value.ndim != 2 or value.shape[-1] != 4:
        raise ValueError("phi must have shape (batch, 4)")
    if not 0 <= start_update < final_update:
        raise ValueError("start_update must precede final_update")
    steps = [start_update]
    history = [value.detach().numpy().copy()]
    mdp = TwoStepTrap()
    for update in range(start_update + 1, final_update + 1):
        value = value + alpha * mdp.exact_reward_gradient(value)
        if update % record_interval == 0 or update == final_update:
            steps.append(update)
            history.append(value.detach().numpy().copy())
    return np.asarray(steps, dtype=np.int64), np.stack(history, axis=0)


def enumerated_fishers(phi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    return enumerated_reduced_fisher(pi0), enumerated_reduced_fisher(pi1)
