"""Undamped action-enumerated Fisher diagnostics for discrete neural policies."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.distributions import Categorical
from torch.func import functional_call, grad as functional_grad, vmap

from vpg.gpomdp import compute_discounted_returns_matrix, trajectories_to_tensors


@dataclass(frozen=True)
class FisherMetrics:
    parameter_count: int
    state_count: int
    action_count: int
    score_row_count: int
    maximum_possible_rank: int
    numerical_rank: int
    eigenvalue_threshold: float
    trace: float
    largest_eigenvalue: float
    smallest_positive_eigenvalue: float
    k90: int
    participation_ratio: float
    entropy_effective_rank: float
    log_pseudodeterminant: float
    positive_spectrum_condition_number: float
    dtype: str
    spectral_floor_used: bool
    decomposition_space: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AlignmentMetrics:
    reward_gradient_norm: float
    captured_euclidean_energy_fraction: float
    leading_k90_euclidean_energy_fraction: float
    leading_k90_natural_energy_fraction: float
    natural_energy: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class FisherSpectrum:
    eigenvalues: torch.Tensor
    parameter_eigenvectors: torch.Tensor
    metrics: FisherMetrics


def state_bank_hash(states: np.ndarray | torch.Tensor) -> str:
    array = np.ascontiguousarray(np.asarray(states, dtype=np.float32))
    return hashlib.sha256(array.tobytes()).hexdigest()


def action_enumerated_score_matrix(
    policy: torch.nn.Module,
    states: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Construct rows ``sqrt(pi(a|s)) grad log pi(a|s)`` in float64."""

    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("states must have shape [M, state_dim]")
    model = policy.to(dtype=torch.float64)
    states = states.detach().cpu().to(torch.float64)
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    parameter_count = sum(parameter.numel() for parameter in params.values())
    with torch.no_grad():
        probabilities = torch.softmax(model(states), dim=-1)
    action_count = int(probabilities.shape[1])
    state_indices = torch.arange(states.shape[0]).repeat_interleave(action_count)
    actions = torch.arange(action_count).repeat(states.shape[0])
    row_states = states[state_indices]
    row_weights = probabilities.reshape(-1).sqrt()

    def log_probability(parameters, state, action):
        logits = functional_call(model, {**parameters, **buffers}, (state.unsqueeze(0),))
        return Categorical(logits=logits.squeeze(0)).log_prob(action)

    per_row = vmap(
        functional_grad(log_probability, argnums=0),
        in_dims=(None, 0, 0),
    )(params, row_states, actions)
    scores = torch.cat([
        gradient.reshape(row_states.shape[0], -1) for gradient in per_row.values()
    ], dim=1)
    scores = scores * row_weights.unsqueeze(1)
    if scores.shape[1] != parameter_count or not torch.isfinite(scores).all():
        raise AssertionError("invalid action-enumerated score matrix")
    return scores, action_count


def fisher_spectrum_from_scores(
    scores: torch.Tensor,
    *,
    state_count: int,
    action_count: int,
    threshold_relative: float = 1e-10,
) -> FisherSpectrum:
    """Compute the nonzero spectrum in the smaller of parameter or row space."""

    scores = scores.detach().cpu().to(torch.float64)
    row_count, parameter_count = scores.shape
    if row_count != state_count * action_count:
        raise ValueError("score rows do not match state/action counts")
    normalization = float(state_count)
    if parameter_count <= row_count:
        matrix = scores.T @ scores / normalization
        matrix = 0.5 * (matrix + matrix.T)
        eigenvalues_all, eigenvectors_all = torch.linalg.eigh(matrix)
        order = torch.argsort(eigenvalues_all, descending=True)
        eigenvalues_all = eigenvalues_all[order]
        eigenvectors_all = eigenvectors_all[:, order]
        decomposition_space = "parameter_fisher"
    else:
        gram = scores @ scores.T / normalization
        gram = 0.5 * (gram + gram.T)
        eigenvalues_all, row_vectors = torch.linalg.eigh(gram)
        order = torch.argsort(eigenvalues_all, descending=True)
        eigenvalues_all = eigenvalues_all[order]
        row_vectors = row_vectors[:, order]
        decomposition_space = "score_gram"
        eigenvectors_all = torch.zeros(
            (parameter_count, row_count), dtype=torch.float64
        )
        positive_for_recovery = eigenvalues_all > 0.0
        if positive_for_recovery.any():
            eigenvectors_all[:, positive_for_recovery] = (
                scores.T @ row_vectors[:, positive_for_recovery]
            ) / torch.sqrt(normalization * eigenvalues_all[positive_for_recovery]).unsqueeze(0)

    largest = max(0.0, float(eigenvalues_all[0])) if eigenvalues_all.numel() else 0.0
    threshold = threshold_relative * max(1.0, largest)
    positive = eigenvalues_all > threshold
    eigenvalues = eigenvalues_all[positive]
    eigenvectors = eigenvectors_all[:, positive]
    if eigenvalues.numel():
        trace = float(eigenvalues.sum())
        shares = eigenvalues / eigenvalues.sum()
        cumulative = torch.cumsum(shares, dim=0)
        k90 = int(torch.searchsorted(cumulative, torch.tensor(0.9, dtype=torch.float64)).item() + 1)
        participation = float(eigenvalues.sum().square() / eigenvalues.square().sum())
        entropy_rank = float(torch.exp(-(shares * torch.log(shares)).sum()))
        log_pseudodeterminant = float(torch.log(eigenvalues).sum())
        condition = float(eigenvalues[0] / eigenvalues[-1])
        smallest = float(eigenvalues[-1])
        largest_positive = float(eigenvalues[0])
    else:
        trace = largest_positive = smallest = 0.0
        k90 = 0
        participation = entropy_rank = 0.0
        log_pseudodeterminant = float("-inf")
        condition = float("inf")

    metrics = FisherMetrics(
        parameter_count=parameter_count,
        state_count=state_count,
        action_count=action_count,
        score_row_count=row_count,
        maximum_possible_rank=min(parameter_count, row_count),
        numerical_rank=int(eigenvalues.numel()),
        eigenvalue_threshold=threshold,
        trace=trace,
        largest_eigenvalue=largest_positive,
        smallest_positive_eigenvalue=smallest,
        k90=k90,
        participation_ratio=participation,
        entropy_effective_rank=entropy_rank,
        log_pseudodeterminant=log_pseudodeterminant,
        positive_spectrum_condition_number=condition,
        dtype="float64",
        spectral_floor_used=False,
        decomposition_space=decomposition_space,
    )
    return FisherSpectrum(eigenvalues, eigenvectors, metrics)


def action_enumerated_fisher_spectrum(
    policy: torch.nn.Module,
    states: torch.Tensor,
    *,
    threshold_relative: float = 1e-10,
) -> FisherSpectrum:
    scores, action_count = action_enumerated_score_matrix(policy, states)
    return fisher_spectrum_from_scores(
        scores,
        state_count=int(states.shape[0]),
        action_count=action_count,
        threshold_relative=threshold_relative,
    )


def analysis_reward_gradient(policy: torch.nn.Module, trajectories, gamma: float) -> torch.Tensor:
    """Raw, uncentered trajectory-mean GPOMDP gradient in float64."""

    model = policy.to(dtype=torch.float64)
    states, actions, rewards, mask = trajectories_to_tensors(trajectories, device="cpu")
    states = states.to(torch.float64)
    rewards = rewards.to(torch.float64)
    returns = compute_discounted_returns_matrix(rewards, gamma)
    n_trajectories, max_length = rewards.shape
    flat_states = states.reshape(n_trajectories * max_length, -1)
    flat_actions = actions.reshape(n_trajectories * max_length).long()
    log_probabilities = model.log_prob(flat_states, flat_actions).reshape(n_trajectories, max_length)
    objective = (returns * log_probabilities * mask.to(torch.float64)).sum(dim=1).mean()
    gradients = torch.autograd.grad(objective, tuple(model.parameters()))
    result = torch.cat([gradient.reshape(-1) for gradient in gradients]).detach()
    if not torch.isfinite(result).all():
        raise FloatingPointError("analysis reward gradient is non-finite")
    return result


def reward_gradient_alignment(
    spectrum: FisherSpectrum,
    reward_gradient: torch.Tensor,
) -> tuple[AlignmentMetrics, dict[str, np.ndarray]]:
    gradient = reward_gradient.detach().cpu().to(torch.float64)
    coordinates = spectrum.parameter_eigenvectors.T @ gradient
    euclidean = coordinates.square()
    natural = euclidean / spectrum.eigenvalues
    total_gradient_energy = float(gradient.square().sum())
    captured = float(euclidean.sum()) / total_gradient_energy if total_gradient_energy > 0.0 else float("nan")
    k90 = spectrum.metrics.k90
    euclidean_total = float(euclidean.sum())
    natural_total = float(natural.sum())
    leading_euclidean = float(euclidean[:k90].sum()) / euclidean_total if euclidean_total > 0.0 else float("nan")
    leading_natural = float(natural[:k90].sum()) / natural_total if natural_total > 0.0 else float("nan")
    metrics = AlignmentMetrics(
        reward_gradient_norm=float(gradient.norm()),
        captured_euclidean_energy_fraction=captured,
        leading_k90_euclidean_energy_fraction=leading_euclidean,
        leading_k90_natural_energy_fraction=leading_natural,
        natural_energy=natural_total,
    )
    arrays = {
        "eigenvalues": spectrum.eigenvalues.numpy(),
        "gradient_coordinates": coordinates.numpy(),
        "euclidean_energy": euclidean.numpy(),
        "natural_energy": natural.numpy(),
        "cumulative_euclidean_fraction": (torch.cumsum(euclidean, 0) / euclidean.sum()).numpy() if euclidean_total > 0.0 else np.full(euclidean.shape, np.nan),
        "cumulative_natural_fraction": (torch.cumsum(natural, 0) / natural.sum()).numpy() if natural_total > 0.0 else np.full(natural.shape, np.nan),
    }
    return metrics, arrays


def save_state_bank(path: str | Path, states: np.ndarray, configuration: dict) -> dict:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"state bank already exists: {path}")
    states = np.ascontiguousarray(states, dtype=np.float32)
    digest = state_bank_hash(states)
    np.savez_compressed(path, states=states)
    metadata = {
        **configuration,
        "state_count": int(states.shape[0]),
        "state_dimension": int(states.shape[1]),
        "sha256": digest,
        "normalization": "raw Gymnasium observations; no state normalization",
    }
    path.with_suffix(".json").write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
    return metadata
