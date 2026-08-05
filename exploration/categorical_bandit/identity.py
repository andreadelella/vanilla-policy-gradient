"""Exact Fisher and log-barrier identities for a categorical softmax policy.

This module deliberately contains no rewards, sampling, optimizer, or RL state
distribution. Expectations over actions are evaluated by enumerating the full
finite action set.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch


DEFAULT_TOLERANCES: dict[str, float] = {
    "expected_score": 1e-12,
    "score_fisher": 1e-12,
    "bartlett": 1e-12,
    "symmetry": 1e-12,
    "null_residual": 1e-12,
    "logdet": 1e-10,
    "gradient": 1e-12,
    "gradient_sum": 1e-12,
    "hessian": 1e-11,
    "finite_difference": 1e-7,
    "minimum_eigenvalue": -1e-12,
    "rank_relative": 1e-12,
}


@dataclass(frozen=True)
class VerificationResult:
    """Numerical residuals for one exact categorical identity check."""

    name: str
    action_count: int
    reference_action: int
    minimum_probability: float
    expected_score_error: float
    score_fisher_error: float
    bartlett_error: float
    symmetry_error: float
    null_residual: float
    numerical_rank: int
    expected_rank: int
    rank_tolerance: float
    minimum_eigenvalue: float
    slogdet_sign: float
    logdet_error: float
    gradient_error: float
    gradient_sum_residual: float
    hessian_error: float
    finite_difference_error: float
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        return asdict(self)


def _as_float64_vector(values, *, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float64, device="cpu")
    if tensor.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    if tensor.numel() < 2:
        raise ValueError(f"{name} must contain at least two actions")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def _as_probabilities(probabilities) -> torch.Tensor:
    probabilities = _as_float64_vector(probabilities, name="probabilities")
    if not bool((probabilities > 0).all()):
        raise ValueError("probabilities must be strictly positive")
    total = probabilities.sum()
    if not bool(torch.isclose(total, torch.ones_like(total), rtol=1e-12, atol=1e-12)):
        raise ValueError("probabilities must sum to one")
    return probabilities


def _as_fisher(fisher) -> torch.Tensor:
    fisher = torch.as_tensor(fisher, dtype=torch.float64, device="cpu")
    if fisher.ndim != 2 or fisher.shape[0] != fisher.shape[1]:
        raise ValueError("fisher must be a square matrix")
    if fisher.shape[0] < 2:
        raise ValueError("fisher must describe at least two actions")
    if not bool(torch.isfinite(fisher).all()):
        raise ValueError("fisher must contain only finite values")
    return fisher


def _normalize_reference_action(reference_action: int, action_count: int) -> int:
    if not isinstance(reference_action, int):
        raise TypeError("reference_action must be an integer")
    normalized = reference_action
    if normalized < 0:
        normalized += action_count
    if normalized < 0 or normalized >= action_count:
        raise ValueError(
            f"reference_action must index one of {action_count} actions"
        )
    return normalized


def categorical_scores(probabilities) -> torch.Tensor:
    """Return all score rows ``e_a - pi`` for a categorical policy."""

    probabilities = _as_probabilities(probabilities)
    identity = torch.eye(
        probabilities.numel(),
        dtype=probabilities.dtype,
        device=probabilities.device,
    )
    return identity - probabilities.unsqueeze(0)


def fisher_from_score_expectation(probabilities) -> torch.Tensor:
    """Evaluate ``E_a[score(a) score(a)^T]`` by exact enumeration."""

    probabilities = _as_probabilities(probabilities)
    scores = categorical_scores(probabilities)
    return (probabilities.unsqueeze(1) * scores).T @ scores


def fisher_closed_form(probabilities) -> torch.Tensor:
    """Return ``diag(pi) - pi pi^T`` for a categorical policy."""

    probabilities = _as_probabilities(probabilities)
    return torch.diag(probabilities) - torch.outer(probabilities, probabilities)


def reduced_fisher(fisher, reference_action: int = -1) -> torch.Tensor:
    """Delete the reference action's row and column from the full Fisher.

    This is the Fisher in the reduced logit chart obtained by fixing the
    selected reference logit.
    """

    fisher = _as_fisher(fisher)
    reference_action = _normalize_reference_action(reference_action, fisher.shape[0])
    indices = torch.tensor(
        [index for index in range(fisher.shape[0]) if index != reference_action],
        dtype=torch.long,
        device=fisher.device,
    )
    return fisher.index_select(0, indices).index_select(1, indices)


def log_barrier(logits) -> torch.Tensor:
    """Return the stable categorical barrier ``sum_a log pi(a)``."""

    logits = _as_float64_vector(logits, name="logits")
    return torch.log_softmax(logits, dim=0).sum()


def analytic_barrier_gradient(logits) -> torch.Tensor:
    """Return the analytical barrier gradient ``1 - K pi``."""

    logits = _as_float64_vector(logits, name="logits")
    probabilities = torch.softmax(logits, dim=0)
    return torch.ones_like(probabilities) - probabilities.numel() * probabilities


def _maximum_absolute(tensor: torch.Tensor) -> float:
    return float(tensor.detach().abs().max().item())


def verify_identity_case(
    name: str,
    logits,
    reference_action: int = -1,
    *,
    tolerances: Mapping[str, float] | None = None,
    finite_difference_step: float = 1e-5,
    finite_difference_seed: int = 91,
) -> VerificationResult:
    """Verify every categorical identity for one deterministic logit vector."""

    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string")
    if finite_difference_step <= 0:
        raise ValueError("finite_difference_step must be positive")

    thresholds = dict(DEFAULT_TOLERANCES)
    if tolerances is not None:
        unknown = set(tolerances) - set(thresholds)
        if unknown:
            raise ValueError(f"unknown tolerance names: {sorted(unknown)}")
        thresholds.update(tolerances)

    base_logits = _as_float64_vector(logits, name="logits").detach().clone()
    action_count = base_logits.numel()
    reference_action = _normalize_reference_action(reference_action, action_count)

    work_logits = base_logits.clone().requires_grad_(True)
    log_probabilities = torch.log_softmax(work_logits, dim=0)
    probabilities = log_probabilities.exp()
    scores = categorical_scores(probabilities)

    expected_score = (probabilities.unsqueeze(1) * scores).sum(dim=0)
    score_fisher = fisher_from_score_expectation(probabilities)
    closed_fisher = fisher_closed_form(probabilities)

    # The expectation weights are held fixed at the current policy while the
    # log likelihood is differentiated, as required by Bartlett's identity.
    fixed_probabilities = probabilities.detach()
    negative_expected_log_likelihood_hessian = -torch.autograd.functional.hessian(
        lambda candidate: (
            fixed_probabilities * torch.log_softmax(candidate, dim=0)
        ).sum(),
        base_logits,
    )

    fisher_eigenvalues = torch.linalg.eigvalsh(closed_fisher.detach())
    largest_eigenvalue = float(fisher_eigenvalues[-1].item())
    rank_tolerance = thresholds["rank_relative"] * max(1.0, largest_eigenvalue)
    numerical_rank = int((fisher_eigenvalues > rank_tolerance).sum().item())

    reduced = reduced_fisher(closed_fisher, reference_action)
    slogdet_sign, reduced_logdet = torch.linalg.slogdet(reduced)
    target_logdet = log_probabilities.sum()

    barrier_gradient = torch.autograd.grad(target_logdet, work_logits)[0]
    expected_gradient = analytic_barrier_gradient(work_logits)
    barrier_hessian = torch.autograd.functional.hessian(log_barrier, base_logits)
    expected_hessian = -action_count * closed_fisher.detach()

    generator = torch.Generator(device="cpu").manual_seed(finite_difference_seed)
    direction = torch.randn(action_count, dtype=torch.float64, generator=generator)
    direction = direction / direction.norm()
    forward = log_barrier(base_logits + finite_difference_step * direction)
    backward = log_barrier(base_logits - finite_difference_step * direction)
    finite_difference = (forward - backward) / (2.0 * finite_difference_step)
    analytical_directional_derivative = (expected_gradient.detach() * direction).sum()

    expected_score_error = _maximum_absolute(expected_score)
    score_fisher_error = _maximum_absolute(score_fisher - closed_fisher)
    bartlett_error = _maximum_absolute(
        negative_expected_log_likelihood_hessian - closed_fisher.detach()
    )
    symmetry_error = _maximum_absolute(closed_fisher - closed_fisher.T)
    null_residual = _maximum_absolute(
        closed_fisher @ torch.ones(action_count, dtype=torch.float64)
    )
    logdet_error = float(
        (reduced_logdet.detach() - target_logdet.detach()).abs().item()
    )
    gradient_error = _maximum_absolute(barrier_gradient - expected_gradient)
    gradient_sum_residual = float(barrier_gradient.detach().sum().abs().item())
    hessian_error = _maximum_absolute(barrier_hessian - expected_hessian)
    finite_difference_error = float(
        (finite_difference - analytical_directional_derivative).abs().item()
    )
    minimum_eigenvalue = float(fisher_eigenvalues[0].item())
    sign_value = float(slogdet_sign.detach().item())

    passed = all(
        (
            expected_score_error <= thresholds["expected_score"],
            score_fisher_error <= thresholds["score_fisher"],
            bartlett_error <= thresholds["bartlett"],
            symmetry_error <= thresholds["symmetry"],
            null_residual <= thresholds["null_residual"],
            numerical_rank == action_count - 1,
            minimum_eigenvalue >= thresholds["minimum_eigenvalue"],
            sign_value == 1.0,
            logdet_error <= thresholds["logdet"],
            gradient_error <= thresholds["gradient"],
            gradient_sum_residual <= thresholds["gradient_sum"],
            hessian_error <= thresholds["hessian"],
            finite_difference_error <= thresholds["finite_difference"],
        )
    )

    return VerificationResult(
        name=name,
        action_count=action_count,
        reference_action=reference_action,
        minimum_probability=float(probabilities.detach().min().item()),
        expected_score_error=expected_score_error,
        score_fisher_error=score_fisher_error,
        bartlett_error=bartlett_error,
        symmetry_error=symmetry_error,
        null_residual=null_residual,
        numerical_rank=numerical_rank,
        expected_rank=action_count - 1,
        rank_tolerance=rank_tolerance,
        minimum_eigenvalue=minimum_eigenvalue,
        slogdet_sign=sign_value,
        logdet_error=logdet_error,
        gradient_error=gradient_error,
        gradient_sum_residual=gradient_sum_residual,
        hessian_error=hessian_error,
        finite_difference_error=finite_difference_error,
        passed=passed,
    )

