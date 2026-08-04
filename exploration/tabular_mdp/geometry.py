"""Exact reduced Fisher geometry and six Step 3 barrier directions."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .model import DTYPE, probabilities_from_reduced_logits, transition_pool_weights


def reduced_scores(probabilities: torch.Tensor) -> torch.Tensor:
    """Scores of all three actions in the two-coordinate reference chart."""
    p = torch.as_tensor(probabilities, dtype=DTYPE, device="cpu")
    if p.shape[-1] != 3:
        raise ValueError("probabilities must end in three actions")
    identity = torch.eye(3, 2, dtype=DTYPE)
    return identity - p[..., None, :2]


def reduced_categorical_fisher(probabilities: torch.Tensor) -> torch.Tensor:
    p = torch.as_tensor(probabilities, dtype=DTYPE, device="cpu")
    return torch.diag_embed(p[..., :2]) - p[..., :2, None] * p[..., None, :2]


def enumerated_reduced_fisher(probabilities: torch.Tensor) -> torch.Tensor:
    p = torch.as_tensor(probabilities, dtype=DTYPE, device="cpu")
    scores = reduced_scores(p)
    return torch.einsum("...a,...ai,...aj->...ij", p, scores, scores)


def pooled_fisher(phi) -> torch.Tensor:
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    mu0, mu1 = transition_pool_weights(phi)
    f0 = reduced_categorical_fisher(pi0)
    f1 = reduced_categorical_fisher(pi1)
    shape = f0.shape[:-2] + (4, 4)
    result = torch.zeros(shape, dtype=DTYPE)
    result[..., :2, :2] = mu0[..., None, None] * f0
    result[..., 2:, 2:] = mu1[..., None, None] * f1
    return result


@dataclass(frozen=True)
class BarrierTerms:
    b0: torch.Tensor
    b1: torch.Tensor
    weighted: torch.Tensor
    uniform: torch.Tensor
    visit: torch.Tensor
    full: torch.Tensor


def barrier_terms(phi) -> BarrierTerms:
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    mu0, mu1 = transition_pool_weights(phi)
    b0 = torch.log(pi0).sum(dim=-1)
    b1 = torch.log(pi1).sum(dim=-1)
    uniform = 0.5 * (b0 + b1)
    visit = torch.log(mu0) + torch.log(mu1)
    return BarrierTerms(b0, b1, mu0 * b0 + mu1 * b1, uniform, visit, uniform + visit)


@dataclass(frozen=True)
class BarrierGradients:
    detached_conditional: torch.Tensor
    weighted_state_term: torch.Tensor
    complete_weighted: torch.Tensor
    uniform_action: torch.Tensor
    visitation_only: torch.Tensor
    full_pooled_fisher: torch.Tensor


def barrier_gradients(phi) -> BarrierGradients:
    """Return analytical gradients for all nonzero regularizers."""
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    mu0, mu1 = transition_pool_weights(phi)
    terms = barrier_terms(phi)
    gb0 = 1.0 - 3.0 * pi0[..., :2]
    gb1 = 1.0 - 3.0 * pi1[..., :2]
    zeros = torch.zeros_like(gb0)
    grad_b0 = torch.cat((gb0, zeros), dim=-1)
    grad_b1 = torch.cat((zeros, gb1), dim=-1)
    conditional = mu0[..., None] * grad_b0 + mu1[..., None] * grad_b1

    q = pi0[..., 1]
    dq0 = -q * pi0[..., 0]
    dq1 = q * (1.0 - q)
    grad_q = torch.cat((torch.stack((dq0, dq1), dim=-1), zeros), dim=-1)
    scale = 1.0 / (1.0 + q).square()
    grad_mu0 = -scale[..., None] * grad_q
    grad_mu1 = scale[..., None] * grad_q
    state_term = terms.b0[..., None] * grad_mu0 + terms.b1[..., None] * grad_mu1
    complete = conditional + state_term

    uniform = 0.5 * (grad_b0 + grad_b1)
    visit_q_derivative = (1.0 - q) / (q * (1.0 + q))
    visit = visit_q_derivative[..., None] * grad_q
    return BarrierGradients(conditional, state_term, complete, uniform, visit, uniform + visit)


def gradient_for_method(phi, method: str) -> torch.Tensor:
    gradients = barrier_gradients(phi)
    if method == "reward_only":
        return torch.zeros_like(torch.as_tensor(phi, dtype=DTYPE, device="cpu"))
    mapping = {
        "detached_conditional": gradients.detached_conditional,
        "complete_weighted": gradients.complete_weighted,
        "uniform_action": gradients.uniform_action,
        "visitation_only": gradients.visitation_only,
        "full_pooled_fisher": gradients.full_pooled_fisher,
    }
    try:
        return mapping[method]
    except KeyError as error:
        raise ValueError(f"unknown method: {method}") from error


@dataclass(frozen=True)
class GeometrySnapshot:
    values: dict[str, torch.Tensor]


def geometry_snapshot(phi) -> GeometrySnapshot:
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    mu0, mu1 = transition_pool_weights(phi)
    f0 = reduced_categorical_fisher(pi0)
    f1 = reduced_categorical_fisher(pi1)
    fp = pooled_fisher(phi)
    terms = barrier_terms(phi)
    sign0, logdet0 = torch.linalg.slogdet(f0)
    sign1, logdet1 = torch.linalg.slogdet(f1)
    signp, logdetp = torch.linalg.slogdet(fp)
    values = {
        **{f"pi0_a{i}": pi0[..., i] for i in range(3)},
        **{f"pi1_a{i}": pi1[..., i] for i in range(3)},
        "q": pi0[..., 1], "mu0": mu0, "mu1": mu1,
        "min_pi0": pi0.min(dim=-1).values, "min_pi1": pi1.min(dim=-1).values,
        "lambda_min_f0": torch.linalg.eigvalsh(f0)[..., 0],
        "lambda_min_f1": torch.linalg.eigvalsh(f1)[..., 0],
        "lambda_min_f_pool": torch.linalg.eigvalsh(fp)[..., 0],
        "logdet_f0": logdet0, "logdet_f1": logdet1, "logdet_f_pool": logdetp,
        "slogdet_sign_f0": sign0, "slogdet_sign_f1": sign1, "slogdet_sign_f_pool": signp,
        "b0": terms.b0, "b1": terms.b1, "b_weighted": terms.weighted,
        "b_uniform": terms.uniform, "b_visit": terms.visit, "b_full": terms.full,
    }
    return GeometrySnapshot(values)
