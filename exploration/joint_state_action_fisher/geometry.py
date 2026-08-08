"""Closed-form and enumerated joint state-action Fisher geometry."""

from __future__ import annotations

import torch

from exploration.tabular_mdp.geometry import pooled_fisher as legacy_pooled_policy_fisher
from exploration.tabular_mdp.geometry import reduced_categorical_fisher
from exploration.tabular_mdp.model import (
    DTYPE,
    probabilities_from_reduced_logits,
    transition_pool_weights,
)

from .definitions import (
    joint_score,
    joint_state_action_probabilities,
    policy_score,
    q_gradient,
    state_distribution_score,
)


def _single_phi(phi) -> torch.Tensor:
    value = torch.as_tensor(phi, dtype=DTYPE, device="cpu")
    if value.ndim not in (1, 2) or value.shape[-1] != 4:
        raise ValueError("phi must have shape (4,) or (batch, 4)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("phi must contain only finite values")
    if value.ndim != 1:
        raise ValueError("this exact Fisher operation requires phi with shape (4,)")
    return value


def pooled_policy_fisher_closed_form(phi) -> torch.Tensor:
    """State-weighted conditional-policy Fisher F_{pi,mu}."""
    value = _single_phi(phi)
    pi0, pi1 = probabilities_from_reduced_logits(value)
    mu0, mu1 = transition_pool_weights(value)
    return torch.block_diag(
        mu0 * reduced_categorical_fisher(pi0),
        mu1 * reduced_categorical_fisher(pi1),
    )


def pooled_policy_fisher_enumerated(phi) -> torch.Tensor:
    """Direct six-outcome enumeration of E_rho[g_pi g_pi^T]."""
    value = _single_phi(phi)
    rho = joint_state_action_probabilities(value)
    result = torch.zeros((4, 4), dtype=DTYPE)
    for state in range(2):
        for action in range(3):
            score = policy_score(value, state, action)
            result += rho[state, action] * torch.outer(score, score)
    return result


def state_distribution_fisher_closed_form(phi) -> torch.Tensor:
    """Rank-one transition-pooled state-distribution Fisher F_mu."""
    value = _single_phi(phi)
    pi0, _ = probabilities_from_reduced_logits(value)
    q = pi0[1]
    v = q_gradient(value)
    return torch.outer(v, v) / (q * (1.0 + q).square())


def state_distribution_fisher_enumerated(phi) -> torch.Tensor:
    """Direct two-state enumeration of E_mu[g_mu g_mu^T]."""
    value = _single_phi(phi)
    mu = torch.stack(transition_pool_weights(value))
    result = torch.zeros((4, 4), dtype=DTYPE)
    for state in range(2):
        score = state_distribution_score(value, state)
        result += mu[state] * torch.outer(score, score)
    return result


def joint_state_action_fisher_enumerated(phi) -> torch.Tensor:
    """Reference implementation: enumerate E_rho[g_rho g_rho^T]."""
    value = _single_phi(phi)
    rho = joint_state_action_probabilities(value)
    result = torch.zeros((4, 4), dtype=DTYPE)
    for state in range(2):
        for action in range(3):
            score = joint_score(value, state, action)
            result += rho[state, action] * torch.outer(score, score)
    return result


def joint_state_action_fisher_decomposed(phi) -> torch.Tensor:
    """Efficient exact identity F_rho = F_{pi,mu} + F_mu."""
    value = _single_phi(phi)
    return pooled_policy_fisher_closed_form(value) + state_distribution_fisher_closed_form(value)


def cross_fisher_term(phi) -> tuple[torch.Tensor, torch.Tensor]:
    """Enumerate E[g_mu g_pi^T] and its separately accumulated transpose."""
    value = _single_phi(phi)
    rho = joint_state_action_probabilities(value)
    left = torch.zeros((4, 4), dtype=DTYPE)
    right = torch.zeros((4, 4), dtype=DTYPE)
    for state in range(2):
        state_score = state_distribution_score(value, state)
        for action in range(3):
            action_score = policy_score(value, state, action)
            left += rho[state, action] * torch.outer(state_score, action_score)
            right += rho[state, action] * torch.outer(action_score, state_score)
    return left, right


def expected_joint_score(phi) -> torch.Tensor:
    value = _single_phi(phi)
    rho = joint_state_action_probabilities(value)
    result = torch.zeros(4, dtype=DTYPE)
    for state in range(2):
        for action in range(3):
            result += rho[state, action] * joint_score(value, state, action)
    return result


def _barrier_components(phi) -> tuple[torch.Tensor, ...]:
    value = _single_phi(phi)
    pi0, pi1 = probabilities_from_reduced_logits(value)
    mu0, mu1 = transition_pool_weights(value)
    b0 = torch.log(pi0).sum()
    b1 = torch.log(pi1).sum()
    return pi0, pi1, mu0, mu1, b0, b1


def pooled_policy_logdet(phi) -> torch.Tensor:
    """One-half log det F_{pi,mu}, historically named full_pooled_fisher."""
    _, _, mu0, mu1, b0, b1 = _barrier_components(phi)
    return 0.5 * (b0 + b1) + torch.log(mu0) + torch.log(mu1)


def joint_logdet_from_matrix(phi) -> torch.Tensor:
    """One-half log determinant computed from the joint Fisher matrix."""
    sign, logabsdet = torch.linalg.slogdet(joint_state_action_fisher_decomposed(phi))
    if float(sign.detach().item()) != 1.0:
        raise ValueError("joint Fisher must have positive determinant for finite interior logits")
    return 0.5 * logabsdet


def joint_logdet_closed_form(phi) -> torch.Tensor:
    """Two-state determinant-lemma expression for one-half log det F_rho."""
    value = _single_phi(phi)
    mu0, _ = transition_pool_weights(value)
    return pooled_policy_logdet(value) + 0.5 * torch.log(2.0 * mu0)


def pooled_policy_logdet_gradient_analytic(phi) -> torch.Tensor:
    """Analytical gradient of one-half log det F_{pi,mu}."""
    value = _single_phi(phi)
    pi0, pi1, _, _, _, _ = _barrier_components(value)
    grad_b0 = torch.cat((1.0 - 3.0 * pi0[:2], torch.zeros(2, dtype=DTYPE)))
    grad_b1 = torch.cat((torch.zeros(2, dtype=DTYPE), 1.0 - 3.0 * pi1[:2]))
    q = pi0[1]
    v = q_gradient(value)
    grad_log_mu0 = -v / (1.0 + q)
    grad_log_mu1 = v / (q * (1.0 + q))
    return 0.5 * (grad_b0 + grad_b1) + grad_log_mu0 + grad_log_mu1


def joint_logdet_gradient_analytic(phi) -> torch.Tensor:
    """Analytical gradient of one-half log det F_rho."""
    value = _single_phi(phi)
    pi0, _ = probabilities_from_reduced_logits(value)
    q = pi0[1]
    grad_log_mu0 = -q_gradient(value) / (1.0 + q)
    return pooled_policy_logdet_gradient_analytic(value) + 0.5 * grad_log_mu0


def joint_visitation_contribution(phi) -> torch.Tensor:
    """q-dependent visitation term, excluding the irrelevant log(2)/2."""
    value = _single_phi(phi)
    mu0, mu1 = transition_pool_weights(value)
    return 1.5 * torch.log(mu0) + torch.log(mu1)
