"""Distribution and score definitions for the exact joint Fisher.

The parameter vector contains two reduced logits for each of two states.  The
third action logit is fixed to zero in each state, so every score is expressed
in the same four-dimensional reduced chart.
"""

from __future__ import annotations

import torch

from exploration.tabular_mdp.model import (
    DTYPE,
    probabilities_from_reduced_logits,
    transition_pool_weights,
)
from exploration.tabular_mdp.geometry import reduced_scores


def _validated_phi(phi) -> torch.Tensor:
    """Validate without detaching, so public definitions remain differentiable."""
    value = torch.as_tensor(phi, dtype=DTYPE, device="cpu")
    if value.ndim not in (1, 2) or value.shape[-1] != 4:
        raise ValueError("phi must have shape (4,) or (batch, 4)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("phi must contain only finite values")
    return value


def _single_phi(phi) -> torch.Tensor:
    value = _validated_phi(phi)
    if value.ndim != 1:
        raise ValueError("this exact score operation requires phi with shape (4,)")
    return value


def _validate_outcome(state: int, action: int | None = None) -> None:
    if not isinstance(state, int) or isinstance(state, bool) or state not in (0, 1):
        raise ValueError("state must be 0 or 1")
    if action is not None and (
        not isinstance(action, int) or isinstance(action, bool) or action not in (0, 1, 2)
    ):
        raise ValueError("action must be 0, 1, or 2")


def joint_state_action_probabilities(phi) -> torch.Tensor:
    """Return rho(s,a)=mu(s)pi(a|s), with final shape ``(..., 2, 3)``."""
    value = _validated_phi(phi)
    pi0, pi1 = probabilities_from_reduced_logits(value)
    mu0, mu1 = transition_pool_weights(value)
    return torch.stack((mu0[..., None] * pi0, mu1[..., None] * pi1), dim=-2)


def policy_score(phi, state: int, action: int) -> torch.Tensor:
    """Return the conditional-policy score embedded in the full 4D chart."""
    _validate_outcome(state, action)
    value = _single_phi(phi)
    pi0, pi1 = probabilities_from_reduced_logits(value)
    local = reduced_scores(pi0 if state == 0 else pi1)[action]
    score = torch.zeros(4, dtype=DTYPE, device="cpu")
    start = 2 * state
    score[start : start + 2] = local
    return score


def q_gradient(phi) -> torch.Tensor:
    """Analytical gradient of q=pi_0(a_1) in the full reduced chart."""
    value = _single_phi(phi)
    pi0, _ = probabilities_from_reduced_logits(value)
    q = pi0[1]
    return torch.stack(
        (-q * pi0[0], q * (1.0 - q), torch.zeros((), dtype=DTYPE), torch.zeros((), dtype=DTYPE))
    )


def state_distribution_score(phi, state: int) -> torch.Tensor:
    """Return grad log mu(state) for the transition-pooled state law."""
    _validate_outcome(state)
    value = _single_phi(phi)
    pi0, _ = probabilities_from_reduced_logits(value)
    q = pi0[1]
    v = q_gradient(value)
    if state == 0:
        return -v / (1.0 + q)
    return v / (q * (1.0 + q))


def joint_score(phi, state: int, action: int) -> torch.Tensor:
    """Return grad log rho(state, action)."""
    _validate_outcome(state, action)
    return state_distribution_score(phi, state) + policy_score(phi, state, action)
