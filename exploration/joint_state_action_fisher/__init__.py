"""Exact joint state-action Fisher geometry for the two-step tabular MDP."""

from .definitions import (
    joint_score,
    joint_state_action_probabilities,
    policy_score,
    q_gradient,
    state_distribution_score,
)
from .geometry import (
    joint_logdet_closed_form,
    joint_logdet_from_matrix,
    joint_logdet_gradient_analytic,
    joint_state_action_fisher_decomposed,
    joint_state_action_fisher_enumerated,
    pooled_policy_fisher_closed_form,
    pooled_policy_fisher_enumerated,
    pooled_policy_logdet,
    state_distribution_fisher_closed_form,
    state_distribution_fisher_enumerated,
)

__all__ = [
    "joint_logdet_closed_form",
    "joint_logdet_from_matrix",
    "joint_logdet_gradient_analytic",
    "joint_score",
    "joint_state_action_fisher_decomposed",
    "joint_state_action_fisher_enumerated",
    "joint_state_action_probabilities",
    "policy_score",
    "pooled_policy_fisher_closed_form",
    "pooled_policy_fisher_enumerated",
    "pooled_policy_logdet",
    "q_gradient",
    "state_distribution_fisher_closed_form",
    "state_distribution_fisher_enumerated",
    "state_distribution_score",
]
