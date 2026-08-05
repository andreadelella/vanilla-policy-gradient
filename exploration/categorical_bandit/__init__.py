"""Exact categorical-bandit Fisher identities."""

from exploration.categorical_bandit.identity import (
    DEFAULT_TOLERANCES,
    VerificationResult,
    analytic_barrier_gradient,
    categorical_scores,
    fisher_closed_form,
    fisher_from_score_expectation,
    log_barrier,
    reduced_fisher,
    verify_identity_case,
)
from exploration.categorical_bandit.algorithms import (
    AlgorithmSpec,
    apply_update,
    exact_expected_direction,
    policy_log_probabilities,
    raw_update_direction,
    recenter_logits,
)
from exploration.categorical_bandit.environment import BanditBatch, generate_paired_bandits

__all__ = [
    "DEFAULT_TOLERANCES",
    "VerificationResult",
    "analytic_barrier_gradient",
    "categorical_scores",
    "fisher_closed_form",
    "fisher_from_score_expectation",
    "log_barrier",
    "reduced_fisher",
    "verify_identity_case",
    "AlgorithmSpec",
    "apply_update",
    "exact_expected_direction",
    "policy_log_probabilities",
    "raw_update_direction",
    "recenter_logits",
    "BanditBatch",
    "generate_paired_bandits",
]
