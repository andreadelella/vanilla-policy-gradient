"""Exact two-state tabular MDP geometry experiments."""

from .experiment import ExactTrainingConfig, ExactTrainingResult, MethodName, train_exact
from .geometry import barrier_gradients, barrier_terms, geometry_snapshot
from .model import TwoStepTrap, probabilities_from_reduced_logits, reduced_logits_from_probabilities

__all__ = [
    "ExactTrainingConfig",
    "ExactTrainingResult",
    "MethodName",
    "TwoStepTrap",
    "barrier_gradients",
    "barrier_terms",
    "geometry_snapshot",
    "probabilities_from_reduced_logits",
    "reduced_logits_from_probabilities",
    "train_exact",
]
