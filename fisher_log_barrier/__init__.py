"""Trajectory-Fisher log-determinant regularization.

This package is intentionally separate from :mod:`log_barrier`, whose
sampled Acrobot regularizer is the statewise categorical action barrier.
"""

from .loss1 import (
    SCORE_BACKENDS,
    FisherInverseEstimate,
    FisherLogDetDiagnostics,
    FisherLogDetDomainError,
    estimate_trajectory_fisher_inverse,
    trajectory_fisher_logdet_surrogate,
)
from .policy import ReferenceMLPSoftmaxPolicy

__all__ = [
    "FisherInverseEstimate",
    "FisherLogDetDiagnostics",
    "FisherLogDetDomainError",
    "ReferenceMLPSoftmaxPolicy",
    "SCORE_BACKENDS",
    "estimate_trajectory_fisher_inverse",
    "trajectory_fisher_logdet_surrogate",
]
