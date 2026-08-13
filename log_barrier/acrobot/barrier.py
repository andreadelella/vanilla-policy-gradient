"""On-policy sampled-state conditional categorical log barrier.

For a categorical policy, the reduced per-state Fisher determinant is the
product of its action probabilities.  Averaging ``log pi(a|s)`` over sampled
states and actions is therefore the action-normalized conditional log-volume
barrier.  It is not a global state-action Fisher log determinant.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class BarrierDiagnostics:
    state_count: int
    action_count: int
    mean_min_probability: float
    global_min_probability: float
    mean_entropy: float
    barrier_value: float

    def to_dict(self) -> dict:
        return asdict(self)


def categorical_log_barrier(logits: torch.Tensor) -> tuple[torch.Tensor, BarrierDiagnostics]:
    """Return ``mean_s mean_a log pi(a|s)`` and detached diagnostics."""

    if logits.ndim != 2 or logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("logits must have shape [states, actions] with at least two actions")
    if not torch.is_floating_point(logits) or not bool(torch.isfinite(logits).all()):
        raise ValueError("logits must be finite floating-point values")
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    minima = probabilities.min(dim=-1).values
    barrier = log_probabilities.mean()
    diagnostics = BarrierDiagnostics(
        state_count=int(logits.shape[0]),
        action_count=int(logits.shape[1]),
        mean_min_probability=float(minima.mean().detach().cpu()),
        global_min_probability=float(minima.min().detach().cpu()),
        mean_entropy=float(entropy.mean().detach().cpu()),
        barrier_value=float(barrier.detach().cpu()),
    )
    return barrier, diagnostics


def analytic_logit_gradient(logits: torch.Tensor) -> torch.Tensor:
    probabilities = torch.softmax(logits, dim=-1)
    state_count, action_count = logits.shape
    return (torch.full_like(probabilities, 1.0 / action_count) - probabilities) / state_count
