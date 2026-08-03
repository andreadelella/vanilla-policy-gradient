"""State-wise categorical log barrier used by the neural experiments.

This is deliberately a conditional action-distribution objective evaluated on
sampled states.  It is not a global neural Fisher log determinant.
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
    mean_log_probability: float
    barrier_value: float
    barrier_gradient_norm: float | None = None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _validate_logits(logits: torch.Tensor) -> torch.Tensor:
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits)
    if logits.ndim != 2:
        raise ValueError("categorical logits must have shape [states, actions]")
    if logits.shape[0] < 1 or logits.shape[1] < 2:
        raise ValueError("at least one state and two actions are required")
    if not torch.is_floating_point(logits):
        raise TypeError("categorical logits must use a floating dtype")
    if not torch.isfinite(logits).all():
        raise ValueError("categorical logits must be finite")
    return logits


def categorical_log_barrier(
    logits: torch.Tensor,
    *,
    gradient_parameters: tuple[torch.Tensor, ...] | list[torch.Tensor] | None = None,
) -> tuple[torch.Tensor, BarrierDiagnostics]:
    """Return the normalized sampled-state categorical log barrier.

    For ``M`` states and ``K`` actions the scalar is

        (1/M) sum_m (1/K) sum_a log pi(a | s_m).

    ``logits`` may retain an autograd graph, but the states used to produce it
    should be detached by the caller.  Passing parameters requests a diagnostic
    gradient norm without populating ``.grad`` fields.
    """

    logits = _validate_logits(logits)
    log_probabilities = torch.log_softmax(logits, dim=-1)
    probabilities = log_probabilities.exp()
    if not torch.isfinite(log_probabilities).all():
        raise FloatingPointError("log probabilities are non-finite")

    barrier = log_probabilities.mean()
    entropy = -(probabilities * log_probabilities).sum(dim=-1)
    minima = probabilities.min(dim=-1).values

    gradient_norm: float | None = None
    if gradient_parameters is not None:
        parameters = tuple(gradient_parameters)
        gradients = torch.autograd.grad(
            barrier,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        pieces = [
            torch.zeros_like(parameter).reshape(-1)
            if gradient is None
            else gradient.reshape(-1)
            for parameter, gradient in zip(parameters, gradients)
        ]
        gradient_norm = float(torch.cat(pieces).norm().detach().cpu())

    diagnostics = BarrierDiagnostics(
        state_count=int(logits.shape[0]),
        action_count=int(logits.shape[1]),
        mean_min_probability=float(minima.mean().detach().cpu()),
        global_min_probability=float(minima.min().detach().cpu()),
        mean_entropy=float(entropy.mean().detach().cpu()),
        mean_log_probability=float(log_probabilities.mean().detach().cpu()),
        barrier_value=float(barrier.detach().cpu()),
        barrier_gradient_norm=gradient_norm,
    )
    return barrier, diagnostics


def analytic_logit_gradient(logits: torch.Tensor) -> torch.Tensor:
    """Analytical derivative of the normalized barrier with respect to logits."""

    logits = _validate_logits(logits)
    probabilities = torch.softmax(logits, dim=-1)
    state_count, action_count = logits.shape
    return (
        torch.full_like(probabilities, 1.0 / action_count) - probabilities
    ) / state_count
