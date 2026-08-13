"""Undamped, action-enumerated empirical policy-Fisher diagnostics."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import torch
from torch.distributions import Categorical
from torch.func import functional_call, grad as functional_grad, vmap


@dataclass(frozen=True)
class EmpiricalFisherMetrics:
    parameter_count: int
    state_count: int
    action_count: int
    numerical_rank: int
    threshold: float
    trace: float
    largest_eigenvalue: float
    smallest_positive_eigenvalue: float
    condition_number: float
    log_pseudodeterminant: float
    k90: int
    k95: int
    k99: int
    effective_rank: float
    stable_rank: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EmpiricalFisherSpectrum:
    eigenvalues: torch.Tensor
    metrics: EmpiricalFisherMetrics


def action_enumerated_score_matrix(policy: torch.nn.Module, states: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Rows are ``sqrt(pi(a|s)) * grad log pi(a|s)`` for all sampled states/actions."""

    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("states must have shape [M, state_dim]")
    # Work on a copy: converting the training policy to float64 would invalidate
    # the float32 Adam state and silently change the experiment.
    model = copy.deepcopy(policy).to(device="cpu", dtype=torch.float64)
    states = states.detach().cpu().to(torch.float64)
    parameters = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    with torch.no_grad():
        probabilities = torch.softmax(model(states), dim=-1)
    action_count = int(probabilities.shape[1])
    state_indices = torch.arange(states.shape[0]).repeat_interleave(action_count)
    actions = torch.arange(action_count).repeat(states.shape[0])
    row_states = states[state_indices]

    def log_probability(params, state, action):
        logits = functional_call(model, {**params, **buffers}, (state.unsqueeze(0),))
        return Categorical(logits=logits.squeeze(0)).log_prob(action)

    gradients = vmap(functional_grad(log_probability), in_dims=(None, 0, 0))(
        parameters, row_states, actions
    )
    scores = torch.cat(
        [gradient.reshape(row_states.shape[0], -1) for gradient in gradients.values()], dim=1
    )
    scores = scores * probabilities.reshape(-1).sqrt().unsqueeze(1)
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("non-finite policy score matrix")
    return scores, action_count


def empirical_policy_fisher_spectrum(
    policy: torch.nn.Module,
    states: torch.Tensor,
    threshold_relative: float | None = None,
) -> EmpiricalFisherSpectrum:
    """Return the positive spectrum of ``S.T S / M`` without damping or floors."""

    scores, action_count = action_enumerated_score_matrix(policy, states)
    state_count, parameter_count = int(states.shape[0]), int(scores.shape[1])
    if parameter_count <= scores.shape[0]:
        matrix = scores.T @ scores / state_count
    else:
        matrix = scores @ scores.T / state_count
    eigenvalues = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T)).flip(0).detach()
    largest_raw = max(0.0, float(eigenvalues[0])) if eigenvalues.numel() else 0.0
    if threshold_relative is None:
        threshold = parameter_count * torch.finfo(torch.float64).eps * largest_raw
    else:
        threshold = threshold_relative * max(1.0, largest_raw)
    positive = eigenvalues[eigenvalues > threshold]
    if positive.numel():
        shares = positive / positive.sum()
        cumulative = torch.cumsum(shares, 0)
        components = lambda fraction: int(
            torch.searchsorted(cumulative, torch.tensor(fraction, dtype=torch.float64)) + 1
        )
        metrics = EmpiricalFisherMetrics(
            parameter_count=parameter_count,
            state_count=state_count,
            action_count=action_count,
            numerical_rank=int(positive.numel()),
            threshold=threshold,
            trace=float(positive.sum()),
            largest_eigenvalue=float(positive[0]),
            smallest_positive_eigenvalue=float(positive[-1]),
            condition_number=float(positive[0] / positive[-1]),
            log_pseudodeterminant=float(torch.log(positive).sum()),
            k90=components(0.90),
            k95=components(0.95),
            k99=components(0.99),
            effective_rank=float(torch.exp(-(shares * torch.log(shares)).sum())),
            stable_rank=float(positive.square().sum() / positive[0].square()),
        )
    else:
        metrics = EmpiricalFisherMetrics(
            parameter_count, state_count, action_count, 0, threshold, 0.0, 0.0, 0.0,
            float("inf"), float("-inf"), 0, 0, 0, 0.0, 0.0,
        )
    return EmpiricalFisherSpectrum(positive.detach().cpu(), metrics)
