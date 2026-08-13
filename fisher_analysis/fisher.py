"""Streaming empirical-Fisher construction and spectral diagnostics."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical, Normal
from torch.func import functional_call, grad as functional_grad, vmap


SCORE_BATCH_SIZE = 1024


def parameter_layout(policy: torch.nn.Module) -> list[dict[str, Any]]:
    """Describe the parameter order used to flatten policy scores."""

    layout: list[dict[str, Any]] = []
    offset = 0
    for name, parameter in policy.named_parameters():
        stop = offset + parameter.numel()
        layout.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "start": offset,
                "stop": stop,
                "numel": parameter.numel(),
                "dtype": str(parameter.dtype),
            }
        )
        offset = stop
    return layout


def state_dict_snapshot(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }


def assert_policy_unchanged(
    policy: torch.nn.Module,
    initial_state: dict[str, torch.Tensor],
) -> None:
    """Verify that a nominally fixed-policy analysis did not mutate the policy."""

    current_state = policy.state_dict()
    if current_state.keys() != initial_state.keys():
        raise AssertionError("policy state layout changed during Fisher analysis")
    for name, initial_value in initial_state.items():
        if not torch.equal(current_state[name].detach().cpu(), initial_value):
            raise AssertionError(
                f"policy tensor {name!r} changed during Fisher analysis"
            )
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise AssertionError(
            "Fisher analysis unexpectedly populated parameter gradients"
        )


def compute_empirical_fisher(
    policy: torch.nn.Module,
    states: np.ndarray | torch.Tensor,
    actions: np.ndarray | torch.Tensor,
    *,
    score_batch_size: int = SCORE_BATCH_SIZE,
) -> torch.Tensor:
    """Accumulate the undamped empirical Fisher in float64 score batches."""

    if score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive")

    parameter_items = list(policy.named_parameters())
    if not parameter_items:
        raise ValueError("policy has no trainable parameters")
    if any(parameter.dtype != torch.float64 for _, parameter in parameter_items):
        raise ValueError("policy parameters must be float64")
    if any(parameter.device.type != "cpu" for _, parameter in parameter_items):
        raise ValueError("Fisher analysis currently requires a CPU policy")

    state_tensor = torch.as_tensor(states, dtype=torch.float64, device="cpu")
    action_tensor = torch.as_tensor(actions, device="cpu")
    if state_tensor.ndim != 2:
        raise ValueError("states must have shape [samples, state_dim]")
    if action_tensor.shape[0] != state_tensor.shape[0]:
        raise ValueError("states and actions must have the same sample count")
    if state_tensor.shape[0] == 0:
        raise ValueError("at least one state/action sample is required")

    parameters = dict(parameter_items)
    buffers = dict(policy.named_buffers())
    if parameters.keys() & buffers.keys():
        raise ValueError("policy parameter and buffer names overlap")

    parameter_count = sum(parameter.numel() for parameter in parameters.values())
    fisher_sum = torch.zeros(
        (parameter_count, parameter_count),
        dtype=torch.float64,
        device="cpu",
    )

    def single_log_prob(
        functional_parameters: dict[str, torch.Tensor],
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        output = functional_call(
            policy,
            {**functional_parameters, **buffers},
            (state.unsqueeze(0),),
        )
        if isinstance(output, tuple):
            mean, std = output
            return Normal(mean.squeeze(0), std).log_prob(action).sum()
        return Categorical(logits=output.squeeze(0)).log_prob(action.long())

    score_function = vmap(
        functional_grad(single_log_prob, argnums=0),
        in_dims=(None, 0, 0),
    )
    sample_count = state_tensor.shape[0]
    for start in range(0, sample_count, score_batch_size):
        stop = min(start + score_batch_size, sample_count)
        batch_actions = action_tensor[start:stop]
        batch_actions = (
            batch_actions.long()
            if batch_actions.ndim == 1
            else batch_actions.to(torch.float64)
        )
        per_parameter_scores = score_function(
            parameters,
            state_tensor[start:stop],
            batch_actions,
        )
        batch_size = stop - start
        score_matrix = torch.cat(
            [
                per_parameter_scores[name].reshape(batch_size, -1)
                for name, _ in parameter_items
            ],
            dim=1,
        ).detach()
        fisher_sum.addmm_(score_matrix.T, score_matrix)

    return fisher_sum / sample_count


def analyze_fisher(
    fisher: np.ndarray | torch.Tensor,
    *,
    sample_count: int,
) -> tuple[np.ndarray, dict[str, float | int], float, float]:
    """Validate a Fisher matrix and summarize its descending eigenspectrum."""

    matrix = (
        fisher.detach().cpu().numpy()
        if isinstance(fisher, torch.Tensor)
        else np.asarray(fisher)
    )
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("fisher must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise AssertionError("Fisher matrix contains non-finite values")

    dimension = matrix.shape[0]
    matrix_scale = max(1.0, float(np.max(np.abs(matrix), initial=0.0)))
    symmetry_tolerance = (
        64.0 * dimension * np.finfo(np.float64).eps * matrix_scale
    )
    symmetry_error = float(np.max(np.abs(matrix - matrix.T), initial=0.0))
    if symmetry_error > symmetry_tolerance:
        raise AssertionError(
            "Fisher matrix is not symmetric: "
            f"max error {symmetry_error:.3e} > {symmetry_tolerance:.3e}"
        )

    eigenvalues = np.linalg.eigvalsh(matrix)[::-1].copy()
    spectral_scale = float(np.max(np.abs(eigenvalues), initial=0.0))
    rank_tolerance = max(
        dimension * np.finfo(np.float64).eps * spectral_scale,
        np.finfo(np.float64).tiny,
    )
    psd_tolerance = max(
        100.0 * rank_tolerance,
        1e-12 * max(1.0, spectral_scale),
    )
    minimum_eigenvalue = float(eigenvalues[-1]) if dimension else 0.0
    if minimum_eigenvalue < -psd_tolerance:
        raise AssertionError(
            "Fisher matrix is not positive semidefinite: "
            f"minimum eigenvalue {minimum_eigenvalue:.3e} "
            f"< {-psd_tolerance:.3e}"
        )

    trace = float(np.trace(matrix))
    eigenvalue_sum = float(np.sum(eigenvalues))
    if not np.isclose(
        trace,
        eigenvalue_sum,
        rtol=1e-10,
        atol=psd_tolerance * max(1, dimension),
    ):
        raise AssertionError(
            f"Fisher trace {trace:.16e} != eigenvalue sum {eigenvalue_sum:.16e}"
        )

    positive = eigenvalues[eigenvalues > rank_tolerance]
    numerical_rank = int(positive.size)
    positive_trace = float(np.sum(positive))
    if numerical_rank:
        shares = positive / positive_trace
        effective_rank = float(np.exp(-np.sum(shares * np.log(shares))))
        stable_rank = float(np.sum(positive**2) / positive[0] ** 2)
        condition_number = float(positive[0] / positive[-1])
        cumulative = np.cumsum(positive) / positive_trace

        def components_for(fraction: float) -> int:
            return int(np.searchsorted(cumulative, fraction, side="left") + 1)

    else:
        effective_rank = 0.0
        stable_rank = 0.0
        condition_number = math.nan

        def components_for(fraction: float) -> int:
            del fraction
            return 0

    metrics: dict[str, float | int] = {
        "matrix_dimension": dimension,
        "sample_count": int(sample_count),
        "trace": trace,
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "positive_condition_number": condition_number,
        "components_90": components_for(0.90),
        "components_95": components_for(0.95),
        "components_99": components_for(0.99),
        "minimum_eigenvalue": minimum_eigenvalue,
        "symmetry_error": symmetry_error,
    }
    return eigenvalues, metrics, rank_tolerance, psd_tolerance
