"""Fast mathematical verification for the neural categorical barrier/Fisher."""

from __future__ import annotations

import json

import torch

from vpg.policy import MLPSoftmaxPolicy

from .barrier import analytic_logit_gradient, categorical_log_barrier
from .fisher import action_enumerated_score_matrix, fisher_spectrum_from_scores


def run_verification() -> dict:
    torch.manual_seed(23)
    logits = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    barrier, diagnostics = categorical_log_barrier(logits)
    autograd_gradient, = torch.autograd.grad(barrier, logits)
    analytic = analytic_logit_gradient(logits.detach())
    gradient_error = float((autograd_gradient - analytic).abs().max())

    direction = torch.randn_like(logits)
    direction /= direction.norm()
    epsilon = 1e-6
    plus, _ = categorical_log_barrier(logits.detach() + epsilon * direction)
    minus, _ = categorical_log_barrier(logits.detach() - epsilon * direction)
    finite_difference = float((plus - minus) / (2.0 * epsilon))
    directional = float((analytic * direction).sum())

    policy = MLPSoftmaxPolicy(4, 3, (4,)).to(torch.float64)
    states = torch.randn(7, 4, dtype=torch.float64)
    scores, action_count = action_enumerated_score_matrix(policy, states)
    spectrum = fisher_spectrum_from_scores(scores, state_count=7, action_count=action_count)
    dense = scores.T @ scores / 7.0
    dense_eigenvalues = torch.linalg.eigvalsh(dense)
    trace_error = abs(float(dense.trace().detach()) - spectrum.metrics.trace)
    psd_minimum = float(dense_eigenvalues.min().detach())
    result = {
        "barrier_value": float(barrier.detach()),
        "probability_sum_error": float((torch.softmax(logits, -1).sum(-1) - 1.0).abs().max().detach()),
        "analytic_gradient_error": gradient_error,
        "finite_difference_error": abs(finite_difference - directional),
        "fisher_symmetry_error": float((dense - dense.T).abs().max()),
        "fisher_minimum_eigenvalue": psd_minimum,
        "fisher_trace_reconstruction_error": trace_error,
        "numerical_rank": spectrum.metrics.numerical_rank,
        "maximum_possible_rank": spectrum.metrics.maximum_possible_rank,
        "all_passed": bool(
            gradient_error < 1e-12
            and abs(finite_difference - directional) < 1e-8
            and float((dense - dense.T).abs().max()) < 1e-12
            and psd_minimum >= -1e-12
            and trace_error < 1e-10
        ),
    }
    return result


def main() -> int:
    result = run_verification()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
