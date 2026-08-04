"""Independent exact verification for Step 3 two-state geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import json
from pathlib import Path
import sys

import torch

from .experiment import ALL_METHODS, ExactTrainingConfig, train_exact
from .geometry import (
    barrier_gradients,
    barrier_terms,
    enumerated_reduced_fisher,
    pooled_fisher,
    reduced_categorical_fisher,
    reduced_scores,
)
from .model import (
    DTYPE,
    TwoStepTrap,
    phi_from_q_and_good,
    probabilities_from_reduced_logits,
    transition_pool_weights,
)


TOLERANCES = {"algebraic": 1e-12, "logdet": 1e-10, "finite_difference": 1e-7, "minimum_eigenvalue": -1e-12}


@dataclass(frozen=True)
class VerificationResult:
    metrics: dict[str, float]
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "passed": self.passed}


def _maxabs(value: torch.Tensor) -> float:
    return float(value.detach().abs().max().item())


def _autograd_gradient(function, phi: torch.Tensor) -> torch.Tensor:
    work = phi.detach().clone().requires_grad_(True)
    return torch.autograd.grad(function(work), work)[0]


def _finite_difference(function, phi: torch.Tensor, direction: torch.Tensor, step: float = 1e-5) -> torch.Tensor:
    return (function(phi + step * direction) - function(phi - step * direction)) / (2 * step)


def run_verification() -> VerificationResult:
    mdp = TwoStepTrap()
    phi = torch.tensor((0.4, -0.7, -0.2, 0.8), dtype=DTYPE)
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    q = pi0[1]
    mu0, mu1 = transition_pool_weights(phi)
    paths = mdp.trajectories(phi)

    path_probability_error = abs(float(sum(path.probability for path in paths).item()) - 1.0)
    return_error = float((mdp.exact_return(phi) - mdp.enumerated_return(phi)).abs().item())
    expected_counts = torch.tensor((1.0, float(q.item())), dtype=DTYPE)
    expected_weights = expected_counts / expected_counts.sum()
    pool_weight_error = _maxabs(torch.stack((mu0, mu1)) - expected_weights)

    f0, f1 = reduced_categorical_fisher(pi0), reduced_categorical_fisher(pi1)
    f0_enum, f1_enum = enumerated_reduced_fisher(pi0), enumerated_reduced_fisher(pi1)
    score0 = (pi0[:, None] * reduced_scores(pi0)).sum(dim=0)
    score1 = (pi1[:, None] * reduced_scores(pi1)).sum(dim=0)
    expected_score_error = max(_maxabs(score0), _maxabs(score1))
    fisher_enumeration_error = max(_maxabs(f0 - f0_enum), _maxabs(f1 - f1_enum))
    fp = pooled_fisher(phi)
    expected_fp = torch.block_diag(mu0 * f0_enum, mu1 * f1_enum)
    pooled_fisher_error = _maxabs(fp - expected_fp)
    symmetry_error = _maxabs(fp - fp.T)
    minimum_eigenvalue = float(torch.linalg.eigvalsh(fp).min().item())

    terms = barrier_terms(phi)
    sign, logdet = torch.linalg.slogdet(fp)
    decomposition_error = float((logdet - (terms.b0 + terms.b1 + 2 * torch.log(mu0) + 2 * torch.log(mu1))).abs().item())
    normalized_value_error = float((0.5 * logdet - terms.full).abs().item())

    # Independent REINFORCE trajectory enumeration.
    eye = torch.eye(3, 2, dtype=DTYPE)
    reinforce = torch.zeros(4, dtype=DTYPE)
    for path in paths:
        score = torch.zeros(4, dtype=DTYPE)
        score[:2] += eye[path.actions[0]] - pi0[:2]
        if len(path.actions) == 2:
            score[2:] += eye[path.actions[1]] - pi1[:2]
        reinforce += path.probability * path.reward * score
    reward_gradient_error = _maxabs(reinforce - mdp.exact_reward_gradient(phi))

    gradients = barrier_gradients(phi)
    detached_autograd = _autograd_gradient(
        lambda x: transition_pool_weights(x)[0].detach() * barrier_terms(x).b0
        + transition_pool_weights(x)[1].detach() * barrier_terms(x).b1,
        phi,
    )
    detached_error = _maxabs(detached_autograd - gradients.detached_conditional)
    weighted_autograd = _autograd_gradient(lambda x: barrier_terms(x).weighted, phi)
    weighted_autograd_error = _maxabs(weighted_autograd - gradients.complete_weighted)
    weighted_identity_error = _maxabs(
        gradients.complete_weighted - gradients.detached_conditional - gradients.weighted_state_term
    )
    visit_autograd = _autograd_gradient(lambda x: barrier_terms(x).visit, phi)
    visit_autograd_error = _maxabs(visit_autograd - gradients.visitation_only)
    visit_q_formula_error = float(
        (
            (1 - q) / (q * (1 + q))
            - _autograd_gradient(
                lambda x: torch.log(x) - 2 * torch.log1p(x), q.detach().clone()
            )
        ).abs().item()
    )
    full_autograd = _autograd_gradient(lambda x: barrier_terms(x).full, phi)
    full_autograd_error = _maxabs(full_autograd - gradients.full_pooled_fisher)
    full_value_error = float((terms.full - terms.uniform - terms.visit).abs().item())
    full_gradient_error = _maxabs(
        gradients.full_pooled_fisher - gradients.uniform_action - gradients.visitation_only
    )

    direction = torch.tensor((0.2, -0.5, 0.7, -0.1), dtype=DTYPE)
    direction /= direction.norm()
    weighted_fd_error = float(abs(_finite_difference(lambda x: barrier_terms(x).weighted, phi, direction).item() - torch.dot(gradients.complete_weighted, direction).item()))
    visit_fd_error = float(abs(_finite_difference(lambda x: barrier_terms(x).visit, phi, direction).item() - torch.dot(gradients.visitation_only, direction).item()))
    full_fd_error = float(abs(_finite_difference(lambda x: barrier_terms(x).full, phi, direction).item() - torch.dot(gradients.full_pooled_fisher, direction).item()))
    detached_weighted_difference = float(torch.linalg.vector_norm(gradients.detached_conditional - gradients.complete_weighted).item())

    grid_phi = phi_from_q_and_good(
        torch.tensor((0.02, 0.9), dtype=DTYPE),
        torch.tensor((0.9, 0.02), dtype=DTYPE),
    )
    grid_pi0, grid_pi1 = probabilities_from_reduced_logits(grid_phi)
    constructor_error = max(_maxabs(grid_pi0[:, 1] - torch.tensor((0.02, 0.9), dtype=DTYPE)), _maxabs(grid_pi1[:, 0] - torch.tensor((0.9, 0.02), dtype=DTYPE)))

    robust_finite = True
    for alpha in (0.025, 0.05, 0.1):
        for method in ALL_METHODS:
            beta = 0.0 if method == "reward_only" else 0.1
            result = train_exact(ExactTrainingConfig(method, alpha, beta, round(1.0 / alpha), "verify"), (2, -2, -2, 2))
            robust_finite &= bool(result.finite.all())

    metrics = {
        "path_probability_error": path_probability_error, "return_error": return_error,
        "pool_weight_error": pool_weight_error, "expected_score_error": expected_score_error,
        "fisher_enumeration_error": fisher_enumeration_error, "pooled_fisher_error": pooled_fisher_error,
        "symmetry_error": symmetry_error, "minimum_eigenvalue": minimum_eigenvalue,
        "slogdet_sign": float(sign.item()), "decomposition_error": decomposition_error,
        "normalized_value_error": normalized_value_error, "reward_gradient_error": reward_gradient_error,
        "detached_error": detached_error, "weighted_autograd_error": weighted_autograd_error,
        "weighted_identity_error": weighted_identity_error, "visit_autograd_error": visit_autograd_error,
        "visit_q_formula_error": visit_q_formula_error, "full_autograd_error": full_autograd_error,
        "full_value_error": full_value_error, "full_gradient_error": full_gradient_error,
        "weighted_fd_error": weighted_fd_error, "visit_fd_error": visit_fd_error,
        "full_fd_error": full_fd_error, "detached_weighted_difference": detached_weighted_difference,
        "constructor_error": constructor_error,
    }
    algebraic_names = [name for name in metrics if name.endswith("error") and "fd_" not in name and name != "decomposition_error"]
    checks = {
        **{name: metrics[name] <= TOLERANCES["algebraic"] for name in algebraic_names},
        "decomposition": decomposition_error <= TOLERANCES["logdet"],
        "weighted_fd": weighted_fd_error <= TOLERANCES["finite_difference"],
        "visit_fd": visit_fd_error <= TOLERANCES["finite_difference"],
        "full_fd": full_fd_error <= TOLERANCES["finite_difference"],
        "positive_definite": minimum_eigenvalue > 0,
        "positive_slogdet": float(sign.item()) == 1.0,
        "detached_is_distinct": detached_weighted_difference > 1e-8,
        "visitation_pressure_positive": float(((1 - q) / (q * (1 + q))).item()) > 0,
        "step_size_runs_finite": robust_finite,
    }
    return VerificationResult(metrics, checks)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args(argv)
    result = run_verification()
    print("Step 3 exact two-state verification")
    for name, passed in result.checks.items():
        print(f"  {name:34s} {'PASS' if passed else 'FAIL'}")
    print(f"all_passed={result.passed}")
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {"schema_version": 1, "dtype": "torch.float64", "torch_version": torch.__version__, "tolerances": TOLERANCES, **result.to_dict()}
        args.json_output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
