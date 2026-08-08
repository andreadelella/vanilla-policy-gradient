"""Independent deterministic verification of the joint state-action Fisher."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import argparse
import csv
import json
from pathlib import Path
import platform
import subprocess
import sys
from typing import Callable

import torch

from exploration.tabular_mdp.geometry import pooled_fisher as legacy_pooled_policy_fisher
from exploration.tabular_mdp.model import (
    DTYPE,
    phi_from_q_and_good,
    probabilities_from_reduced_logits,
    transition_pool_weights,
)

from .definitions import (
    joint_score,
    joint_state_action_probabilities,
    policy_score,
    q_gradient,
    state_distribution_score,
)
from .geometry import (
    cross_fisher_term,
    expected_joint_score,
    joint_logdet_closed_form,
    joint_logdet_from_matrix,
    joint_logdet_gradient_analytic,
    joint_state_action_fisher_decomposed,
    joint_state_action_fisher_enumerated,
    joint_visitation_contribution,
    pooled_policy_fisher_closed_form,
    pooled_policy_fisher_enumerated,
    pooled_policy_logdet,
    state_distribution_fisher_closed_form,
    state_distribution_fisher_enumerated,
)


SCHEMA_VERSION = 1
TOLERANCES = {
    "algebraic": 1e-12,
    "logdet": 1e-10,
    "finite_difference": 1e-7,
    "minimum_eigenvalue": -1e-12,
}
DEFAULT_OUTPUT = Path("exploration/results/joint_state_action_fisher/step1_identity")


@dataclass(frozen=True)
class IdentityCaseResult:
    name: str
    phi: tuple[float, float, float, float]
    metrics: dict[str, float]
    checks: dict[str, bool]

    @property
    def passed(self) -> bool:
        return all(self.checks.values())

    def to_dict(self) -> dict[str, object]:
        return {**asdict(self), "passed": self.passed}


@dataclass(frozen=True)
class IdentityVerificationResult:
    cases: tuple[IdentityCaseResult, ...]

    @property
    def passed(self) -> bool:
        return all(case.passed for case in self.cases)

    def to_dict(self) -> dict[str, object]:
        return {"cases": [case.to_dict() for case in self.cases], "all_passed": self.passed}


def _maxabs(value: torch.Tensor) -> float:
    return float(value.detach().abs().max().item())


def _frobenius(value: torch.Tensor) -> float:
    return float(torch.linalg.matrix_norm(value.detach(), ord="fro").item())


def _autograd_gradient(function: Callable[[torch.Tensor], torch.Tensor], phi: torch.Tensor) -> torch.Tensor:
    work = phi.detach().clone().requires_grad_(True)
    return torch.autograd.grad(function(work), work)[0]


def _directional_finite_difference(
    function: Callable[[torch.Tensor], torch.Tensor],
    phi: torch.Tensor,
    direction: torch.Tensor,
    step: float = 1e-5,
) -> torch.Tensor:
    return (function(phi + step * direction) - function(phi - step * direction)) / (2.0 * step)


def _autograd_score_errors(phi: torch.Tensor) -> tuple[float, float, float, float]:
    q_auto = _autograd_gradient(lambda x: probabilities_from_reduced_logits(x)[0][1], phi)
    q_error = _maxabs(q_auto - q_gradient(phi))
    policy_error = 0.0
    state_error = 0.0
    joint_error = 0.0
    for state in range(2):
        state_auto = _autograd_gradient(lambda x, s=state: torch.log(transition_pool_weights(x)[s]), phi)
        state_error = max(state_error, _maxabs(state_auto - state_distribution_score(phi, state)))
        for action in range(3):
            policy_auto = _autograd_gradient(
                lambda x, s=state, a=action: torch.log(probabilities_from_reduced_logits(x)[s][a]), phi
            )
            joint_auto = _autograd_gradient(
                lambda x, s=state, a=action: torch.log(joint_state_action_probabilities(x)[s, a]), phi
            )
            policy_error = max(policy_error, _maxabs(policy_auto - policy_score(phi, state, action)))
            joint_error = max(joint_error, _maxabs(joint_auto - joint_score(phi, state, action)))
    return q_error, policy_error, state_error, joint_error


def verify_case(name: str, phi) -> IdentityCaseResult:
    value = torch.as_tensor(phi, dtype=DTYPE, device="cpu")
    if value.shape != (4,) or not bool(torch.isfinite(value).all()):
        raise ValueError("verification phi must be a finite vector with shape (4,)")

    pi0, pi1 = probabilities_from_reduced_logits(value)
    mu0, mu1 = transition_pool_weights(value)
    mu = torch.stack((mu0, mu1))
    rho = joint_state_action_probabilities(value)

    total_probability_error = float((rho.sum() - 1.0).abs().item())
    marginal_error = _maxabs(rho.sum(dim=1) - mu)
    conditional_error = max(_maxabs(rho[0] / mu0 - pi0), _maxabs(rho[1] / mu1 - pi1))

    q_score_error, policy_score_error, state_score_error, joint_score_error = _autograd_score_errors(value)
    expected_score_error = _maxabs(expected_joint_score(value))

    f_policy_closed = pooled_policy_fisher_closed_form(value)
    f_policy_enum = pooled_policy_fisher_enumerated(value)
    f_state_closed = state_distribution_fisher_closed_form(value)
    f_state_enum = state_distribution_fisher_enumerated(value)
    f_joint_enum = joint_state_action_fisher_enumerated(value)
    f_joint_decomposed = joint_state_action_fisher_decomposed(value)
    cross, cross_transpose = cross_fisher_term(value)

    policy_fisher_enumeration_error = _frobenius(f_policy_closed - f_policy_enum)
    legacy_policy_fisher_error = _frobenius(f_policy_closed - legacy_pooled_policy_fisher(value))
    state_fisher_closed_form_error = _frobenius(f_state_closed - f_state_enum)
    joint_fisher_decomposition_error = _frobenius(f_joint_enum - f_joint_decomposed)
    cross_term_error = _frobenius(cross)
    transpose_cross_term_error = _frobenius(cross_transpose)
    symmetry_error = max(
        _maxabs(f_policy_closed - f_policy_closed.T),
        _maxabs(f_state_closed - f_state_closed.T),
        _maxabs(f_joint_decomposed - f_joint_decomposed.T),
    )

    policy_eigenvalues = torch.linalg.eigvalsh(f_policy_closed)
    state_eigenvalues = torch.linalg.eigvalsh(f_state_closed)
    joint_eigenvalues = torch.linalg.eigvalsh(f_joint_decomposed)
    minimum_eigenvalue = float(torch.min(torch.cat((policy_eigenvalues, state_eigenvalues, joint_eigenvalues))).item())
    joint_minimum_eigenvalue = float(joint_eigenvalues[0].item())
    state_rank = int(torch.linalg.matrix_rank(f_state_closed, atol=1e-12, rtol=0.0).item())
    policy_rank = int(torch.linalg.matrix_rank(f_policy_closed, atol=1e-12, rtol=0.0).item())
    joint_rank = int(torch.linalg.matrix_rank(f_joint_decomposed, atol=1e-12, rtol=0.0).item())

    sign_policy, logdet_policy = torch.linalg.slogdet(f_policy_closed)
    sign_joint, logdet_joint = torch.linalg.slogdet(f_joint_decomposed)
    policy_logdet_identity = (
        torch.log(pi0).sum() + torch.log(pi1).sum() + 2.0 * torch.log(mu0) + 2.0 * torch.log(mu1)
    )
    policy_logdet_identity_error = float((logdet_policy - policy_logdet_identity).abs().item())
    determinant_lemma_error = float((logdet_joint - logdet_policy - torch.log(2.0 * mu0)).abs().item())
    joint_logdet_closed_form_error = float((joint_logdet_from_matrix(value) - joint_logdet_closed_form(value)).abs().item())
    pooled_logdet_matrix_error = float((0.5 * logdet_policy - pooled_policy_logdet(value)).abs().item())

    analytic_gradient = joint_logdet_gradient_analytic(value)
    closed_form_autograd = _autograd_gradient(joint_logdet_closed_form, value)
    matrix_autograd = _autograd_gradient(joint_logdet_from_matrix, value)
    closed_form_gradient_error = _maxabs(analytic_gradient - closed_form_autograd)
    matrix_gradient_error = _maxabs(analytic_gradient - matrix_autograd)

    directions = (
        torch.tensor((1.0, 0.0, 0.0, 0.0), dtype=DTYPE),
        torch.tensor((0.2, -0.5, 0.7, -0.1), dtype=DTYPE),
        torch.tensor((-0.3, 0.8, 0.1, 0.5), dtype=DTYPE),
    )
    finite_difference_error = 0.0
    for direction in directions:
        unit = direction / torch.linalg.vector_norm(direction)
        finite = _directional_finite_difference(joint_logdet_closed_form, value, unit)
        exact = torch.dot(analytic_gradient, unit)
        finite_difference_error = max(finite_difference_error, float((finite - exact).abs().item()))

    q = pi0[1]
    expected_visitation_derivative = (1.0 - 1.5 * q) / (q * (1.0 + q))
    scalar_q = q.detach().clone().requires_grad_(True)
    scalar_visitation = torch.log(scalar_q) - 2.5 * torch.log1p(scalar_q)
    visitation_derivative = torch.autograd.grad(scalar_visitation, scalar_q)[0]
    visitation_derivative_absolute_error = float(
        (visitation_derivative - expected_visitation_derivative).abs().item()
    )
    # The derivative grows like 1/q.  At the declared q=1e-6 stress case,
    # comparing the quotient form with autograd by an absolute tolerance would
    # test operation ordering rather than the identity.  Keep the absolute
    # residual for audit and use a scale-normalized residual for the check.
    visitation_derivative_error = visitation_derivative_absolute_error / max(
        1.0, abs(float(expected_visitation_derivative.item()))
    )
    visitation_value_error = float(
        (joint_visitation_contribution(value) - (torch.log(q) - 2.5 * torch.log1p(q))).abs().item()
    )

    metrics = {
        "q": float(q.item()),
        "mu0": float(mu0.item()),
        "mu1": float(mu1.item()),
        "minimum_joint_probability": float(rho.min().item()),
        "total_probability_error": total_probability_error,
        "marginal_error": marginal_error,
        "conditional_error": conditional_error,
        "q_gradient_error": q_score_error,
        "policy_score_autograd_error": policy_score_error,
        "state_score_autograd_error": state_score_error,
        "joint_score_autograd_error": joint_score_error,
        "expected_joint_score_error": expected_score_error,
        "cross_term_frobenius": cross_term_error,
        "transpose_cross_term_frobenius": transpose_cross_term_error,
        "policy_fisher_enumeration_error": policy_fisher_enumeration_error,
        "legacy_policy_fisher_error": legacy_policy_fisher_error,
        "state_fisher_closed_form_error": state_fisher_closed_form_error,
        "joint_fisher_decomposition_error": joint_fisher_decomposition_error,
        "symmetry_error": symmetry_error,
        "minimum_eigenvalue_all_fishers": minimum_eigenvalue,
        "joint_minimum_eigenvalue": joint_minimum_eigenvalue,
        "state_fisher_rank": float(state_rank),
        "pooled_policy_fisher_rank": float(policy_rank),
        "joint_fisher_rank": float(joint_rank),
        "pooled_policy_slogdet_sign": float(sign_policy.item()),
        "joint_slogdet_sign": float(sign_joint.item()),
        "policy_logdet_identity_error": policy_logdet_identity_error,
        "determinant_lemma_error": determinant_lemma_error,
        "joint_logdet_closed_form_error": joint_logdet_closed_form_error,
        "pooled_logdet_matrix_error": pooled_logdet_matrix_error,
        "closed_form_gradient_error": closed_form_gradient_error,
        "matrix_gradient_error": matrix_gradient_error,
        "finite_difference_error": finite_difference_error,
        "visitation_derivative_error": visitation_derivative_error,
        "visitation_derivative_absolute_error": visitation_derivative_absolute_error,
        "visitation_value_error": visitation_value_error,
        "joint_visitation_derivative": float(expected_visitation_derivative.item()),
    }

    algebraic_metrics = (
        "total_probability_error",
        "marginal_error",
        "conditional_error",
        "q_gradient_error",
        "policy_score_autograd_error",
        "state_score_autograd_error",
        "joint_score_autograd_error",
        "expected_joint_score_error",
        "cross_term_frobenius",
        "transpose_cross_term_frobenius",
        "policy_fisher_enumeration_error",
        "legacy_policy_fisher_error",
        "state_fisher_closed_form_error",
        "joint_fisher_decomposition_error",
        "symmetry_error",
        "closed_form_gradient_error",
        "matrix_gradient_error",
        "visitation_derivative_error",
        "visitation_value_error",
    )
    logdet_metrics = (
        "policy_logdet_identity_error",
        "determinant_lemma_error",
        "joint_logdet_closed_form_error",
        "pooled_logdet_matrix_error",
    )
    checks = {
        **{metric: metrics[metric] <= TOLERANCES["algebraic"] for metric in algebraic_metrics},
        **{metric: metrics[metric] <= TOLERANCES["logdet"] for metric in logdet_metrics},
        "finite_difference": finite_difference_error <= TOLERANCES["finite_difference"],
        "strictly_positive_distribution": bool((rho > 0).all()),
        "positive_slogdet_signs": float(sign_policy.item()) == 1.0 and float(sign_joint.item()) == 1.0,
        "positive_semidefinite": minimum_eigenvalue >= TOLERANCES["minimum_eigenvalue"],
        "joint_positive_definite": joint_minimum_eigenvalue > 0.0,
        "expected_ranks": state_rank == 1 and policy_rank == 4 and joint_rank == 4,
    }
    return IdentityCaseResult(name, tuple(float(x) for x in value.tolist()), metrics, checks)


def declared_cases() -> tuple[tuple[str, torch.Tensor], ...]:
    cases: list[tuple[str, torch.Tensor]] = [
        ("uniform", torch.zeros(4, dtype=DTYPE)),
        ("adverse", torch.tensor((2.0, -2.0, -2.0, 2.0), dtype=DTYPE)),
        ("generic_asymmetric", torch.tensor((0.4, -0.7, -0.2, 0.8), dtype=DTYPE)),
    ]
    for q in (1e-2, 1e-4, 1e-6):
        cases.append((f"rare_state_q_{q:.0e}", phi_from_q_and_good(q, 0.9)))
    for q in (0.9, 0.99):
        cases.append((f"high_access_q_{q:g}", phi_from_q_and_good(q, 0.9)))
    generator = torch.Generator(device="cpu").manual_seed(23)
    for index in range(3):
        cases.append((f"random_seed23_case{index}", torch.randn(4, generator=generator, dtype=DTYPE)))
    return tuple(cases)


def run_verification() -> IdentityVerificationResult:
    return IdentityVerificationResult(tuple(verify_case(name, phi) for name, phi in declared_cases()))


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=10
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def write_artifacts(result: IdentityVerificationResult, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for case in result.cases:
        rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "case": case.name,
                **{f"phi_{index}": value for index, value in enumerate(case.phi)},
                **case.metrics,
                "passed": case.passed,
            }
        )
    with (output_dir / "identity_cases.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    verification = {
        "schema_version": SCHEMA_VERSION,
        "dtype": "torch.float64",
        "device": "cpu",
        "seed": 23,
        "tolerances": TOLERANCES,
        **result.to_dict(),
    }
    (output_dir / "verification.json").write_text(
        json.dumps(verification, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "algebraic_verification",
        "status": "complete" if result.passed else "failed",
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "dtype": "torch.float64",
        "device": "cpu",
        "random_seeds": [23],
        "state_weighting_convention": "transition_pooled_population",
        "estimand": "joint_state_action_fisher_of_rho_mu_times_pi",
        "normalization": "one_half_logdet",
        "validation_tolerances": TOLERANCES,
        "case_count": len(result.cases),
        "artifacts": ["identity_cases.csv", "verification.json", "manifest.json"],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    result = run_verification()
    print("Joint state-action Fisher: exact Step 1 verification")
    for case in result.cases:
        failed = sum(not passed for passed in case.checks.values())
        print(
            f"  {case.name:28s} q={case.metrics['q']:.6g} "
            f"decomp={case.metrics['joint_fisher_decomposition_error']:.3e} "
            f"logdet={case.metrics['determinant_lemma_error']:.3e} "
            f"fd={case.metrics['finite_difference_error']:.3e} "
            f"{'PASS' if case.passed else f'FAIL({failed})'}"
        )
    print(f"all_passed={result.passed}")
    if not args.no_write:
        write_artifacts(result, args.output_dir)
        print(f"artifacts={args.output_dir}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
