"""Deterministic verification gate for the exact finite MDP."""

from __future__ import annotations

import json

import torch

from .geometry import (
    barrier_gradient,
    enumerated_joint_fisher,
    joint_fisher,
    policy_fisher,
    state_fisher,
)
from .model import DTYPE, ThreeStateChain, discounted_occupancy
from .training import ExactTrainingConfig, train


def verification_results() -> dict:
    phi = torch.tensor((0.3, -0.7, -0.2, 0.8, 0.5, -0.4), dtype=DTYPE)
    occupancy = discounted_occupancy(phi, 0.99)
    mdp = ThreeStateChain()
    differentiable_phi = phi.clone().requires_grad_(True)
    autograd_reward = torch.autograd.grad(mdp.exact_return(differentiable_phi), differentiable_phi)[0]
    policy = policy_fisher(phi)
    state = state_fisher(phi)
    joint = joint_fisher(phi)
    enumerated = enumerated_joint_fisher(phi)
    checks = {
        "occupancy_sum_error": abs(float(occupancy.sum()) - 1.0),
        "reward_gradient_error": float(torch.max(torch.abs(mdp.reward_gradient(phi) - autograd_reward))),
        "joint_decomposition_error": float(torch.max(torch.abs(joint - policy - state))),
        "joint_enumeration_error": float(torch.max(torch.abs(joint - enumerated))),
        "policy_symmetry_error": float(torch.max(torch.abs(policy - policy.T))),
        "joint_symmetry_error": float(torch.max(torch.abs(joint - joint.T))),
        "policy_minimum_eigenvalue": float(torch.linalg.eigvalsh(policy)[0]),
        "joint_minimum_eigenvalue": float(torch.linalg.eigvalsh(joint)[0]),
        "policy_barrier_gradient_finite": bool(torch.isfinite(barrier_gradient(phi, "policy")).all()),
        "joint_barrier_gradient_finite": bool(torch.isfinite(barrier_gradient(phi, "joint")).all()),
    }
    exact_errors = [value for key, value in checks.items() if key.endswith("error")]
    checks["all_passed"] = (
        max(exact_errors) < 1e-11
        and checks["policy_minimum_eigenvalue"] > 0.0
        and checks["joint_minimum_eigenvalue"] > 0.0
        and checks["policy_barrier_gradient_finite"]
        and checks["joint_barrier_gradient_finite"]
    )
    smoke = train(ExactTrainingConfig("joint_fisher_logdet", updates=10))
    checks["smoke_trajectory_rows"] = len(smoke.trajectory)
    checks["smoke_spectrum_rows"] = len(smoke.spectra)
    checks["all_passed"] = checks["all_passed"] and len(smoke.trajectory) == 11 and len(smoke.spectra) == 22
    return checks


def main() -> int:
    results = verification_results()
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if results["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
