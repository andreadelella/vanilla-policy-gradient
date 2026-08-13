"""Exact discounted policy and joint state-action Fisher geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

from .model import DTYPE, as_phi, chain_probabilities, discounted_occupancy


def reduced_categorical_fisher(probabilities: torch.Tensor) -> torch.Tensor:
    """Fisher in the chart where categorical action two is the fixed reference."""

    reduced = probabilities[:2]
    return torch.diag(reduced) - torch.outer(reduced, reduced)


def categorical_score(probabilities: torch.Tensor, action: int) -> torch.Tensor:
    score = -probabilities[:2].clone()
    if action < 2:
        score[action] += 1.0
    return score


def policy_score(phi, state: int, action: int) -> torch.Tensor:
    pi = chain_probabilities(phi)
    score = torch.zeros(6, dtype=DTYPE)
    score[2 * state : 2 * state + 2] = categorical_score(pi[state], action)
    return score


def _q_gradient(phi, state: int) -> torch.Tensor:
    value = as_phi(phi)
    pi = chain_probabilities(value)
    q = pi[state, 1]
    result = torch.zeros_like(value)
    result[2 * state] = -q * pi[state, 0]
    result[2 * state + 1] = q * (1.0 - q)
    return result


def state_score(phi, gamma: float, state: int) -> torch.Tensor:
    """Score of the normalized discounted state distribution, including terminal."""

    value = as_phi(phi)
    pi = chain_probabilities(value)
    q0, q1 = pi[0, 1], pi[1, 1]
    dq0, dq1 = _q_gradient(value, 0), _q_gradient(value, 1)
    if state == 0:
        return torch.zeros(6, dtype=DTYPE)
    if state == 1:
        return dq0 / q0
    if state == 2:
        return dq0 / q0 + dq1 / q1
    if state == 3:
        terminal = discounted_occupancy(value, gamma)[3]
        derivative = -(1.0 - gamma) * (
            (gamma + gamma**2 * q1) * dq0 + gamma**2 * q0 * dq1
        )
        return derivative / terminal
    raise ValueError("state must be 0, 1, 2, or terminal state 3")


def policy_fisher(phi, gamma: float = 0.99) -> torch.Tensor:
    """Discounted policy Fisher E_d E_pi[score_pi score_pi^T]."""

    value = as_phi(phi)
    pi = chain_probabilities(value)
    occupancy = discounted_occupancy(value, gamma)
    return torch.block_diag(
        *(occupancy[state] * reduced_categorical_fisher(pi[state]) for state in range(3))
    )


def state_fisher(phi, gamma: float = 0.99) -> torch.Tensor:
    """Fisher of the policy-dependent discounted state distribution."""

    occupancy = discounted_occupancy(phi, gamma)
    result = torch.zeros((6, 6), dtype=DTYPE)
    for state in range(4):
        score = state_score(phi, gamma, state)
        result = result + occupancy[state] * torch.outer(score, score)
    return result


def joint_fisher(phi, gamma: float = 0.99) -> torch.Tensor:
    """Joint state-action Fisher of rho(s,a)=d_pi(s) pi(a|s)."""

    return policy_fisher(phi, gamma) + state_fisher(phi, gamma)


def enumerated_joint_fisher(phi, gamma: float = 0.99) -> torch.Tensor:
    """Independently enumerate joint scores, including terminal-state mass."""

    value = as_phi(phi)
    pi = chain_probabilities(value)
    occupancy = discounted_occupancy(value, gamma)
    result = torch.zeros((6, 6), dtype=DTYPE)
    for state in range(3):
        state_part = state_score(value, gamma, state)
        for action in range(3):
            score = state_part + policy_score(value, state, action)
            result = result + occupancy[state] * pi[state, action] * torch.outer(score, score)
    terminal_score = state_score(value, gamma, 3)
    return result + occupancy[3] * torch.outer(terminal_score, terminal_score)


def half_logdet(matrix: torch.Tensor) -> torch.Tensor:
    sign, logabsdet = torch.linalg.slogdet(matrix)
    if float(sign.detach()) != 1.0:
        raise FloatingPointError("Fisher matrix does not have a positive determinant")
    return 0.5 * logabsdet


@dataclass(frozen=True)
class ExactFisherMetrics:
    eigenvalues: tuple[float, ...]
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    condition_number: float
    trace: float
    logdet: float
    half_logdet: float
    k90: int
    k95: int
    k99: int
    effective_rank: float
    stable_rank: float

    def to_dict(self, prefix: str = "") -> dict[str, float]:
        row = {
            f"{prefix}eigenvalue_{index + 1}": value
            for index, value in enumerate(self.eigenvalues)
        }
        scalars = asdict(self)
        scalars.pop("eigenvalues")
        row.update({f"{prefix}{key}": value for key, value in scalars.items()})
        return row


def fisher_metrics(matrix: torch.Tensor) -> ExactFisherMetrics:
    symmetric = 0.5 * (matrix + matrix.T)
    eigenvalues = torch.linalg.eigvalsh(symmetric).flip(0)
    if float(eigenvalues[-1]) <= 0.0:
        raise FloatingPointError("exact reduced Fisher must be positive definite")
    sign, logdet = torch.linalg.slogdet(symmetric)
    if float(sign) != 1.0:
        raise FloatingPointError("exact reduced Fisher determinant must be positive")
    shares = eigenvalues / eigenvalues.sum()
    cumulative = torch.cumsum(shares, 0)
    components = lambda fraction: int(
        torch.searchsorted(cumulative, torch.tensor(fraction, dtype=DTYPE)) + 1
    )
    return ExactFisherMetrics(
        eigenvalues=tuple(float(value) for value in eigenvalues),
        minimum_eigenvalue=float(eigenvalues[-1]),
        maximum_eigenvalue=float(eigenvalues[0]),
        condition_number=float(eigenvalues[0] / eigenvalues[-1]),
        trace=float(torch.trace(symmetric)),
        logdet=float(logdet),
        half_logdet=float(0.5 * logdet),
        k90=components(0.90),
        k95=components(0.95),
        k99=components(0.99),
        effective_rank=float(torch.exp(-(shares * torch.log(shares)).sum())),
        stable_rank=float(eigenvalues.square().sum() / eigenvalues[0].square()),
    )


def barrier_gradient(phi, fisher_name: str, gamma: float = 0.99) -> torch.Tensor:
    if fisher_name not in ("policy", "joint"):
        raise ValueError("fisher_name must be policy or joint")
    value = as_phi(phi).detach().clone().requires_grad_(True)
    matrix = policy_fisher(value, gamma) if fisher_name == "policy" else joint_fisher(value, gamma)
    return torch.autograd.grad(half_logdet(matrix), value)[0]
