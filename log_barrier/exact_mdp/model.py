"""Exact finite three-state MDP used for the policy-versus-joint Fisher study."""

from __future__ import annotations

from dataclasses import dataclass

import torch


DTYPE = torch.float64


def as_phi(phi) -> torch.Tensor:
    """Convert one reduced-logit vector to a finite CPU float64 tensor."""

    value = torch.as_tensor(phi, dtype=DTYPE, device="cpu")
    if value.shape != (6,):
        raise ValueError("phi must contain six reduced logits")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("phi must be finite")
    return value


def chain_probabilities(phi) -> torch.Tensor:
    """Return the three categorical policies; action two is the reference."""

    value = as_phi(phi)
    blocks = []
    for state in range(3):
        reduced = value[2 * state : 2 * state + 2]
        logits = torch.cat((reduced, torch.zeros(1, dtype=DTYPE)))
        blocks.append(torch.softmax(logits, dim=0))
    return torch.stack(blocks)


def discounted_occupancy(phi, gamma: float = 0.99) -> torch.Tensor:
    """Normalized discounted occupancy of s0, s1, s2, and the terminal state."""

    if not 0.0 < float(gamma) < 1.0:
        raise ValueError("gamma must lie strictly between zero and one")
    pi = chain_probabilities(phi)
    q0, q1 = pi[0, 1], pi[1, 1]
    active = torch.stack(
        (
            (1.0 - gamma) * torch.ones((), dtype=DTYPE),
            (1.0 - gamma) * gamma * q0,
            (1.0 - gamma) * gamma**2 * q0 * q1,
        )
    )
    return torch.cat((active, 1.0 - active.sum().reshape(1)))


@dataclass(frozen=True)
class ThreeStateChain:
    """Two sequential continuation decisions followed by a terminal reward."""

    root_safe_reward: float = 0.5
    local_safe_reward: float = 0.55
    deep_rewards: tuple[float, float, float] = (1.0, 0.2, 0.0)

    def values(self, phi) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pi = chain_probabilities(phi)
        deep_rewards = torch.tensor(self.deep_rewards, dtype=DTYPE)
        v2 = torch.dot(pi[2], deep_rewards)
        v1 = self.local_safe_reward * pi[1, 0] + pi[1, 1] * v2
        value = self.root_safe_reward * pi[0, 0] + pi[0, 1] * v1
        return value, v1, v2

    def exact_return(self, phi) -> torch.Tensor:
        return self.values(phi)[0]

    def reward_gradient(self, phi) -> torch.Tensor:
        """Analytical Euclidean gradient of the exact expected return."""

        value = as_phi(phi)
        pi = chain_probabilities(value)
        total, v1, v2 = self.values(value)
        deep_rewards = torch.tensor(self.deep_rewards, dtype=DTYPE)

        root_q = torch.stack(
            (torch.tensor(self.root_safe_reward, dtype=DTYPE), v1, torch.zeros((), dtype=DTYPE))
        )
        middle_q = torch.stack(
            (torch.tensor(self.local_safe_reward, dtype=DTYPE), v2, torch.zeros((), dtype=DTYPE))
        )
        grad0 = pi[0, :2] * (root_q[:2] - total)
        grad1 = pi[0, 1] * pi[1, :2] * (middle_q[:2] - v1)
        grad2 = pi[0, 1] * pi[1, 1] * pi[2, :2] * (deep_rewards[:2] - v2)
        return torch.cat((grad0, grad1, grad2))

