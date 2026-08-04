"""Exact deterministic two-step trap MDP and reduced softmax policy."""

from __future__ import annotations

from dataclasses import dataclass

import torch


DTYPE = torch.float64
DEVICE = torch.device("cpu")


def as_phi(phi, *, requires_grad: bool = False) -> torch.Tensor:
    """Validate reduced policy parameters with final dimension four."""
    value = torch.as_tensor(phi, dtype=DTYPE, device=DEVICE)
    if value.ndim not in (1, 2) or value.shape[-1] != 4:
        raise ValueError("phi must have shape (4,) or (batch, 4)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("phi must contain only finite values")
    return value.detach().clone().requires_grad_(requires_grad)


def probabilities_from_reduced_logits(phi) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the two three-action policies; action 2 is the reference."""
    value = torch.as_tensor(phi, dtype=DTYPE, device=DEVICE)
    if value.ndim not in (1, 2) or value.shape[-1] != 4:
        raise ValueError("phi must have shape (4,) or (batch, 4)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError("phi must contain only finite values")
    zeros = torch.zeros_like(value[..., :1])
    pi0 = torch.softmax(torch.cat((value[..., :2], zeros), dim=-1), dim=-1)
    pi1 = torch.softmax(torch.cat((value[..., 2:], zeros), dim=-1), dim=-1)
    return pi0, pi1


def reduced_logits_from_probabilities(probabilities) -> torch.Tensor:
    """Construct stable reduced logits from a strictly interior policy."""
    value = torch.as_tensor(probabilities, dtype=DTYPE, device=DEVICE)
    if value.ndim not in (1, 2) or value.shape[-1] != 3:
        raise ValueError("probabilities must have shape (3,) or (batch, 3)")
    if not bool(torch.isfinite(value).all()) or not bool((value > 0).all()):
        raise ValueError("probabilities must be finite and strictly positive")
    if not bool(torch.allclose(value.sum(dim=-1), torch.ones_like(value[..., 0]), atol=1e-12, rtol=1e-12)):
        raise ValueError("probabilities must sum to one")
    return torch.log(value[..., :2]) - torch.log(value[..., 2:3])


def phi_from_q_and_good(q, p_good) -> torch.Tensor:
    """Build grid policies using the declared equal remaining-mass split."""
    q_value, good_value = torch.broadcast_tensors(
        torch.as_tensor(q, dtype=DTYPE, device=DEVICE),
        torch.as_tensor(p_good, dtype=DTYPE, device=DEVICE),
    )
    if not bool(((q_value > 0) & (q_value < 1)).all()):
        raise ValueError("q must lie strictly between zero and one")
    if not bool(((good_value > 0) & (good_value < 1)).all()):
        raise ValueError("p_good must lie strictly between zero and one")
    pi0 = torch.stack((0.5 * (1 - q_value), q_value, 0.5 * (1 - q_value)), dim=-1)
    pi1 = torch.stack((good_value, 0.5 * (1 - good_value), 0.5 * (1 - good_value)), dim=-1)
    return torch.cat((reduced_logits_from_probabilities(pi0), reduced_logits_from_probabilities(pi1)), dim=-1)


def transition_pool_weights(phi) -> tuple[torch.Tensor, torch.Tensor]:
    """Population weights from uniformly selecting a pooled valid transition."""
    pi0, _ = probabilities_from_reduced_logits(phi)
    q = pi0[..., 1]
    return 1.0 / (1.0 + q), q / (1.0 + q)


@dataclass(frozen=True)
class Trajectory:
    probability: torch.Tensor
    reward: float
    states: tuple[int, ...]
    actions: tuple[int, ...]


@dataclass(frozen=True)
class TwoStepTrap:
    """The exact MDP used throughout Step 3."""

    safe_reward: float = 0.5
    state1_rewards: tuple[float, float, float] = (1.0, 0.2, 0.0)
    horizon: int = 2
    gamma: float = 1.0

    def trajectories(self, phi) -> tuple[Trajectory, ...]:
        pi0, pi1 = probabilities_from_reduced_logits(phi)
        if pi0.ndim != 1:
            raise ValueError("trajectory enumeration requires one policy")
        paths = [
            Trajectory(pi0[0], self.safe_reward, (0,), (0,)),
            Trajectory(pi0[2], 0.0, (0,), (2,)),
        ]
        for action, reward in enumerate(self.state1_rewards):
            paths.append(Trajectory(pi0[1] * pi1[action], reward, (0, 1), (1, action)))
        return tuple(paths)

    def exact_return(self, phi) -> torch.Tensor:
        pi0, pi1 = probabilities_from_reduced_logits(phi)
        rewards1 = torch.tensor(self.state1_rewards, dtype=DTYPE, device=DEVICE)
        return self.safe_reward * pi0[..., 0] + pi0[..., 1] * (pi1 * rewards1).sum(dim=-1)

    def enumerated_return(self, phi) -> torch.Tensor:
        return sum(path.probability * path.reward for path in self.trajectories(phi))

    def exact_reward_gradient(self, phi) -> torch.Tensor:
        """Closed-form policy gradient in the four reduced coordinates."""
        pi0, pi1 = probabilities_from_reduced_logits(phi)
        rewards1 = torch.tensor(self.state1_rewards, dtype=DTYPE, device=DEVICE)
        value1 = (pi1 * rewards1).sum(dim=-1)
        q_values = torch.stack(
            (torch.full_like(value1, self.safe_reward), value1, torch.zeros_like(value1)),
            dim=-1,
        )
        value0 = (pi0 * q_values).sum(dim=-1)
        grad0 = pi0[..., :2] * (q_values[..., :2] - value0.unsqueeze(-1))
        grad1 = pi0[..., 1:2] * pi1[..., :2] * (rewards1[:2] - value1.unsqueeze(-1))
        return torch.cat((grad0, grad1), dim=-1)
