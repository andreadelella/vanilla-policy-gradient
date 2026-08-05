"""Explicit stochastic-gradient updates used in the bandit reproduction."""

from dataclasses import dataclass
import math

import torch


ALGORITHM_KINDS = ("sgb", "entropy_sgb", "npg", "lb_sgb")


@dataclass(frozen=True)
class AlgorithmSpec:
    """A fully explicit algorithm variant and its exploration coefficient."""

    key: str
    kind: str
    learning_rate: float
    eta: float | None = None
    entropy_coefficient: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ALGORITHM_KINDS:
            raise ValueError(f"Unknown algorithm kind: {self.kind}")
        if not self.key:
            raise ValueError("Algorithm key cannot be empty")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.kind == "lb_sgb" and (self.eta is None or self.eta <= 0.0):
            raise ValueError("LB-SGB requires eta > 0")
        if self.kind == "entropy_sgb" and self.entropy_coefficient < 0.0:
            raise ValueError("entropy_coefficient cannot be negative")

    @property
    def barrier_coefficient(self) -> float:
        if self.kind != "lb_sgb":
            return 0.0
        assert self.eta is not None
        return 0.0 if math.isinf(self.eta) else 1.0 / self.eta

    def to_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "kind": self.kind,
            "learning_rate": self.learning_rate,
            "eta": self.eta,
            "entropy_coefficient": self.entropy_coefficient,
            "barrier_coefficient": self.barrier_coefficient,
        }


def recenter_logits(logits: torch.Tensor) -> torch.Tensor:
    """Fix the redundant common-logit gauge without changing the policy."""
    return logits - logits.mean(dim=-1, keepdim=True)


def policy_log_probabilities(logits: torch.Tensor) -> torch.Tensor:
    """Compute stable categorical log probabilities."""
    return torch.log_softmax(logits, dim=-1)


def raw_update_direction(
    logits: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    spec: AlgorithmSpec,
) -> torch.Tensor:
    """Return the update direction before multiplication by the learning rate."""
    log_probabilities = policy_log_probabilities(logits)
    probabilities = log_probabilities.exp()
    selected_log_probability = log_probabilities.gather(1, actions[:, None]).squeeze(1)
    selected_probability = probabilities.gather(1, actions[:, None]).squeeze(1)

    if spec.kind == "npg":
        direction = torch.zeros_like(logits)
        # Deliberately no clipping: underflow/overflow is detected by the runner.
        natural_increment = rewards / selected_probability
        direction.scatter_(1, actions[:, None], natural_increment[:, None])
        return direction

    effective_reward = rewards
    if spec.kind == "entropy_sgb":
        effective_reward = rewards - spec.entropy_coefficient * selected_log_probability

    direction = -effective_reward[:, None] * probabilities
    direction.scatter_add_(1, actions[:, None], effective_reward[:, None])

    if spec.kind == "lb_sgb":
        num_actions = logits.shape[1]
        direction = direction + spec.barrier_coefficient * (
            torch.ones_like(probabilities) - num_actions * probabilities
        )
    return direction


def apply_update(
    logits: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    spec: AlgorithmSpec,
) -> torch.Tensor:
    """Apply one explicit update and recenter its logits."""
    return recenter_logits(
        logits + spec.learning_rate * raw_update_direction(logits, actions, rewards, spec)
    )


def exact_expected_direction(
    logits: torch.Tensor,
    mean_rewards: torch.Tensor,
    spec: AlgorithmSpec,
) -> torch.Tensor:
    """Enumerate every action to obtain the exact mean update direction.

    This is intended as a small diagnostic oracle, not as the training update.
    Rows of ``mean_rewards`` correspond to rows of ``logits``.
    """
    if logits.shape != mean_rewards.shape:
        raise ValueError("logits and mean_rewards must have the same shape")
    num_runs, num_actions = logits.shape
    probabilities = policy_log_probabilities(logits).exp()
    expectation = torch.zeros_like(logits)
    for action in range(num_actions):
        actions = torch.full(
            (num_runs,), action, dtype=torch.long, device=logits.device
        )
        action_means = mean_rewards[:, action]
        direction = raw_update_direction(logits, actions, action_means, spec)
        expectation = expectation + probabilities[:, action, None] * direction
    return expectation
