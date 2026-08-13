"""Training diagnostics that are independent of rollout orchestration."""

import csv
import os

import torch
from torch.distributions import kl_divergence


METRIC_FIELDS = (
    "iteration",
    "train_reward",
    "best_reward",
    "kl",
    "grad_norm",
    "nat_grad_norm",
    "entropy",
    "mean_std",
    "return_mean",
    "return_std",
    "episode_len",
    "rollout_time",
    "update_time",
    "total_time",
)


def gradient_norm(policy) -> float:
    gradients = [
        parameter.grad.reshape(-1)
        for parameter in policy.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return float("nan")
    return float(torch.cat(gradients).norm())


def freeze_distribution(policy, states):
    """Detach a distribution snapshot for post-update KL measurement."""

    with torch.no_grad():
        distribution = policy.distribution(states)
        if hasattr(distribution, "scale"):
            return type(distribution)(
                distribution.loc.clone(), distribution.scale.clone()
            )
        if hasattr(distribution, "logits"):
            return type(distribution)(logits=distribution.logits.clone())
    return None


def policy_stats(policy, states) -> tuple[float, float]:
    with torch.no_grad():
        distribution = policy.distribution(states)
        entropy = distribution.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(-1)
        scale = getattr(distribution, "scale", None)
        mean_std = float(scale.mean()) if scale is not None else float("nan")
        return float(entropy.mean()), mean_std


def measure_kl(policy, states, previous_distribution) -> float:
    if previous_distribution is None:
        return float("nan")
    with torch.no_grad():
        divergence = kl_divergence(
            previous_distribution,
            policy.distribution(states),
        )
        if divergence.dim() > 1:
            divergence = divergence.sum(-1)
        return float(divergence.mean())


def append_metrics_row(path, row) -> None:
    """Append one diagnostics row without buffering the entire training run."""

    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
