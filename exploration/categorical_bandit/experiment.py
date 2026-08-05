"""Vectorized stochastic training and result serialization for Step 2."""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exploration.categorical_bandit.algorithms import (
    AlgorithmSpec,
    apply_update,
    policy_log_probabilities,
)
from exploration.categorical_bandit.environment import BanditBatch


METRIC_NAMES = (
    "minimum_probability",
    "normalized_log_fisher_volume",
    "entropy",
    "optimal_arm_probability",
    "expected_reward",
    "expected_pseudo_regret",
    "instantaneous_pseudo_regret",
    "cumulative_pseudo_regret",
    "numerically_collapsed",
)


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration for one action-count/learning-rate/algorithm unit."""

    preset: str
    num_actions: int
    num_runs: int
    horizon: int
    record_interval: int
    reward_std: float
    collapse_threshold: float
    seed: int
    algorithm: AlgorithmSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "preset": self.preset,
            "num_actions": self.num_actions,
            "num_runs": self.num_runs,
            "horizon": self.horizon,
            "record_interval": self.record_interval,
            "reward_std": self.reward_std,
            "collapse_threshold": self.collapse_threshold,
            "seed": self.seed,
            "algorithm": self.algorithm.to_dict(),
        }

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(encoded).hexdigest()


@dataclass
class TrainingResult:
    """Complete per-run trajectories for one independently saved unit."""

    config: TrainingConfig
    steps: np.ndarray
    metrics: dict[str, np.ndarray]
    final_probabilities: np.ndarray
    failed: np.ndarray
    failure_steps: np.ndarray
    spectrum_steps: np.ndarray
    positive_eigenspectra: np.ndarray

    def save(self, path: str | Path) -> None:
        """Atomically write a compressed result archive."""
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp.npz")
        arrays: dict[str, Any] = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "config_json": np.asarray(json.dumps(self.config.to_dict(), sort_keys=True)),
            "config_fingerprint": np.asarray(self.config.fingerprint),
            "steps": self.steps,
            "final_probabilities": self.final_probabilities,
            "failed": self.failed,
            "failure_steps": self.failure_steps,
            "spectrum_steps": self.spectrum_steps,
            "positive_eigenspectra": self.positive_eigenspectra,
        }
        arrays.update({f"metric_{name}": value for name, value in self.metrics.items()})
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, destination)

    @classmethod
    def load(cls, path: str | Path, config: TrainingConfig | None = None) -> "TrainingResult":
        with np.load(path, allow_pickle=False) as archive:
            saved_config = json.loads(str(archive["config_json"].item()))
            algorithm = saved_config.pop("algorithm")
            algorithm.pop("barrier_coefficient", None)
            reconstructed = TrainingConfig(
                algorithm=AlgorithmSpec(**algorithm), **saved_config
            )
            if config is not None and reconstructed.fingerprint != config.fingerprint:
                raise ValueError(f"Incompatible saved unit: {path}")
            metrics = {
                name: archive[f"metric_{name}"].copy() for name in METRIC_NAMES
            }
            return cls(
                config=reconstructed,
                steps=archive["steps"].copy(),
                metrics=metrics,
                final_probabilities=archive["final_probabilities"].copy(),
                failed=archive["failed"].copy(),
                failure_steps=archive["failure_steps"].copy(),
                spectrum_steps=archive["spectrum_steps"].copy(),
                positive_eigenspectra=archive["positive_eigenspectra"].copy(),
            )


def stable_seed(base_seed: int, *parts: object) -> int:
    """Derive a stable independent Torch seed without Python's randomized hash."""
    payload = json.dumps([int(base_seed), *parts], separators=(",", ":")).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return value % (2**63 - 1)


def recording_steps(horizon: int, interval: int) -> np.ndarray:
    if horizon < 1 or interval < 1:
        raise ValueError("horizon and interval must be positive")
    values = list(range(0, horizon + 1, interval))
    if values[-1] != horizon:
        values.append(horizon)
    return np.asarray(values, dtype=np.int64)


def decile_steps(horizon: int) -> np.ndarray:
    """Eleven unique, ordered checkpoints for representative spectra."""
    values = np.rint(np.linspace(0, horizon, 11)).astype(np.int64)
    return np.unique(values)


def _policy_metrics(
    logits: torch.Tensor,
    bandits: BanditBatch,
    cumulative_regret: torch.Tensor,
    instantaneous_regret: torch.Tensor,
    collapse_threshold: float,
) -> dict[str, torch.Tensor]:
    log_probabilities = policy_log_probabilities(logits)
    probabilities = log_probabilities.exp()
    optimal_probability = probabilities.gather(
        1, bandits.optimal_actions[:, None]
    ).squeeze(1)
    expected_reward = (probabilities * bandits.mean_rewards).sum(dim=1)
    best_reward = bandits.mean_rewards.gather(
        1, bandits.optimal_actions[:, None]
    ).squeeze(1)
    minimum_probability = probabilities.min(dim=1).values
    return {
        "minimum_probability": minimum_probability,
        "normalized_log_fisher_volume": log_probabilities.mean(dim=1)
        + np.log(logits.shape[1]),
        "entropy": -(probabilities * log_probabilities).sum(dim=1),
        "optimal_arm_probability": optimal_probability,
        "expected_reward": expected_reward,
        "expected_pseudo_regret": best_reward - expected_reward,
        "instantaneous_pseudo_regret": instantaneous_regret,
        "cumulative_pseudo_regret": cumulative_regret,
        "numerically_collapsed": minimum_probability <= collapse_threshold,
    }


def _positive_eigenspectra(logits: torch.Tensor) -> torch.Tensor:
    """Return the K-1 non-structural full-Fisher eigenvalues in ascending order."""
    probabilities = policy_log_probabilities(logits).exp()
    fisher = torch.diag_embed(probabilities) - probabilities[:, :, None] * probabilities[:, None, :]
    return torch.linalg.eigvalsh(fisher)[:, 1:]


def run_training_unit(
    config: TrainingConfig,
    bandits: BanditBatch,
    *,
    device: torch.device | str = "cpu",
    progress: bool = False,
) -> TrainingResult:
    """Run one vectorized algorithm unit and retain honest numerical failures."""
    if bandits.num_runs != config.num_runs or bandits.num_actions != config.num_actions:
        raise ValueError("Bandit batch shape is incompatible with the training config")
    resolved_device = torch.device(device)
    local_bandits = bandits.to(resolved_device)
    logits = torch.zeros(
        config.num_runs,
        config.num_actions,
        dtype=torch.float64,
        device=resolved_device,
    )
    generator = torch.Generator(device=resolved_device.type)
    generator.manual_seed(config.seed)

    steps = recording_steps(config.horizon, config.record_interval)
    step_to_index = {int(step): index for index, step in enumerate(steps)}
    metrics = {
        name: np.full((config.num_runs, len(steps)), np.nan, dtype=np.float64)
        for name in METRIC_NAMES
    }
    failed = torch.zeros(config.num_runs, dtype=torch.bool, device=resolved_device)
    failure_steps = torch.full(
        (config.num_runs,), -1, dtype=torch.long, device=resolved_device
    )
    cumulative_regret = torch.zeros(
        config.num_runs, dtype=torch.float64, device=resolved_device
    )
    instantaneous_regret = torch.zeros_like(cumulative_regret)

    spectrum_steps = decile_steps(config.horizon) if config.num_actions == 10 else np.empty(0, dtype=np.int64)
    representative_count = min(5, config.num_runs) if config.num_actions == 10 else 0
    spectra = np.full(
        (representative_count, len(spectrum_steps), max(0, config.num_actions - 1)),
        np.nan,
        dtype=np.float64,
    )
    spectrum_index = {int(step): index for index, step in enumerate(spectrum_steps)}

    def record(step: int) -> None:
        if step in step_to_index:
            values = _policy_metrics(
                logits,
                local_bandits,
                cumulative_regret,
                instantaneous_regret,
                config.collapse_threshold,
            )
            active_cpu = (~failed).detach().cpu().numpy()
            column = step_to_index[step]
            for name, value in values.items():
                data = value.to(dtype=torch.float64).detach().cpu().numpy()
                metrics[name][active_cpu, column] = data[active_cpu]
        if step in spectrum_index and representative_count:
            values = _positive_eigenspectra(logits[:representative_count])
            data = values.detach().cpu().numpy()
            rep_active = (~failed[:representative_count]).detach().cpu().numpy()
            spectra[rep_active, spectrum_index[step], :] = data[rep_active]

    record(0)
    progress_steps = set(
        int(round(config.horizon * fraction / 10.0)) for fraction in range(1, 11)
    )
    best_means = local_bandits.mean_rewards.gather(
        1, local_bandits.optimal_actions[:, None]
    ).squeeze(1)

    for step in range(1, config.horizon + 1):
        active_indices = torch.nonzero(~failed, as_tuple=False).squeeze(1)
        if active_indices.numel() > 0:
            active_logits = logits[active_indices]
            probabilities = policy_log_probabilities(active_logits).exp()
            actions = torch.multinomial(probabilities, 1, generator=generator).squeeze(1)
            selected_means = local_bandits.mean_rewards[active_indices].gather(
                1, actions[:, None]
            ).squeeze(1)
            rewards = selected_means + config.reward_std * torch.randn(
                active_indices.numel(),
                dtype=torch.float64,
                device=resolved_device,
                generator=generator,
            )
            updated = apply_update(active_logits, actions, rewards, config.algorithm)

            instantaneous_regret.zero_()
            active_regret = best_means[active_indices] - selected_means
            instantaneous_regret[active_indices] = active_regret
            cumulative_regret[active_indices] += active_regret

            finite = torch.isfinite(updated).all(dim=1)
            finite_indices = active_indices[finite]
            failed_indices = active_indices[~finite]
            if finite_indices.numel() > 0:
                logits[finite_indices] = updated[finite]
            if failed_indices.numel() > 0:
                failed[failed_indices] = True
                failure_steps[failed_indices] = step

        record(step)
        if progress and step in progress_steps:
            print(
                f"  {config.algorithm.key}: {100 * step // config.horizon:3d}% "
                f"({int(failed.sum().item())}/{config.num_runs} failed)",
                flush=True,
            )

    final_log_probabilities = policy_log_probabilities(logits)
    final_probabilities = final_log_probabilities.exp().detach().cpu().numpy()
    failed_cpu = failed.detach().cpu().numpy()
    final_probabilities[failed_cpu] = np.nan
    return TrainingResult(
        config=config,
        steps=steps,
        metrics=metrics,
        final_probabilities=final_probabilities,
        failed=failed_cpu,
        failure_steps=failure_steps.detach().cpu().numpy(),
        spectrum_steps=spectrum_steps,
        positive_eigenspectra=spectra,
    )
