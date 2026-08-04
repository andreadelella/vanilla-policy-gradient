"""Estimator audits and sampled exact-tabular training for Step 4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from exploration.tabular_mdp.geometry import barrier_gradients, geometry_snapshot, pooled_fisher
from exploration.tabular_mdp.model import DTYPE, TwoStepTrap, as_phi, phi_from_q_and_good, probabilities_from_reduced_logits

from .estimators import (
    exact_finite_batch_moments,
    population_targets,
    sampled_conditional_gradient,
    sampled_empirical_fisher,
    sampled_reward_gradient,
)
from .sampling import sample_batch


METHODS = (
    "reward_only",
    "detached_conditional_oracle",
    "complete_weighted_oracle",
    "uniform_action_oracle",
    "visitation_only_oracle",
    "full_pooled_fisher_oracle",
    "detached_conditional_sampled",
)


INITIALIZATIONS = {
    "uniform": (0.0, 0.0, 0.0, 0.0),
    "adverse": (2.0, -2.0, -2.0, 2.0),
}


AUDIT_POLICIES = {
    "uniform": ("phi", (0.0, 0.0, 0.0, 0.0)),
    "adverse": ("phi", (2.0, -2.0, -2.0, 2.0)),
    "rare_good": ("probability", (0.02, 0.9)),
    "common_bad": ("probability", (0.9, 0.02)),
    "near_optimal": ("probability", (0.9, 0.9)),
}


def audit_policy(name: str) -> torch.Tensor:
    try:
        kind, values = AUDIT_POLICIES[name]
    except KeyError as error:
        raise ValueError(f"unknown audit policy: {name}") from error
    if kind == "phi":
        return as_phi(values)
    return phi_from_q_and_good(*values)


@dataclass(frozen=True)
class EstimatorAuditConfig:
    policy_names: tuple[str, ...] = tuple(AUDIT_POLICIES)
    batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64, 128, 256)
    repetitions: int = 50_000
    seed: int = 23
    chunk_size: int = 1_000

    def __post_init__(self) -> None:
        if not self.policy_names or any(name not in AUDIT_POLICIES for name in self.policy_names):
            raise ValueError("policy_names contains an unknown policy")
        if not self.batch_sizes or any(size < 1 for size in self.batch_sizes):
            raise ValueError("batch_sizes must be positive")
        if self.repetitions < 1 or self.chunk_size < 1:
            raise ValueError("repetitions and chunk_size must be positive")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["policy_names"] = list(self.policy_names)
        value["batch_sizes"] = list(self.batch_sizes)
        return value


@dataclass(frozen=True)
class EstimatorAuditResult:
    config: EstimatorAuditConfig
    rows: tuple[dict[str, Any], ...]

    def save_json(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps({"schema_version": 1, "config": self.config.to_dict(), "rows": self.rows}, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @classmethod
    def load_json(cls, path: str | Path, expected: EstimatorAuditConfig) -> "EstimatorAuditResult":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("config") != expected.to_dict():
            raise ValueError(f"audit has incompatible configuration: {path}")
        return cls(expected, tuple(payload["rows"]))


def _norm(vector: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(vector, dim=-1)


def _cosine(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    nl, nr = _norm(left), _norm(right)
    result = torch.full_like(nl, torch.nan)
    valid = (nl > 0) & (nr > 0)
    result[valid] = (left[valid] * right[valid]).sum(dim=-1) / (nl[valid] * nr[valid])
    return result


def _numerical_rank(eigenvalues: torch.Tensor) -> torch.Tensor:
    threshold = 1e-12 * torch.maximum(torch.ones_like(eigenvalues[..., -1]), eigenvalues[..., -1])
    return (eigenvalues > threshold[..., None]).sum(dim=-1)


def run_estimator_audit(config: EstimatorAuditConfig) -> EstimatorAuditResult:
    rows: list[dict[str, Any]] = []
    for policy_index, policy_name in enumerate(config.policy_names):
        phi = audit_policy(policy_name)
        pi0, pi1 = probabilities_from_reduced_logits(phi)
        targets = population_targets(phi)
        target_cond = targets["conditional_gradient"]
        target_reward = targets["reward_gradient"]
        target_fisher = targets["pooled_fisher"]
        for n in config.batch_sizes:
            exact = exact_finite_batch_moments(phi, n)
            generator = torch.Generator(device="cpu").manual_seed(config.seed + 10_000 * policy_index + n)
            count = 0
            mu1_sum = mu1_square_sum = zero_sum = 0.0
            cond_sum = torch.zeros(4, dtype=DTYPE)
            cond_error_square_sum = 0.0
            reward_sum = torch.zeros(4, dtype=DTYPE)
            reward_error_square_sum = 0.0
            fisher_sum = torch.zeros((4, 4), dtype=DTYPE)
            fisher_error_square_sum = 0.0
            rank_sum = full_rank_sum = defined_sum = 0.0
            min_eigenvalue_sum = 0.0
            logdet_sum = 0.0
            while count < config.repetitions:
                take = min(config.chunk_size, config.repetitions - count)
                phis = phi.expand(take, 4)
                batch = sample_batch(phis, n, generator=generator)
                cond = sampled_conditional_gradient(phis, batch)
                reward = sampled_reward_gradient(phis, batch)
                fisher = sampled_empirical_fisher(phis, batch)
                mu1 = batch.k1.to(DTYPE) / batch.m.to(DTYPE)
                eig = torch.linalg.eigvalsh(fisher)
                rank = _numerical_rank(eig)
                sign, logabsdet = torch.linalg.slogdet(fisher)
                defined = (sign > 0) & (rank == 4)
                mu1_sum += float(mu1.sum())
                mu1_square_sum += float(mu1.square().sum())
                zero_sum += float((batch.k1 == 0).sum())
                cond_sum += cond.sum(dim=0)
                cond_error_square_sum += float((cond - target_cond).square().sum())
                reward_sum += reward.sum(dim=0)
                reward_error_square_sum += float((reward - target_reward).square().sum())
                fisher_sum += fisher.sum(dim=0)
                fisher_error_square_sum += float((fisher - target_fisher).square().sum())
                rank_sum += float(rank.sum())
                full_rank_sum += float((rank == 4).sum())
                defined_sum += float(defined.sum())
                min_eigenvalue_sum += float(eig[..., 0].sum())
                logdet_sum += float(logabsdet[defined].sum())
                count += take

            repetitions = float(config.repetitions)
            mu1_mc = mu1_sum / repetitions
            cond_mc = cond_sum / repetitions
            reward_mc = reward_sum / repetitions
            fisher_mc = fisher_sum / repetitions
            cond_target_norm = float(_norm(target_cond))
            cosine = float("nan")
            if cond_target_norm > 0 and float(_norm(cond_mc)) > 0:
                cosine = float(torch.dot(cond_mc, target_cond) / (_norm(cond_mc) * _norm(target_cond)))
            rows.append(
                {
                    "policy": policy_name,
                    "n": n,
                    "repetitions": config.repetitions,
                    "q": float(pi0[1]),
                    "p_good": float(pi1[0]),
                    "population_mu1": float(pi0[1] / (1 + pi0[1])),
                    "exact_mu1_mean": float(exact.mu1_mean),
                    "exact_mu1_bias": float(exact.mu1_mean - pi0[1] / (1 + pi0[1])),
                    "exact_mu1_variance": float(exact.mu1_variance),
                    "mc_mu1_mean": mu1_mc,
                    "mc_mu1_variance": mu1_square_sum / repetitions - mu1_mc * mu1_mc,
                    "zero_s1_probability": float(exact.zero_s1_probability),
                    "mc_zero_s1_fraction": zero_sum / repetitions,
                    "conditional_exact_bias_norm": float(_norm(exact.conditional_mean - target_cond)),
                    "conditional_exact_sd_norm": float(torch.sqrt(torch.trace(exact.conditional_covariance))),
                    "conditional_mc_bias_norm": float(_norm(cond_mc - target_cond)),
                    "conditional_mc_rmse": math.sqrt(cond_error_square_sum / repetitions),
                    "conditional_mc_mean_cosine": cosine,
                    "reward_exact_bias_norm": float(_norm(exact.reward_mean - target_reward)),
                    "reward_mc_bias_norm": float(_norm(reward_mc - target_reward)),
                    "reward_mc_rmse": math.sqrt(reward_error_square_sum / repetitions),
                    "fisher_exact_bias_fro": float(torch.linalg.matrix_norm(exact.fisher_mean - target_fisher)),
                    "fisher_mc_bias_fro": float(torch.linalg.matrix_norm(fisher_mc - target_fisher)),
                    "fisher_mc_rmse_fro": math.sqrt(fisher_error_square_sum / repetitions),
                    "fisher_rank_mean": rank_sum / repetitions,
                    "fisher_full_rank_fraction": full_rank_sum / repetitions,
                    "fisher_logdet_defined_fraction": defined_sum / repetitions,
                    "fisher_min_eigenvalue_mean": min_eigenvalue_sum / repetitions,
                    "fisher_logdet_mean_when_defined": logdet_sum / defined_sum if defined_sum else float("nan"),
                }
            )
    return EstimatorAuditResult(config, tuple(rows))


@dataclass(frozen=True)
class SampledTrainingConfig:
    method: str
    initialization: str
    n_trajectories: int
    n_seeds: int
    alpha: float = 0.05
    beta: float = 0.1
    beta_after: float | None = None
    handoff_update: int | None = None
    updates: int = 2000
    record_interval: int = 10
    base_seed: int = 23
    reward_noise_std: float = 0.0
    center_returns: bool = False
    normalize_returns: bool = False
    label: str = "main"

    def __post_init__(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method: {self.method}")
        if self.initialization not in INITIALIZATIONS:
            raise ValueError(f"unknown initialization: {self.initialization}")
        if self.n_trajectories < 1 or self.n_seeds < 1 or self.updates < 1:
            raise ValueError("counts must be positive")
        if self.alpha <= 0 or not math.isfinite(self.alpha):
            raise ValueError("alpha must be finite and positive")
        if self.beta < 0 or not math.isfinite(self.beta):
            raise ValueError("beta must be finite and nonnegative")
        if (self.beta_after is None) != (self.handoff_update is None):
            raise ValueError("beta_after and handoff_update must be specified together")
        if self.beta_after is not None:
            if self.beta_after < 0 or not math.isfinite(self.beta_after):
                raise ValueError("beta_after must be finite and nonnegative")
            if not 0 < self.handoff_update < self.updates:
                raise ValueError("handoff_update must be strictly between zero and updates")
        if self.record_interval < 1:
            raise ValueError("record_interval must be positive")
        if self.reward_noise_std < 0 or not math.isfinite(self.reward_noise_std):
            raise ValueError("reward_noise_std must be finite and nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def beta_for_step(self, step: int) -> float:
        """Return beta for a one-based optimizer step.

        The scientific schedule is indexed by zero-based update t=step-1.
        Thus handoff_update=2000 applies the initial beta to exactly the first
        2000 updates and beta_after from optimizer step 2001 onward.
        """
        if not 1 <= step <= self.updates:
            raise ValueError("step must be between one and updates")
        if self.handoff_update is None or (step - 1) < self.handoff_update:
            return self.beta
        return self.beta_after


@dataclass
class SampledTrainingResult:
    config: SampledTrainingConfig
    steps: np.ndarray
    phi: np.ndarray
    metrics: dict[str, np.ndarray]
    finite: np.ndarray

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + ".tmp.npz")
        arrays: dict[str, Any] = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "config_json": np.asarray(json.dumps(self.config.to_dict(), sort_keys=True)),
            "steps": self.steps,
            "phi": self.phi,
            "finite": self.finite,
        }
        arrays.update({f"metric__{key}": value for key, value in self.metrics.items()})
        np.savez_compressed(temporary, **arrays)
        temporary.replace(destination)

    @classmethod
    def load(cls, path: str | Path, expected: SampledTrainingConfig) -> "SampledTrainingResult":
        with np.load(path, allow_pickle=False) as archive:
            saved = json.loads(str(archive["config_json"].item()))
            # Schema-v1 archives predate scheduled coefficients. Missing keys
            # mean the original constant-beta behavior.
            saved.setdefault("beta_after", None)
            saved.setdefault("handoff_update", None)
            if saved != expected.to_dict():
                raise ValueError(f"result has incompatible configuration: {path}")
            metrics = {key.removeprefix("metric__"): archive[key].copy() for key in archive.files if key.startswith("metric__")}
            # Schema-v1 archives created before the declared numerical-rank
            # gate used only the slogdet sign. Reconstruct the scientifically
            # intended derived fields from the stored rank without resampling.
            if "empirical_fisher_rank" in metrics:
                defined = (metrics["empirical_fisher_rank"] == 4) & np.isfinite(
                    metrics["empirical_fisher_logdet"]
                )
                metrics["empirical_fisher_logdet_defined"] = defined
                metrics["empirical_fisher_logdet"] = np.where(
                    defined, metrics["empirical_fisher_logdet"], np.nan
                )
            return cls(expected, archive["steps"].copy(), archive["phi"].copy(), metrics, archive["finite"].copy())


def _regularizer_gradient(phi: torch.Tensor, method: str, batch) -> tuple[torch.Tensor, torch.Tensor]:
    exact = barrier_gradients(phi)
    mapping = {
        "reward_only": torch.zeros_like(phi),
        "detached_conditional_oracle": exact.detached_conditional,
        "complete_weighted_oracle": exact.complete_weighted,
        "uniform_action_oracle": exact.uniform_action,
        "visitation_only_oracle": exact.visitation_only,
        "full_pooled_fisher_oracle": exact.full_pooled_fisher,
        "detached_conditional_sampled": sampled_conditional_gradient(phi, batch),
    }
    selected = mapping[method]
    target = exact.detached_conditional if method == "detached_conditional_sampled" else selected
    return selected, target


def _record_population(phi: torch.Tensor, mdp: TwoStepTrap) -> dict[str, torch.Tensor]:
    values = dict(geometry_snapshot(phi).values)
    values["population_return"] = mdp.exact_return(phi)
    return values


def _record_update(
    phi_before: torch.Tensor,
    batch,
    sampled_reward: torch.Tensor,
    exact_reward: torch.Tensor,
    selected_regularizer: torch.Tensor,
    target_regularizer: torch.Tensor,
    total_gradient: torch.Tensor,
    effective_beta: float,
) -> dict[str, torch.Tensor]:
    empirical_fisher = sampled_empirical_fisher(phi_before, batch)
    eig = torch.linalg.eigvalsh(empirical_fisher)
    rank = _numerical_rank(eig)
    sign, logabsdet = torch.linalg.slogdet(empirical_fisher)
    defined = (sign > 0) & (rank == 4)
    logdet = torch.full_like(logabsdet, torch.nan)
    logdet[defined] = logabsdet[defined]
    sampled_cond = sampled_conditional_gradient(phi_before, batch)
    exact_cond = barrier_gradients(phi_before).detached_conditional
    sampled_return = batch.rewards.sum(dim=-1).mean(dim=-1)
    result = {
        "sampled_batch_return": sampled_return,
        "k1": batch.k1.to(DTYPE),
        "m": batch.m.to(DTYPE),
        "mu0_hat": batch.n_trajectories / batch.m.to(DTYPE),
        "mu1_hat": batch.k1.to(DTYPE) / batch.m.to(DTYPE),
        "zero_s1": batch.k1 == 0,
        "empirical_fisher_rank": rank.to(DTYPE),
        "empirical_fisher_min_eigenvalue": eig[..., 0],
        "empirical_fisher_logdet": logdet,
        "empirical_fisher_logdet_defined": defined,
        "empirical_fisher_fro_error": torch.linalg.matrix_norm(empirical_fisher - pooled_fisher(phi_before), dim=(-2, -1)),
        "sampled_reward_gradient_norm": _norm(sampled_reward),
        "exact_reward_gradient_norm": _norm(exact_reward),
        "sampled_reward_gradient_error": _norm(sampled_reward - exact_reward),
        "sampled_reward_gradient_cosine": _cosine(sampled_reward, exact_reward),
        "selected_regularizer_norm": _norm(selected_regularizer),
        "target_regularizer_norm": _norm(target_regularizer),
        "regularizer_gradient_error": _norm(selected_regularizer - target_regularizer),
        "regularizer_gradient_cosine": _cosine(selected_regularizer, target_regularizer),
        "sampled_conditional_error": _norm(sampled_cond - exact_cond),
        "sampled_conditional_cosine": _cosine(sampled_cond, exact_cond),
        "total_update_gradient_norm": _norm(total_gradient),
        "effective_beta": torch.full_like(sampled_return, effective_beta),
    }
    return result


def train_sampled(config: SampledTrainingConfig, *, mdp: TwoStepTrap | None = None) -> SampledTrainingResult:
    mdp = mdp or TwoStepTrap()
    initial = torch.tensor(INITIALIZATIONS[config.initialization], dtype=DTYPE)
    phi = initial.expand(config.n_seeds, 4).clone()
    active = torch.ones(config.n_seeds, dtype=torch.bool)
    generator = torch.Generator(device="cpu").manual_seed(config.base_seed)
    record_steps = sorted(set(range(0, config.updates + 1, config.record_interval)) | {config.updates})
    step_to_index = {step: index for index, step in enumerate(record_steps)}
    population0 = _record_population(phi, mdp)
    update_keys = (
        "sampled_batch_return", "k1", "m", "mu0_hat", "mu1_hat", "zero_s1",
        "empirical_fisher_rank", "empirical_fisher_min_eigenvalue", "empirical_fisher_logdet",
        "empirical_fisher_logdet_defined", "empirical_fisher_fro_error",
        "sampled_reward_gradient_norm", "exact_reward_gradient_norm", "sampled_reward_gradient_error",
        "sampled_reward_gradient_cosine", "selected_regularizer_norm", "target_regularizer_norm",
        "regularizer_gradient_error", "regularizer_gradient_cosine", "sampled_conditional_error",
        "sampled_conditional_cosine", "total_update_gradient_norm",
        "effective_beta",
    )
    metrics: dict[str, np.ndarray] = {
        key: np.full((len(record_steps), config.n_seeds), np.nan, dtype=np.bool_ if value.dtype == torch.bool else np.float64)
        for key, value in population0.items()
    }
    for key in update_keys:
        bool_key = key in {"zero_s1", "empirical_fisher_logdet_defined"}
        metrics[key] = np.full((len(record_steps), config.n_seeds), False if bool_key else np.nan, dtype=np.bool_ if bool_key else np.float64)
    phi_history = np.full((len(record_steps), config.n_seeds, 4), np.nan, dtype=np.float64)

    def record_population(index: int, values: dict[str, torch.Tensor]) -> None:
        phi_history[index] = phi.detach().numpy()
        for key, value in values.items():
            metrics[key][index] = value.detach().numpy()

    record_population(0, population0)
    for step in range(1, config.updates + 1):
        sample_phi = torch.where(active[:, None], phi, torch.zeros_like(phi))
        batch = sample_batch(
            sample_phi,
            config.n_trajectories,
            generator=generator,
            reward_noise_std=config.reward_noise_std,
            mdp=mdp,
        )
        reward = sampled_reward_gradient(
            sample_phi,
            batch,
            center_returns=config.center_returns,
            normalize_returns=config.normalize_returns,
        )
        exact_reward = mdp.exact_reward_gradient(sample_phi)
        regularizer, target_regularizer = _regularizer_gradient(sample_phi, config.method, batch)
        effective_beta = config.beta_for_step(step)
        total = reward + effective_beta * regularizer
        candidate = phi + config.alpha * total
        newly_finite = torch.isfinite(candidate).all(dim=-1)
        active &= newly_finite
        phi = torch.where(active[:, None], candidate, torch.full_like(candidate, torch.nan))
        if step in step_to_index:
            index = step_to_index[step]
            update_values = _record_update(
                sample_phi, batch, reward, exact_reward, regularizer, target_regularizer, total,
                effective_beta,
            )
            population = _record_population(torch.where(active[:, None], phi, torch.zeros_like(phi)), mdp)
            for key, value in population.items():
                if value.dtype != torch.bool:
                    value = torch.where(active, value, torch.full_like(value, torch.nan))
                population[key] = value
            record_population(index, population)
            for key, value in update_values.items():
                metrics[key][index] = value.detach().numpy()
    return SampledTrainingResult(
        config,
        np.asarray(record_steps, dtype=np.int64),
        phi_history,
        metrics,
        active.detach().numpy(),
    )
