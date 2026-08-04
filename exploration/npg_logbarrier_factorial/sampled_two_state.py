"""Sampled two-state Euclidean/natural × temporary barrier factorial."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from vpg.stats import mean_confidence_interval

from exploration.sampled_tabular_mdp.estimators import (
    sampled_conditional_gradient,
    sampled_empirical_fisher,
    sampled_reward_gradient,
)
from exploration.sampled_tabular_mdp.sampling import sample_batch
from exploration.tabular_mdp.geometry import barrier_gradients, pooled_fisher
from exploration.tabular_mdp.model import (
    DTYPE,
    TwoStepTrap,
    as_phi,
    probabilities_from_reduced_logits,
    transition_pool_weights,
)

from .natural_step import cosine, target_kl_natural_step


METHODS = (
    "sampled_pg_reward_only",
    "sampled_pg_logbarrier_handoff",
    "sampled_npg_reward_only",
    "sampled_npg_logbarrier_handoff",
    "sampled_pg_logbarrier_fixed",
    "sampled_npg_logbarrier_fixed",
)
INITIALIZATIONS = {
    "uniform": (0.0, 0.0, 0.0, 0.0),
    "adverse": (2.0, -2.0, -2.0, 2.0),
}


@dataclass(frozen=True)
class SampledFactorialConfig:
    method: str
    initialization: str
    n_trajectories: int
    n_seeds: int
    updates: int = 4000
    alpha: float = 0.05
    beta: float = 0.2
    handoff_update: int = 2000
    damping: float = 0.01
    target_kl: float = 1e-3
    record_interval: int = 10
    base_seed: int = 91_000

    def validate(self) -> None:
        if self.method not in METHODS or self.initialization not in INITIALIZATIONS:
            raise ValueError("unknown sampled method or initialization")
        if min(self.n_trajectories, self.n_seeds, self.updates, self.record_interval) < 1:
            raise ValueError("counts must be positive")
        if self.alpha <= 0 or self.beta < 0 or self.damping < 0 or self.target_kl <= 0:
            raise ValueError("invalid numerical setting")
        if not 0 < self.handoff_update < self.updates:
            raise ValueError("handoff must lie inside the update horizon")

    @property
    def natural(self) -> bool:
        return "_npg_" in self.method

    @property
    def barrier(self) -> bool:
        return "logbarrier" in self.method

    @property
    def fixed(self) -> bool:
        return self.method.endswith("_fixed")

    def beta_at(self, update: int) -> float:
        if not self.barrier:
            return 0.0
        return self.beta if self.fixed or update < self.handoff_update else 0.0


def _mean_forward_kl(old_phi, new_phi) -> float:
    old0, old1 = probabilities_from_reduced_logits(old_phi)
    new0, new1 = probabilities_from_reduced_logits(new_phi)
    mu0, mu1 = transition_pool_weights(old_phi)
    return float(
        mu0 * (old0 * (torch.log(old0) - torch.log(new0))).sum()
        + mu1 * (old1 * (torch.log(old1) - torch.log(new1))).sum()
    )


def _rank(matrix: torch.Tensor) -> int:
    eigenvalues = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    threshold = 1e-12 * max(1.0, float(eigenvalues[-1]))
    return int((eigenvalues > threshold).sum())


def _row(
    config, seed_index, update, phi, gradient_phi, batch, reward, barrier, fisher,
    total, beta, realized_kl, natural_result,
):
    mdp = TwoStepTrap()
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    v1 = float((pi1 * torch.tensor(mdp.state1_rewards, dtype=DTYPE)).sum())
    exact_reward = mdp.exact_reward_gradient(gradient_phi)
    exact_barrier = barrier_gradients(gradient_phi).detached_conditional
    damped = fisher + config.damping * torch.eye(4, dtype=DTYPE)
    natural_reward = torch.linalg.solve(damped, reward)
    natural_barrier = torch.linalg.solve(damped, beta * barrier)
    cos_e, cos_e_defined = cosine(reward, barrier)
    cos_n, cos_n_defined = cosine(natural_reward, natural_barrier)
    reward_norm = float(reward.norm())
    natural_reward_norm = float(natural_reward.norm())
    row = {
        "method": config.method,
        "initialization": config.initialization,
        "n_trajectories": config.n_trajectories,
        "seed": config.base_seed + seed_index,
        "update": update,
        "beta": beta,
        "barrier_active": beta > 0.0,
        "return": float(mdp.exact_return(phi)),
        "q": float(pi0[1]),
        "pi1_good": float(pi1[0]),
        "v1": v1,
        "delta_safe": v1 - mdp.safe_reward,
        "k1": int(batch.k1),
        "valid_transition_count": int(batch.m),
        "zero_s1": int(batch.k1) == 0,
        "s0_coordinates_observed": True,
        "s1_coordinates_observed": int(batch.k1) > 0,
        "undamped_fisher_rank": _rank(fisher),
        "undamped_fisher_minimum_eigenvalue": float(torch.linalg.eigvalsh(fisher)[0]),
        "reward_gradient_norm": reward_norm,
        "barrier_gradient_norm": float(barrier.norm()),
        "total_gradient_norm": float(total.norm()),
        "reward_gradient_error_from_population": float((reward - exact_reward).norm()),
        "barrier_gradient_error_from_population": float((barrier - exact_barrier).norm()),
        "naturalized_reward_norm": natural_reward_norm,
        "naturalized_barrier_norm": float(natural_barrier.norm()),
        "euclidean_barrier_to_reward_ratio": float((beta * barrier).norm()) / reward_norm if reward_norm else float("nan"),
        "natural_barrier_to_reward_ratio": float(natural_barrier.norm()) / natural_reward_norm if natural_reward_norm else float("nan"),
        "euclidean_cosine": cos_e,
        "euclidean_cosine_defined": cos_e_defined,
        "natural_cosine": cos_n,
        "natural_cosine_defined": cos_n_defined,
        "realized_kl": realized_kl,
        "invalid_solve": bool(natural_result is not None and not natural_result.valid),
        "invalid_reason": natural_result.invalid_reason if natural_result is not None else "",
    }
    if natural_result is not None:
        row.update(natural_result.diagnostics())
    return row


def run_one(config: SampledFactorialConfig) -> tuple[list[dict], list[dict], list[dict]]:
    config.validate()
    mdp = TwoStepTrap()
    initial = as_phi(INITIALIZATIONS[config.initialization])
    checkpoints, endpoints, missing = [], [], []
    for seed_index in range(config.n_seeds):
        phi = initial.clone()
        generator = torch.Generator(device="cpu").manual_seed(config.base_seed + seed_index)
        finite = True
        invalid_count = 0
        zero_count = 0
        first_positive = None
        attempted_updates = 0
        last_row = None
        for update in range(config.updates):
            attempted_updates += 1
            phi_before = phi.clone()
            batch = sample_batch(phi, config.n_trajectories, generator=generator, mdp=mdp)
            reward = sampled_reward_gradient(phi, batch, center_returns=False, normalize_returns=False)
            barrier = sampled_conditional_gradient(phi, batch) if config.barrier else torch.zeros(4, dtype=DTYPE)
            beta = config.beta_at(update)
            total = reward + beta * barrier
            fisher = sampled_empirical_fisher(phi, batch)
            natural_result = None
            if config.natural:
                natural_result = target_kl_natural_step(
                    total, fisher, damping=config.damping, target_kl=config.target_kl
                )
                if not natural_result.valid:
                    invalid_count += 1
                    finite = False
                    break
                step = natural_result.step
            else:
                step = config.alpha * total
            new_phi = phi + step
            realized = _mean_forward_kl(phi, new_phi)
            zero_count += int(batch.k1) == 0
            pi1_new = probabilities_from_reduced_logits(new_phi)[1]
            v1_new = float((pi1_new * torch.tensor(mdp.state1_rewards, dtype=DTYPE)).sum())
            if first_positive is None and v1_new > mdp.safe_reward:
                first_positive = update + 1
            phi = new_phi
            completed = update + 1
            if completed % config.record_interval == 0 or completed in {
                config.handoff_update - 1, config.handoff_update,
                config.handoff_update + 1, config.updates,
            }:
                last_row = _row(
                    config, seed_index, completed, phi, phi_before, batch, reward, barrier,
                    fisher, total, beta, realized, natural_result,
                )
                checkpoints.append(last_row)
                missing.append({
                    "method": config.method,
                    "initialization": config.initialization,
                    "n_trajectories": config.n_trajectories,
                    "seed": config.base_seed + seed_index,
                    "update": completed,
                    "k1": int(batch.k1),
                    "zero_s1": int(batch.k1) == 0,
                    "undamped_rank": _rank(fisher),
                    "s1_reward_gradient_norm": float(reward[2:].norm()),
                    "s1_barrier_gradient_norm": float(barrier[2:].norm()),
                    "s1_direct_update_abs_max": float(step[2:].abs().max()),
                })
            if not torch.isfinite(phi).all():
                finite = False
                break
        if last_row is None:
            pi0, pi1 = probabilities_from_reduced_logits(phi)
            final_return = float(mdp.exact_return(phi))
            final_q, final_good = float(pi0[1]), float(pi1[0])
        else:
            final_return, final_q, final_good = last_row["return"], last_row["q"], last_row["pi1_good"]
        endpoints.append({
            "method": config.method,
            "initialization": config.initialization,
            "n_trajectories": config.n_trajectories,
            "seed": config.base_seed + seed_index,
            "finite": finite,
            "invalid_solve_count": invalid_count,
            "final_return": final_return,
            "final_q": final_q,
            "final_pi1_good": final_good,
            "near_optimal_basin": final_q >= 0.9 and final_good >= 0.9,
            "first_delta_safe_positive_update": first_positive,
            "zero_s1_batch_fraction": zero_count / attempted_updates,
        })
    return checkpoints, endpoints, missing


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(endpoints, path):
    methods = list(METHODS[:4])
    values = []
    for method in methods:
        selected = [row["final_return"] for row in endpoints if row["method"] == method and row["initialization"] == "adverse" and row["n_trajectories"] == 32]
        values.append(float(np.mean(selected)) if selected else np.nan)
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(range(len(methods)), values)
    axis.set_xticks(range(len(methods)), methods, rotation=20, ha="right")
    axis.set_ylabel("mean final exact return")
    axis.set_title("Sampled two-state adverse start, N=32")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _wilson(count: int, total: int) -> tuple[float, float]:
    z = 1.959963984540054
    proportion = count / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z * z / (4.0 * total * total)
    ) / denominator
    return center - half, center + half


def _summaries(endpoints: list[dict]) -> tuple[list[dict], list[dict]]:
    summaries: list[dict] = []
    paired: list[dict] = []
    group_keys = sorted({
        (row["initialization"], int(row["n_trajectories"]))
        for row in endpoints
    })
    comparisons = (
        ("sampled_pg_logbarrier_handoff", "sampled_pg_reward_only", "barrier_within_euclidean"),
        ("sampled_npg_logbarrier_handoff", "sampled_npg_reward_only", "barrier_within_natural"),
        ("sampled_npg_reward_only", "sampled_pg_reward_only", "optimizer_reward_only"),
        ("sampled_npg_logbarrier_handoff", "sampled_pg_logbarrier_handoff", "optimizer_with_barrier"),
        ("sampled_pg_logbarrier_fixed", "sampled_pg_logbarrier_handoff", "fixed_vs_handoff_euclidean"),
        ("sampled_npg_logbarrier_fixed", "sampled_npg_logbarrier_handoff", "fixed_vs_handoff_natural"),
    )
    continuous = (
        "final_return", "final_q", "final_pi1_good", "zero_s1_batch_fraction"
    )
    for initialization, batch_size in group_keys:
        group = [
            row for row in endpoints
            if row["initialization"] == initialization
            and int(row["n_trajectories"]) == batch_size
        ]
        by_method = {
            method: [row for row in group if row["method"] == method]
            for method in METHODS
        }
        for method, rows in by_method.items():
            successes = sum(bool(row["near_optimal_basin"]) for row in rows)
            low, high = _wilson(successes, len(rows))
            summary = {
                "method": method,
                "initialization": initialization,
                "n_trajectories": batch_size,
                "n": len(rows),
                "successes": successes,
                "success_rate": successes / len(rows),
                "success_rate_wilson95_low": low,
                "success_rate_wilson95_high": high,
                "finite_rate": sum(bool(row["finite"]) for row in rows) / len(rows),
            }
            for metric in continuous:
                values = np.asarray([float(row[metric]) for row in rows])
                mean, ci_low, ci_high = mean_confidence_interval(values)
                summary.update({
                    f"{metric}_mean": float(mean),
                    f"{metric}_median": float(np.median(values)),
                    f"{metric}_ci95_low": float(ci_low),
                    f"{metric}_ci95_high": float(ci_high),
                })
            reached = [
                int(row["first_delta_safe_positive_update"])
                for row in rows
                if row["first_delta_safe_positive_update"] is not None
            ]
            summary["delta_safe_positive_count"] = len(reached)
            summary["delta_safe_positive_rate"] = len(reached) / len(rows)
            summary["first_delta_safe_positive_update_median_when_reached"] = (
                float(np.median(reached)) if reached else ""
            )
            summaries.append(summary)
        for method, reference_method, family in comparisons:
            current = {row["seed"]: row for row in by_method[method]}
            reference = {row["seed"]: row for row in by_method[reference_method]}
            seeds = sorted(set(current) & set(reference))
            for metric in continuous + ("near_optimal_basin",):
                differences = np.asarray([
                    float(current[seed][metric]) - float(reference[seed][metric])
                    for seed in seeds
                ])
                mean, ci_low, ci_high = mean_confidence_interval(differences)
                paired.append({
                    "comparison_family": family,
                    "method": method,
                    "reference": reference_method,
                    "initialization": initialization,
                    "n_trajectories": batch_size,
                    "metric": metric,
                    "n": len(seeds),
                    "mean_difference": float(mean),
                    "median_difference": float(np.median(differences)),
                    "ci95_low": float(ci_low),
                    "ci95_high": float(ci_high),
                })
            method_failed_reference_succeeded = sum(
                not bool(current[seed]["near_optimal_basin"])
                and bool(reference[seed]["near_optimal_basin"])
                for seed in seeds
            )
            method_succeeded_reference_failed = sum(
                bool(current[seed]["near_optimal_basin"])
                and not bool(reference[seed]["near_optimal_basin"])
                for seed in seeds
            )
            discordant = (
                method_failed_reference_succeeded
                + method_succeeded_reference_failed
            )
            if discordant:
                smaller = min(
                    method_failed_reference_succeeded,
                    method_succeeded_reference_failed,
                )
                lower_tail = sum(
                    math.comb(discordant, k) for k in range(smaller + 1)
                ) / (2 ** discordant)
                p_value = min(1.0, 2.0 * lower_tail)
            else:
                p_value = 1.0
            paired.append({
                "comparison_family": family,
                "method": method,
                "reference": reference_method,
                "initialization": initialization,
                "n_trajectories": batch_size,
                "metric": "exact_mcnemar_p",
                "n": len(seeds),
                "mean_difference": p_value,
                "median_difference": "",
                "ci95_low": "",
                "ci95_high": "",
                "method_failed_reference_succeeded": method_failed_reference_succeeded,
                "method_succeeded_reference_failed": method_succeeded_reference_failed,
            })
    return summaries, paired


def run_sampled_factorial(output_directory: str | Path, *, preset: str = "smoke") -> dict:
    if preset not in {"smoke", "pilot", "full"}:
        raise ValueError("preset must be smoke, pilot, or full")
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        manifest = output / "manifest.json"
        if manifest.exists():
            return json.loads(manifest.read_text(encoding="utf-8"))
        raise FileExistsError(f"nonempty incomplete sampled output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    n_seeds = {"smoke": 4, "pilot": 20, "full": 100}[preset]
    updates = 50 if preset == "smoke" else 4000
    handoff = 25 if preset == "smoke" else 2000
    batch_sizes = (4, 32) if preset == "smoke" else (4, 32, 128)
    checkpoints, endpoints, missing, configs = [], [], [], []
    for initialization in INITIALIZATIONS:
        for n in batch_sizes:
            for method in METHODS:
                config = SampledFactorialConfig(
                    method, initialization, n, n_seeds,
                    updates=updates, handoff_update=handoff,
                    record_interval=10,
                )
                a, b, c = run_one(config)
                checkpoints.extend(a); endpoints.extend(b); missing.extend(c)
                configs.append(asdict(config))
    _write_csv(output / "sampled_checkpoints.csv", checkpoints)
    _write_csv(output / "sampled_endpoints.csv", endpoints)
    _write_csv(output / "sampled_missing_state_audit.csv", missing)
    _write_csv(output / "method_configs.csv", configs)
    summaries, paired = _summaries(endpoints)
    _write_csv(output / "sampled_method_summaries.csv", summaries)
    _write_csv(output / "sampled_paired_differences.csv", paired)
    _plot(endpoints, output / "sampled_endpoints.png")
    manifest = {
        "schema_version": 1, "complete": True, "preset": preset,
        "seed_count": n_seeds, "endpoint_rows": len(endpoints),
        "checkpoint_rows": len(checkpoints), "missing_state_rows": len(missing),
        "summary_rows": len(summaries), "paired_difference_rows": len(paired),
        "raw_uncentered_unnormalized_returns": True,
        "oracle_information_inserted_for_missing_s1": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
