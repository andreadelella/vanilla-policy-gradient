"""Exact 2×2 Euclidean/natural × reward/barrier factorial."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from exploration.tabular_mdp.geometry import (
    barrier_gradients,
    geometry_snapshot,
    pooled_fisher,
)
from exploration.tabular_mdp.model import (
    DTYPE,
    TwoStepTrap,
    as_phi,
    probabilities_from_reduced_logits,
    phi_from_q_and_good,
    transition_pool_weights,
)

from .natural_step import cosine, target_kl_natural_step


METHODS = (
    "exact_pg_reward_only",
    "exact_pg_logbarrier_handoff",
    "exact_npg_reward_only",
    "exact_npg_logbarrier_handoff",
    "exact_pg_logbarrier_fixed",
    "exact_npg_logbarrier_fixed",
)
INITIALIZATIONS = {
    "uniform": (0.0, 0.0, 0.0, 0.0),
    "adverse": (2.0, -2.0, -2.0, 2.0),
}


@dataclass(frozen=True)
class ExactFactorialConfig:
    method: str
    initialization: str
    damping: float
    updates: int = 2000
    alpha: float = 0.05
    beta: float = 0.1
    handoff_update: int = 500
    target_kl: float = 1e-3
    record_interval: int = 10

    def validate(self) -> None:
        if self.method not in METHODS or self.initialization not in INITIALIZATIONS:
            raise ValueError("unknown exact method or initialization")
        if self.updates < 1 or self.alpha <= 0 or self.beta < 0 or self.damping < 0:
            raise ValueError("invalid numerical configuration")
        if not 0 < self.handoff_update < self.updates:
            raise ValueError("handoff must be inside the optimization horizon")
        if self.target_kl <= 0 or self.record_interval < 1:
            raise ValueError("target KL and record interval must be positive")

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
        if self.fixed or update < self.handoff_update:
            return self.beta
        return 0.0


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    value = float(denominator.norm())
    return float(numerator.norm()) / value if value > 0.0 else float("nan")


def _mean_forward_kl(phi_old: torch.Tensor, phi_new: torch.Tensor) -> float:
    old0, old1 = probabilities_from_reduced_logits(phi_old)
    new0, new1 = probabilities_from_reduced_logits(phi_new)
    mu0, mu1 = transition_pool_weights(phi_old)
    kl0 = (old0 * (torch.log(old0) - torch.log(new0))).sum()
    kl1 = (old1 * (torch.log(old1) - torch.log(new1))).sum()
    return float(mu0 * kl0 + mu1 * kl1)


def _record(
    config: ExactFactorialConfig,
    update: int,
    phi: torch.Tensor,
    *,
    reward_gradient: torch.Tensor | None = None,
    barrier_gradient: torch.Tensor | None = None,
    total_gradient: torch.Tensor | None = None,
    natural_diagnostics: dict | None = None,
    realized_kl: float = float("nan"),
    effective_beta: float | None = None,
    diagnostic_fisher: torch.Tensor | None = None,
) -> dict:
    mdp = TwoStepTrap()
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    rewards1 = torch.tensor(mdp.state1_rewards, dtype=DTYPE)
    value1 = float((pi1 * rewards1).sum())
    geometry = geometry_snapshot(phi).values
    eigenvalues = torch.linalg.eigvalsh(pooled_fisher(phi))
    recorded_beta = config.beta_at(update) if effective_beta is None else effective_beta
    row = {
        "method": config.method,
        "initialization": config.initialization,
        "damping": config.damping,
        "target_kl": config.target_kl if config.natural else "",
        "update": update,
        "beta": recorded_beta,
        "barrier_active": recorded_beta > 0.0,
        "return": float(mdp.exact_return(phi)),
        "q": float(pi0[1]),
        "pi1_good": float(pi1[0]),
        "v1": value1,
        "delta_safe": value1 - mdp.safe_reward,
        "realized_kl": realized_kl,
        **{f"pi0_a{index}": float(pi0[index]) for index in range(3)},
        **{f"pi1_a{index}": float(pi1[index]) for index in range(3)},
        **{f"fisher_eigenvalue_{index}": float(value) for index, value in enumerate(eigenvalues)},
        "mu0": float(geometry["mu0"]),
        "mu1": float(geometry["mu1"]),
        "lambda_min_f_pool": float(geometry["lambda_min_f_pool"]),
        "logdet_f_pool": float(geometry["logdet_f_pool"]),
    }
    if reward_gradient is not None:
        beta_barrier = recorded_beta * barrier_gradient
        fisher = pooled_fisher(phi) if diagnostic_fisher is None else diagnostic_fisher
        damped = fisher + config.damping * torch.eye(4, dtype=DTYPE)
        natural_reward = torch.linalg.solve(damped, reward_gradient)
        natural_barrier = torch.linalg.solve(damped, beta_barrier)
        natural_total = torch.linalg.solve(damped, total_gradient)
        cos_e, cos_e_defined = cosine(reward_gradient, barrier_gradient)
        cos_n, cos_n_defined = cosine(natural_reward, natural_barrier)
        row.update({
            **{f"reward_gradient_{i}": float(reward_gradient[i]) for i in range(4)},
            **{f"barrier_gradient_{i}": float(barrier_gradient[i]) for i in range(4)},
            **{f"natural_reward_direction_{i}": float(natural_reward[i]) for i in range(4)},
            **{f"natural_direction_{i}": float(natural_total[i]) for i in range(4)},
            "reward_gradient_norm": float(reward_gradient.norm()),
            "barrier_gradient_norm": float(barrier_gradient.norm()),
            "total_gradient_norm": float(total_gradient.norm()),
            "euclidean_barrier_to_reward_ratio": _safe_ratio(beta_barrier, reward_gradient),
            "natural_barrier_to_reward_ratio": _safe_ratio(natural_barrier, natural_reward),
            "euclidean_cosine": cos_e,
            "euclidean_cosine_defined": cos_e_defined,
            "natural_cosine": cos_n,
            "natural_cosine_defined": cos_n_defined,
        })
    if natural_diagnostics:
        row.update(natural_diagnostics)
    return row


def run_one(config: ExactFactorialConfig) -> tuple[list[dict], dict]:
    config.validate()
    mdp = TwoStepTrap()
    phi = as_phi(INITIALIZATIONS[config.initialization])
    rows = [_record(config, 0, phi)]
    finite = True
    invalid_updates = 0
    first_positive_delta = None
    for update in range(config.updates):
        reward = mdp.exact_reward_gradient(phi)
        barrier = barrier_gradients(phi).detached_conditional
        fisher_before = pooled_fisher(phi)
        beta = config.beta_at(update)
        total = reward + beta * barrier
        diagnostics = None
        if config.natural:
            result = target_kl_natural_step(
                total, pooled_fisher(phi), damping=config.damping, target_kl=config.target_kl
            )
            diagnostics = result.diagnostics()
            if not result.valid:
                invalid_updates += 1
                finite = False
                break
            step = result.step
        else:
            step = config.alpha * total
        new_phi = phi + step
        realized = _mean_forward_kl(phi, new_phi)
        phi = new_phi
        pi1 = probabilities_from_reduced_logits(phi)[1]
        v1 = float((pi1 * torch.tensor(mdp.state1_rewards, dtype=DTYPE)).sum())
        if first_positive_delta is None and v1 > mdp.safe_reward:
            first_positive_delta = update + 1
        completed = update + 1
        if completed % config.record_interval == 0 or completed in {
            config.handoff_update - 1, config.handoff_update,
            config.handoff_update + 1, config.updates,
        }:
            rows.append(_record(
                config, completed, phi,
                reward_gradient=reward,
                barrier_gradient=barrier,
                total_gradient=total,
                natural_diagnostics=diagnostics,
                realized_kl=realized,
                effective_beta=beta,
                diagnostic_fisher=fisher_before,
            ))
        if not torch.isfinite(phi).all():
            finite = False
            break
    final = rows[-1]
    endpoint = {
        "method": config.method,
        "initialization": config.initialization,
        "damping": config.damping,
        "finite": finite,
        "invalid_updates": invalid_updates,
        "first_delta_safe_positive_update": first_positive_delta,
        "final_return": final["return"],
        "final_q": final["q"],
        "final_pi1_good": final["pi1_good"],
        "final_delta_safe": final["delta_safe"],
        "exact_npg_escaped_adverse": bool(
            config.natural and config.initialization == "adverse"
            and final["q"] >= 0.9 and final["pi1_good"] >= 0.9
        ),
    }
    return rows, endpoint


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


def _plots(rows: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for initialization in INITIALIZATIONS:
        subset = [row for row in rows if row["initialization"] == initialization and float(row["damping"]) == 0.01]
        figure, axes = plt.subplots(2, 2, figsize=(10, 7))
        for method in METHODS[:4]:
            values = [row for row in subset if row["method"] == method]
            if not values:
                continue
            x = [row["update"] for row in values]
            axes[0, 0].plot(x, [row["return"] for row in values], label=method)
            axes[0, 1].plot(x, [row["q"] for row in values], label=method)
            axes[1, 0].plot(x, [row["pi1_good"] for row in values], label=method)
            axes[1, 1].plot([row["q"] for row in values], [row["pi1_good"] for row in values], label=method)
        for axis, title in zip(axes.reshape(-1), ("return", "q", "pi1 good", "phase")):
            axis.set_title(title)
            axis.grid(True, alpha=0.25)
        axes[0, 0].legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(directory / f"{initialization}_factorial.png", dpi=180)
        plt.close(figure)

    adverse = [
        row for row in rows
        if row["initialization"] == "adverse" and float(row["damping"]) == 0.01
    ]
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in METHODS[:4]:
        values = [row for row in adverse if row["method"] == method]
        if values:
            axes[0].plot(
                [row["update"] for row in values],
                [row["fisher_eigenvalue_0"] for row in values],
                label=method,
            )
            axes[1].plot(
                [row["update"] for row in values],
                [row["fisher_eigenvalue_3"] for row in values],
                label=method,
            )
    axes[0].set_title("Smallest pooled-Fisher eigenvalue")
    axes[1].set_title("Largest pooled-Fisher eigenvalue")
    for axis in axes:
        axis.set_xlabel("update")
        axis.set_yscale("log")
        axis.grid(True, alpha=0.25)
    axes[0].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(directory / "adverse_fisher_eigenvalues.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4))
    for method in ("exact_npg_reward_only", "exact_npg_logbarrier_handoff"):
        values = [
            row for row in adverse
            if row["method"] == method and "predicted_kl" in row
        ]
        if values:
            axis.plot(
                [row["update"] for row in values],
                [row["realized_kl"] for row in values],
                label=f"{method}: realized",
            )
            axis.plot(
                [row["update"] for row in values],
                [row["predicted_kl"] for row in values],
                linestyle="--",
                label=f"{method}: predicted",
            )
    axis.set_xlabel("update")
    axis.set_ylabel("mean forward KL")
    axis.set_title("Predicted versus realized exact-policy KL")
    axis.grid(True, alpha=0.25)
    axis.legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(directory / "adverse_predicted_vs_realized_kl.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, family in enumerate(("pg", "npg")):
        for suffix in ("handoff", "fixed"):
            method = f"exact_{family}_logbarrier_{suffix}"
            values = [row for row in adverse if row["method"] == method]
            if values:
                axes[index].plot(
                    [row["update"] for row in values],
                    [row["return"] for row in values],
                    label=suffix,
                )
        axes[index].set_title(f"{family.upper()}: fixed versus handoff")
        axes[index].set_xlabel("update")
        axes[index].set_ylabel("exact return")
        axes[index].grid(True, alpha=0.25)
        axes[index].legend()
    figure.tight_layout()
    figure.savefig(directory / "adverse_fixed_vs_handoff.png", dpi=180)
    plt.close(figure)

    grid = torch.linspace(0.05, 0.95, 13, dtype=DTYPE)
    q_grid, good_grid = torch.meshgrid(grid, grid, indexing="xy")
    phi_grid = phi_from_q_and_good(q_grid.reshape(-1), good_grid.reshape(-1))
    gradient = TwoStepTrap().exact_reward_gradient(phi_grid)
    small_step = 1e-3
    next_pi0, next_pi1 = probabilities_from_reduced_logits(
        phi_grid + small_step * gradient
    )
    dq = ((next_pi0[:, 1] - q_grid.reshape(-1)) / small_step).reshape(q_grid.shape)
    dgood = ((next_pi1[:, 0] - good_grid.reshape(-1)) / small_step).reshape(good_grid.shape)
    figure, axis = plt.subplots(figsize=(6, 5))
    axis.quiver(
        q_grid.numpy(), good_grid.numpy(), dq.numpy(), dgood.numpy(),
        angles="xy",
    )
    axis.axhline(0.5, color="black", linestyle="--", linewidth=0.8)
    axis.set_xlabel(r"$q=\pi_0(a_1)$")
    axis.set_ylabel(r"$\pi_1(a_0)$")
    axis.set_title("Exact Euclidean reward vector field")
    figure.tight_layout()
    figure.savefig(directory / "reward_vector_field.png", dpi=180)
    plt.close(figure)


def run_exact_factorial(output_directory: str | Path) -> dict:
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        manifest = output / "manifest.json"
        if manifest.exists():
            return json.loads(manifest.read_text(encoding="utf-8"))
        raise FileExistsError(f"nonempty incomplete exact output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rows, endpoints, configs = [], [], []
    for initialization in INITIALIZATIONS:
        for method in METHODS:
            dampings = (0.0, 0.01, 0.1) if "_npg_" in method else (0.01,)
            for damping in dampings:
                config = ExactFactorialConfig(method, initialization, damping)
                checkpoint_rows, endpoint = run_one(config)
                rows.extend(checkpoint_rows)
                endpoints.append(endpoint)
                configs.append(asdict(config))
    _write_csv(output / "exact_checkpoints.csv", rows)
    _write_csv(output / "exact_endpoints.csv", endpoints)
    _write_csv(output / "method_configs.csv", configs)
    _plots(rows, output / "plots")
    adverse_npg = [
        row for row in endpoints
        if row["method"] == "exact_npg_reward_only"
        and row["initialization"] == "adverse"
    ]
    escape_by_damping = {
        str(row["damping"]): bool(row["exact_npg_escaped_adverse"])
        for row in adverse_npg
    }
    manifest = {
        "schema_version": 1,
        "complete": True,
        "checkpoint_rows": len(rows),
        "endpoint_rows": len(endpoints),
        "primary_damping": 0.01,
        "damping_controls": [0.0, 0.1],
        "exact_npg_reward_only_escapes_adverse_at_primary_damping": escape_by_damping["0.01"],
        "exact_npg_reward_only_escape_by_damping": escape_by_damping,
        "ordering_asserted_by_tests": False,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest
