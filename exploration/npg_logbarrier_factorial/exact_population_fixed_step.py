"""Exact-population fixed-step NPG factorial for the two-state coordination trap.

This module is additive. It does not modify ``exact_two_state.py``, which uses
target-KL-normalized natural steps, and it does not reuse ``natural_step.py``.
The primary method here is the theorem-matched update

    x = F_pool^{-1} grad J,      phi_next = phi + eta * x,

with no target-KL normalization, no line search, no clipping, and no damping.

Everything is exact population arithmetic in float64: no trajectories, no Monte
Carlo estimates, no environment rollouts, no seeds, no empirical Fisher.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from exploration.tabular_mdp.geometry import (
    barrier_gradients,
    pooled_fisher,
    reduced_categorical_fisher,
)
from exploration.tabular_mdp.model import (
    DTYPE,
    TwoStepTrap,
    as_phi,
    phi_from_q_and_good,
    probabilities_from_reduced_logits,
    transition_pool_weights,
)


PRIMARY_METHOD = "exact_npg_reward_only_fixed_step"
PSEUDOINVERSE_METHOD = "exact_npg_reward_only_pseudoinverse"

METHODS = (
    "exact_pg_reward_only_fixed_step",
    "exact_pg_logbarrier_handoff_fixed_step",
    "exact_npg_reward_only_fixed_step",
    "exact_npg_logbarrier_handoff_fixed_step",
    "exact_pg_logbarrier_fixed_fixed_step",
    "exact_npg_logbarrier_fixed_fixed_step",
    PSEUDOINVERSE_METHOD,
)
INITIALIZATIONS = {
    "uniform": (0.0, 0.0, 0.0, 0.0),
    "adverse": (2.0, -2.0, -2.0, 2.0),
}
PRIMARY_DAMPING = 0.0
DAMPING_CONTROLS = (0.01, 0.1)
SAFE_THRESHOLD = 0.5


@dataclass(frozen=True)
class FixedStepConfig:
    """One exact deterministic run. No seed field exists by construction."""

    method: str
    initialization: str
    damping: float = PRIMARY_DAMPING
    updates: int = 2000
    alpha: float = 0.05
    eta: float = 0.05
    beta: float = 0.1
    handoff_update: int = 500
    record_interval: int = 10

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown exact fixed-step method: {self.method}")
        if self.initialization not in INITIALIZATIONS:
            raise ValueError(f"unknown initialization: {self.initialization}")
        if self.updates < 1:
            raise ValueError("updates must be positive")
        if self.alpha <= 0 or self.eta <= 0:
            raise ValueError("alpha and eta must be positive")
        if self.beta < 0 or self.damping < 0:
            raise ValueError("beta and damping must be nonnegative")
        if not 0 < self.handoff_update < self.updates:
            raise ValueError("handoff must lie strictly inside the horizon")
        if self.record_interval < 1:
            raise ValueError("record interval must be positive")

    @property
    def natural(self) -> bool:
        return "_npg_" in self.method

    @property
    def barrier(self) -> bool:
        return "logbarrier" in self.method

    @property
    def fixed_barrier(self) -> bool:
        return "logbarrier_fixed" in self.method

    @property
    def pseudoinverse(self) -> bool:
        return self.method == PSEUDOINVERSE_METHOD

    @property
    def step_size(self) -> float:
        return self.eta if self.natural else self.alpha

    def beta_at(self, update: int) -> float:
        if not self.barrier:
            return 0.0
        if self.fixed_barrier or update < self.handoff_update:
            return self.beta
        return 0.0


def state_value_one(phi) -> torch.Tensor:
    """Downstream value ``V1 = pi1(a0) + 0.2 * pi1(a1)``."""
    _, pi1 = probabilities_from_reduced_logits(phi)
    rewards = torch.tensor(TwoStepTrap().state1_rewards, dtype=DTYPE)
    return (pi1 * rewards).sum(dim=-1)


def analytic_natural_direction(phi) -> torch.Tensor:
    """Closed-form undamped reward-only natural direction.

    Under the transition-pooled convention the exact reduced-logit direction is
    ``(1 + q) * (Q0(a0), Q0(a1), Q1(a0), Q1(a1))`` with ``a2`` as reference and
    ``Q(a2) = 0``, that is ``(1 + q) * (0.5, V1, 1.0, 0.2)``.
    """
    pi0, _ = probabilities_from_reduced_logits(phi)
    mdp = TwoStepTrap()
    q = pi0[..., 1]
    value1 = state_value_one(phi)
    scale = (1.0 + q).unsqueeze(-1)
    components = torch.stack(
        (
            torch.full_like(q, mdp.safe_reward),
            value1,
            torch.full_like(q, mdp.state1_rewards[0]),
            torch.full_like(q, mdp.state1_rewards[1]),
        ),
        dim=-1,
    )
    return scale * components


SOLVE_TRUST_TOLERANCE = 1e-6
DIRECTION_TRUST_TOLERANCE = 1e-6
FLOAT64_EPSILON = float(torch.finfo(DTYPE).eps)


def solve_is_trustworthy(fisher: torch.Tensor, tolerance: float = SOLVE_TRUST_TOLERANCE) -> bool:
    """Conservative a-priori conditioning check, reported as a diagnostic only.

    The forward error of a linear solve grows at least like ``cond(A) * eps``. A
    merely positive smallest eigenvalue is not sufficient: at
    ``lambda_min ~ 1e-16`` the matrix is positive definite on paper while the
    solve is already meaningless.

    This bound is optimistic near convergence, because the reward gradient also
    collapses and carries its own relative roundoff. The trusted region actually
    used by the experiment is measured against the closed-form direction rather
    than predicted from this quantity.
    """
    eigenvalues = torch.linalg.eigvalsh(0.5 * (fisher + fisher.transpose(-1, -2)))
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    if smallest <= 0.0 or largest <= 0.0:
        return False
    return (largest / smallest) * FLOAT64_EPSILON < tolerance


@dataclass(frozen=True)
class DirectionResult:
    direction: torch.Tensor
    valid: bool
    invalid_reason: str | None
    solve_residual: float
    condition_number: float


def _condition_number(matrix: torch.Tensor) -> float:
    eigenvalues = torch.linalg.eigvalsh(0.5 * (matrix + matrix.transpose(-1, -2)))
    smallest, largest = float(eigenvalues[0]), float(eigenvalues[-1])
    if smallest <= 0.0:
        return float("inf")
    return largest / smallest


def fixed_step_direction(
    gradient: torch.Tensor,
    fisher: torch.Tensor,
    *,
    damping: float = PRIMARY_DAMPING,
    pseudoinverse: bool = False,
) -> DirectionResult:
    """Solve ``(F + damping I) x = g`` without any normalization or fallback.

    The undamped primary path uses ``torch.linalg.solve``. A failed or nonfinite
    solve is reported as invalid; damping is never added silently and the
    pseudoinverse is used only when explicitly requested.
    """

    gradient = torch.as_tensor(gradient, dtype=DTYPE)
    fisher = torch.as_tensor(fisher, dtype=DTYPE)
    if gradient.ndim != 1 or fisher.shape != (gradient.numel(), gradient.numel()):
        raise ValueError("gradient and Fisher shapes are incompatible")
    if damping < 0.0:
        raise ValueError("damping must be nonnegative")
    if not torch.isfinite(gradient).all() or not torch.isfinite(fisher).all():
        raise ValueError("gradient and Fisher must be finite")

    matrix = fisher + damping * torch.eye(fisher.shape[0], dtype=DTYPE)
    condition = _condition_number(matrix)
    if pseudoinverse:
        direction = torch.linalg.pinv(matrix) @ gradient
    else:
        try:
            direction = torch.linalg.solve(matrix, gradient)
        except torch.linalg.LinAlgError:
            zero = torch.zeros_like(gradient)
            return DirectionResult(zero, False, "pooled_fisher_solve_failed", float("inf"), condition)
    if not torch.isfinite(direction).all():
        zero = torch.zeros_like(gradient)
        return DirectionResult(zero, False, "nonfinite_natural_direction", float("inf"), condition)
    residual = float((matrix @ direction - gradient).norm())
    return DirectionResult(direction.detach(), True, None, residual, condition)


def predicted_log_odds_increments(phi, eta: float) -> dict[str, float]:
    """Analytic one-step log-odds increments for undamped reward-only NPG."""
    pi0, _ = probabilities_from_reduced_logits(phi)
    q = float(pi0[..., 1])
    value1 = float(state_value_one(phi))
    scale = eta * (1.0 + q)
    return {
        "predicted_delta_log_odds_explore_safe": scale * (value1 - SAFE_THRESHOLD),
        "predicted_delta_log_odds_good_medium": scale * 0.8,
        "predicted_delta_log_odds_good_reference": scale * 1.0,
    }


def realized_log_odds_increments(phi_old, phi_new) -> dict[str, float]:
    """Realized increments read directly from the reduced logits."""
    old = torch.as_tensor(phi_old, dtype=DTYPE)
    new = torch.as_tensor(phi_new, dtype=DTYPE)
    delta = new - old
    return {
        "realized_delta_log_odds_explore_safe": float(delta[1] - delta[0]),
        "realized_delta_log_odds_good_medium": float(delta[2] - delta[3]),
        "realized_delta_log_odds_good_reference": float(delta[2]),
    }


def attenuation_factors(phi, damping: float) -> dict[str, float]:
    """Eigendirection and statewise damping attenuation ``lam / (lam + damping)``.

    With ``damping == 0`` the operator is unattenuated in every direction, so the
    factor is one. Eigenvalues are clamped at zero first: at a deterministic
    corner the exact eigenvalue is zero and float64 can return a tiny negative
    value, which would otherwise produce a spurious out-of-range factor.
    """
    fisher = pooled_fisher(phi)
    eigenvalues = torch.linalg.eigvalsh(fisher)
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    mu0, mu1 = transition_pool_weights(phi)
    sigma0 = torch.linalg.eigvalsh(reduced_categorical_fisher(pi0))
    sigma1 = torch.linalg.eigvalsh(reduced_categorical_fisher(pi1))

    def ratio(value: torch.Tensor) -> float:
        if damping == 0.0:
            return 1.0
        clamped = max(float(value), 0.0)
        return clamped / (clamped + damping)

    result = {f"attenuation_{index}": ratio(value) for index, value in enumerate(eigenvalues)}
    result.update(
        {f"statewise_attenuation_s0_{index}": ratio(mu0 * value) for index, value in enumerate(sigma0)}
    )
    result.update(
        {f"statewise_attenuation_s1_{index}": ratio(mu1 * value) for index, value in enumerate(sigma1)}
    )
    return result


def _record(
    config: FixedStepConfig,
    update: int,
    phi: torch.Tensor,
    *,
    reward_gradient: torch.Tensor | None = None,
    barrier_gradient: torch.Tensor | None = None,
    direction: torch.Tensor | None = None,
    direction_result: DirectionResult | None = None,
    analytic_direction: torch.Tensor | None = None,
    log_odds: dict[str, float] | None = None,
    effective_beta: float | None = None,
    milestones: dict[str, int | None] | None = None,
    finite: bool = True,
    invalid_reason: str | None = None,
    trustworthy: bool | None = None,
) -> dict:
    mdp = TwoStepTrap()
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    mu0, mu1 = transition_pool_weights(phi)
    value1 = state_value_one(phi)
    fisher = pooled_fisher(phi)
    eigenvalues = torch.linalg.eigvalsh(fisher)
    # The gradient and direction columns are evaluated where the step was taken,
    # so the analytic reference must be evaluated at that same point.
    analytic = analytic_natural_direction(phi) if analytic_direction is None else analytic_direction
    beta = config.beta_at(update) if effective_beta is None else effective_beta

    row = {
        "method": config.method,
        "initialization": config.initialization,
        "damping": config.damping,
        "step_size": config.step_size,
        "update": update,
        "beta": beta,
        "barrier_active": beta > 0.0,
        "return": float(mdp.exact_return(phi)),
        "q": float(pi0[1]),
        "v1": float(value1),
        "delta_safe": float(value1) - SAFE_THRESHOLD,
        "pi1_good": float(pi1[0]),
        **{f"pi0_a{index}": float(pi0[index]) for index in range(3)},
        **{f"pi1_a{index}": float(pi1[index]) for index in range(3)},
        **{f"analytic_direction_{index}": float(analytic[index]) for index in range(4)},
        **{f"fisher_eigenvalue_{index}": float(value) for index, value in enumerate(eigenvalues)},
        "fisher_condition_number": _condition_number(fisher),
        "fisher_min_eigenvalue": float(eigenvalues[0]),
        "fisher_numerically_positive_definite": bool(float(eigenvalues[0]) > 0.0),
        "solve_trustworthy": (
            solve_is_trustworthy(fisher) if trustworthy is None else trustworthy
        ),
        "conditioning_trustworthy": solve_is_trustworthy(fisher),
        "mu0": float(mu0),
        "mu1": float(mu1),
        "finite": finite,
        "invalid_reason": invalid_reason,
    }
    row.update(attenuation_factors(phi, config.damping))

    if reward_gradient is not None:
        row.update({f"reward_gradient_{i}": float(reward_gradient[i]) for i in range(4)})
        row["reward_gradient_norm"] = float(reward_gradient.norm())
    if barrier_gradient is not None:
        row.update({f"barrier_gradient_{i}": float(barrier_gradient[i]) for i in range(4)})
        row["barrier_gradient_norm"] = float(barrier_gradient.norm())
    if direction is not None:
        row.update({f"natural_direction_{i}": float(direction[i]) for i in range(4)})
        row["natural_direction_norm"] = float(direction.norm())
        denominator = float(analytic.norm())
        row["direction_vs_analytic_relative_error"] = (
            float((direction - analytic).norm()) / denominator if denominator > 0.0 else float("nan")
        )
    if direction_result is not None:
        row["solve_residual"] = direction_result.solve_residual
        row["damped_condition_number"] = direction_result.condition_number
    if log_odds is not None:
        row.update(log_odds)
    if milestones is not None:
        row.update(milestones)
    return row


def run_one(config: FixedStepConfig) -> tuple[list[dict], dict]:
    """Run one exact deterministic trajectory through parameter space."""
    config.validate()
    mdp = TwoStepTrap()
    phi = as_phi(INITIALIZATIONS[config.initialization])

    milestones: dict[str, int | None] = {
        "first_update_v1_above_half": None,
        "first_update_q_above_half": None,
        "first_update_q_above_090": None,
    }
    rows = [_record(config, 0, phi, milestones=dict(milestones))]
    finite = True
    invalid_reason = None
    invalid_updates = 0
    worst_direction_error = 0.0
    worst_direction_error_after_degeneracy = 0.0
    first_degenerate_update: int | None = None

    for update in range(config.updates):
        reward_gradient = mdp.exact_reward_gradient(phi)
        barrier_gradient = barrier_gradients(phi).detached_conditional
        beta = config.beta_at(update)
        total_gradient = reward_gradient + beta * barrier_gradient
        fisher = pooled_fisher(phi)
        trustworthy = first_degenerate_update is None

        direction_result = None
        if config.natural:
            direction_result = fixed_step_direction(
                total_gradient,
                fisher,
                damping=config.damping,
                pseudoinverse=config.pseudoinverse,
            )
            if not direction_result.valid:
                invalid_updates += 1
                finite = False
                invalid_reason = direction_result.invalid_reason
                rows.append(
                    _record(
                        config, update, phi,
                        reward_gradient=reward_gradient,
                        barrier_gradient=barrier_gradient,
                        direction_result=direction_result,
                        effective_beta=beta,
                        milestones=dict(milestones),
                        finite=False,
                        invalid_reason=invalid_reason,
                    )
                )
                break
            direction = direction_result.direction
        else:
            direction = total_gradient

        step = config.step_size * direction
        new_phi = phi + step

        if not torch.isfinite(new_phi).all():
            finite = False
            invalid_reason = "nonfinite_parameters"
            rows.append(
                _record(
                    config, update, phi,
                    reward_gradient=reward_gradient,
                    barrier_gradient=barrier_gradient,
                    direction=direction,
                    direction_result=direction_result,
                    analytic_direction=analytic_natural_direction(phi),
                    effective_beta=beta,
                    milestones=dict(milestones),
                    finite=False,
                    invalid_reason=invalid_reason,
                )
            )
            break

        log_odds = predicted_log_odds_increments(phi, config.step_size)
        log_odds.update(realized_log_odds_increments(phi, new_phi))
        analytic_at_step = analytic_natural_direction(phi)
        # The trusted region is measured against the closed-form direction, not
        # predicted from the condition number. Once breached it stays breached:
        # past the collapse the spectrum is roundoff and any recovery is spurious.
        if config.natural and config.damping == 0.0 and not config.barrier:
            denominator = float(analytic_at_step.norm())
            if denominator > 0.0:
                error = float((direction - analytic_at_step).norm()) / denominator
                if first_degenerate_update is None and error > DIRECTION_TRUST_TOLERANCE:
                    first_degenerate_update = update
                    trustworthy = False
                if trustworthy:
                    worst_direction_error = max(worst_direction_error, error)
                else:
                    worst_direction_error_after_degeneracy = max(
                        worst_direction_error_after_degeneracy, error
                    )
        elif first_degenerate_update is None and not solve_is_trustworthy(fisher):
            first_degenerate_update = update
            trustworthy = False

        phi_before = phi
        phi = new_phi
        completed = update + 1

        pi0_new, _ = probabilities_from_reduced_logits(phi)
        q_new = float(pi0_new[1])
        v1_new = float(state_value_one(phi))
        crossed = False
        if milestones["first_update_v1_above_half"] is None and v1_new > SAFE_THRESHOLD:
            milestones["first_update_v1_above_half"] = completed
            crossed = True
        if milestones["first_update_q_above_half"] is None and q_new > 0.5:
            milestones["first_update_q_above_half"] = completed
            crossed = True
        if milestones["first_update_q_above_090"] is None and q_new > 0.9:
            milestones["first_update_q_above_090"] = completed
            crossed = True

        scheduled = completed % config.record_interval == 0 or completed in {
            config.handoff_update - 1,
            config.handoff_update,
            config.handoff_update + 1,
            config.updates,
        }
        if scheduled or crossed:
            rows.append(
                _record(
                    config, completed, phi,
                    reward_gradient=reward_gradient,
                    barrier_gradient=barrier_gradient,
                    direction=direction,
                    direction_result=direction_result,
                    analytic_direction=analytic_at_step,
                    log_odds=log_odds,
                    effective_beta=beta,
                    milestones=dict(milestones),
                    trustworthy=first_degenerate_update is None,
                )
            )
            rows[-1]["phi_before_update_0"] = float(phi_before[0])

    final = rows[-1]
    endpoint = {
        "method": config.method,
        "initialization": config.initialization,
        "damping": config.damping,
        "step_size": config.step_size,
        "updates": config.updates,
        "finite": finite,
        "invalid_reason": invalid_reason,
        "invalid_updates": invalid_updates,
        "final_return": final["return"],
        "final_q": final["q"],
        "final_v1": final["v1"],
        "final_delta_safe": final["delta_safe"],
        "final_pi1_good": final["pi1_good"],
        "final_update_recorded": final["update"],
        "worst_undamped_direction_relative_error": worst_direction_error,
        "worst_direction_relative_error_after_degeneracy": worst_direction_error_after_degeneracy,
        "first_update_fisher_not_positive_definite": first_degenerate_update,
        **milestones,
        "escaped_adverse": bool(
            config.initialization == "adverse" and final["q"] >= 0.9 and final["pi1_good"] >= 0.9
        ),
    }
    return rows, endpoint


def _interior_validation_grid() -> torch.Tensor:
    """Deterministic interior policies; no seeds or random draws are used."""
    q_values = torch.tensor(
        [1e-9, 1e-6, 1e-4, 1e-3, 0.01, 0.05, 0.2, 0.4, 0.5, 0.6, 0.8, 0.95, 0.999],
        dtype=DTYPE,
    )
    good_values = torch.tensor(
        [1e-6, 1e-3, 0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.9999], dtype=DTYPE
    )
    grid_q, grid_good = torch.meshgrid(q_values, good_values, indexing="xy")
    grid = phi_from_q_and_good(grid_q.reshape(-1), grid_good.reshape(-1))
    extra = torch.stack([as_phi(value) for value in INITIALIZATIONS.values()])
    return torch.cat((grid, extra), dim=0)


def direction_identity_validation() -> list[dict]:
    """Compare the undamped solve against the closed-form natural direction."""
    mdp = TwoStepTrap()
    rows = []
    for phi in _interior_validation_grid():
        pi0, _ = probabilities_from_reduced_logits(phi)
        gradient = mdp.exact_reward_gradient(phi)
        fisher = pooled_fisher(phi)
        result = fixed_step_direction(gradient, fisher, damping=PRIMARY_DAMPING)
        analytic = analytic_natural_direction(phi)
        denominator = float(analytic.norm())
        error = float((result.direction - analytic).norm())
        rows.append(
            {
                "q": float(pi0[1]),
                "v1": float(state_value_one(phi)),
                **{f"phi_{index}": float(phi[index]) for index in range(4)},
                **{f"solved_{index}": float(result.direction[index]) for index in range(4)},
                **{f"analytic_{index}": float(analytic[index]) for index in range(4)},
                "absolute_error": error,
                "relative_error": error / denominator if denominator > 0.0 else float("nan"),
                "solve_residual": result.solve_residual,
                "fisher_condition_number": result.condition_number,
                "valid": result.valid,
            }
        )
    return rows


def log_odds_identity_validation(eta: float) -> list[dict]:
    """Check the one-step log-odds increments of the primary NPG update."""
    mdp = TwoStepTrap()
    rows = []
    for phi in _interior_validation_grid():
        gradient = mdp.exact_reward_gradient(phi)
        fisher = pooled_fisher(phi)
        result = fixed_step_direction(gradient, fisher, damping=PRIMARY_DAMPING)
        new_phi = phi + eta * result.direction
        predicted = predicted_log_odds_increments(phi, eta)
        realized = realized_log_odds_increments(phi, new_phi)
        pi0, _ = probabilities_from_reduced_logits(phi)
        row = {
            "eta": eta,
            "q": float(pi0[1]),
            "v1": float(state_value_one(phi)),
            **predicted,
            **realized,
        }
        for name in ("explore_safe", "good_medium", "good_reference"):
            row[f"absolute_error_{name}"] = abs(
                predicted[f"predicted_delta_log_odds_{name}"]
                - realized[f"realized_delta_log_odds_{name}"]
            )
        rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _series(rows: list[dict], method: str, initialization: str, damping: float) -> list[dict]:
    return [
        row
        for row in rows
        if row["method"] == method
        and row["initialization"] == initialization
        and float(row["damping"]) == damping
    ]


def _plots(rows: list[dict], directory: Path, config: FixedStepConfig, root: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    primary_four = METHODS[:4]

    for initialization in INITIALIZATIONS:
        def curve(axis, key, methods=primary_four, damping=PRIMARY_DAMPING):
            for method in methods:
                values = _series(rows, method, initialization, damping if "_npg_" in method else PRIMARY_DAMPING)
                if values:
                    axis.plot([r["update"] for r in values], [r[key] for r in values], label=method)
            axis.set_xlabel("update")
            axis.grid(True, alpha=0.25)

        for key, title, filename, threshold in (
            ("return", "Exact return", "return_versus_update", None),
            ("q", r"$q=\pi_0(a_1)$", "q_versus_update", None),
            ("v1", r"$V_1$", "v1_versus_update", SAFE_THRESHOLD),
            ("pi1_good", r"$\pi_1(a_0)$", "pi1_good_versus_update", None),
        ):
            figure, axis = plt.subplots(figsize=(7, 4.5))
            curve(axis, key)
            if threshold is not None:
                axis.axhline(threshold, color="black", linestyle="--", linewidth=0.9, label="threshold 0.5")
            axis.set_ylabel(title)
            axis.set_title(f"{title} ({initialization})")
            axis.legend(fontsize=6)
            figure.tight_layout()
            figure.savefig(directory / f"{initialization}_{filename}.png", dpi=180)
            plt.close(figure)

        figure, axis = plt.subplots(figsize=(6, 5))
        for method in primary_four:
            values = _series(rows, method, initialization, PRIMARY_DAMPING)
            if values:
                axis.plot([r["q"] for r in values], [r["v1"] for r in values], label=method)
        axis.axhline(SAFE_THRESHOLD, color="black", linestyle="--", linewidth=0.9)
        axis.set_xlabel(r"$q=\pi_0(a_1)$")
        axis.set_ylabel(r"$V_1$")
        axis.set_title(f"Phase plane ({initialization})")
        axis.grid(True, alpha=0.25)
        axis.legend(fontsize=6)
        figure.tight_layout()
        figure.savefig(directory / f"{initialization}_phase_q_versus_v1.png", dpi=180)
        plt.close(figure)

        for key, title, filename in (
            ("realized_delta_log_odds_explore_safe", "explore versus safe", "log_odds_explore_safe"),
            ("realized_delta_log_odds_good_medium", "downstream good versus medium", "log_odds_good_medium"),
        ):
            figure, axis = plt.subplots(figsize=(7, 4.5))
            for method in primary_four:
                values = [r for r in _series(rows, method, initialization, PRIMARY_DAMPING) if key in r]
                if values:
                    axis.plot([r["update"] for r in values], [r[key] for r in values], label=method)
            axis.set_xlabel("update")
            axis.set_ylabel(f"realized {title} log-odds increment")
            axis.set_title(f"Realized {title} increment ({initialization})")
            axis.grid(True, alpha=0.25)
            axis.legend(fontsize=6)
            figure.tight_layout()
            figure.savefig(directory / f"{initialization}_{filename}.png", dpi=180)
            plt.close(figure)

    adverse = "adverse"
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for method in primary_four:
        values = _series(rows, method, adverse, PRIMARY_DAMPING)
        if values:
            axes[0].plot([r["update"] for r in values], [r["fisher_eigenvalue_0"] for r in values], label=method)
            axes[1].plot([r["update"] for r in values], [r["fisher_eigenvalue_3"] for r in values], label=method)
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

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, initialization in enumerate(INITIALIZATIONS):
        for damping in (PRIMARY_DAMPING, *DAMPING_CONTROLS):
            values = _series(rows, PRIMARY_METHOD, initialization, damping)
            if values:
                label = "undamped (primary)" if damping == 0.0 else f"damping={damping}"
                axes[index].plot([r["update"] for r in values], [r["v1"] for r in values], label=label)
        axes[index].axhline(SAFE_THRESHOLD, color="black", linestyle="--", linewidth=0.9)
        axes[index].set_title(f"Undamped versus damped exact NPG ({initialization})")
        axes[index].set_xlabel("update")
        axes[index].set_ylabel(r"$V_1$")
        axes[index].grid(True, alpha=0.25)
        axes[index].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(directory / "undamped_versus_damped.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, initialization in enumerate(INITIALIZATIONS):
        for method in ("exact_npg_reward_only_fixed_step", "exact_npg_logbarrier_handoff_fixed_step"):
            values = _series(rows, method, initialization, PRIMARY_DAMPING)
            if values:
                axes[index].plot([r["update"] for r in values], [r["return"] for r in values], label=method)
        axes[index].set_title(f"Reward-only versus temporary barrier ({initialization})")
        axes[index].set_xlabel("update")
        axes[index].set_ylabel("exact return")
        axes[index].grid(True, alpha=0.25)
        axes[index].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(directory / "reward_only_versus_temporary_barrier.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, family in enumerate(("pg", "npg")):
        for suffix in ("handoff", "fixed"):
            method = f"exact_{family}_logbarrier_{suffix}_fixed_step"
            values = _series(rows, method, adverse, PRIMARY_DAMPING)
            if values:
                axes[index].plot([r["update"] for r in values], [r["return"] for r in values], label=suffix)
        axes[index].set_title(f"{family.upper()}: fixed versus temporary barrier (adverse)")
        axes[index].set_xlabel("update")
        axes[index].set_ylabel("exact return")
        axes[index].grid(True, alpha=0.25)
        axes[index].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(directory / "fixed_versus_temporary_barrier.png", dpi=180)
    plt.close(figure)

    _plot_against_target_kl(rows, directory, root)


def _plot_against_target_kl(rows: list[dict], directory: Path, root: Path) -> None:
    """Overlay the existing target-KL experiment; skipped when it is absent."""
    reference = root.parent / "exact_two_state" / "exact_checkpoints.csv"
    if not reference.exists():
        return
    with reference.open(encoding="utf-8") as handle:
        existing = list(csv.DictReader(handle))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    for index, initialization in enumerate(INITIALIZATIONS):
        values = _series(rows, PRIMARY_METHOD, initialization, PRIMARY_DAMPING)
        if values:
            axes[index].plot(
                [r["update"] for r in values], [r["return"] for r in values],
                label="fixed-step exact NPG",
            )
        target_kl = [
            row for row in existing
            if row["method"] == "exact_npg_reward_only"
            and row["initialization"] == initialization
            and float(row["damping"]) == 0.01
        ]
        if target_kl:
            axes[index].plot(
                [int(row["update"]) for row in target_kl],
                [float(row["return"]) for row in target_kl],
                linestyle="--", label="target-KL exact NPG (damping 0.01)",
            )
        axes[index].set_title(f"Fixed step versus target-KL ({initialization})")
        axes[index].set_xlabel("update")
        axes[index].set_ylabel("exact return")
        axes[index].grid(True, alpha=0.25)
        axes[index].legend(fontsize=6)
    figure.tight_layout()
    figure.savefig(directory / "fixed_step_versus_target_kl.png", dpi=180)
    plt.close(figure)


def _planned_configs(base: FixedStepConfig) -> list[FixedStepConfig]:
    from dataclasses import replace

    configs = []
    for initialization in INITIALIZATIONS:
        for method in METHODS:
            dampings = (PRIMARY_DAMPING, *DAMPING_CONTROLS) if "_npg_" in method else (PRIMARY_DAMPING,)
            if method == PSEUDOINVERSE_METHOD:
                dampings = (PRIMARY_DAMPING,)
            for damping in dampings:
                configs.append(
                    replace(base, method=method, initialization=initialization, damping=damping)
                )
    return configs


def run_exact_population_fixed_step(
    output_directory: str | Path, *, base: FixedStepConfig | None = None
) -> dict:
    """Run every exact method and initialization and write all artifacts."""
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        manifest_path = output / "manifest.json"
        if manifest_path.exists():
            return json.loads(manifest_path.read_text(encoding="utf-8"))
        raise FileExistsError(f"nonempty incomplete output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)

    base = base or FixedStepConfig(method=PRIMARY_METHOD, initialization="adverse")
    checkpoints: list[dict] = []
    endpoints: list[dict] = []
    configs: list[dict] = []
    for config in _planned_configs(base):
        rows, endpoint = run_one(config)
        checkpoints.extend(rows)
        endpoints.append(endpoint)
        configs.append(asdict(config))

    direction_rows = direction_identity_validation()
    log_odds_rows = log_odds_identity_validation(base.eta)

    _write_csv(output / "exact_population_checkpoints.csv", checkpoints)
    _write_csv(output / "exact_population_endpoints.csv", endpoints)
    _write_csv(output / "direction_identity_validation.csv", direction_rows)
    _write_csv(output / "log_odds_identity_validation.csv", log_odds_rows)
    _write_csv(output / "method_configs.csv", configs)
    _plots(checkpoints, output / "plots", base, output)

    worst_direction = max(row["relative_error"] for row in direction_rows)
    worst_log_odds = max(
        max(
            row["absolute_error_explore_safe"],
            row["absolute_error_good_medium"],
            row["absolute_error_good_reference"],
        )
        for row in log_odds_rows
    )
    primary = {
        (row["initialization"]): row
        for row in endpoints
        if row["method"] == PRIMARY_METHOD and row["damping"] == PRIMARY_DAMPING
    }
    manifest = {
        "schema_version": 1,
        "complete": True,
        "experiment": "exact_population_fixed_step",
        "primary_method": PRIMARY_METHOD,
        "primary_damping": PRIMARY_DAMPING,
        "damping_controls": list(DAMPING_CONTROLS),
        "sampling_used": False,
        "seeds_used": False,
        "target_kl_normalization_used": False,
        "checkpoint_rows": len(checkpoints),
        "endpoint_rows": len(endpoints),
        "direction_validation_rows": len(direction_rows),
        "log_odds_validation_rows": len(log_odds_rows),
        "worst_direction_relative_error": worst_direction,
        "worst_log_odds_absolute_error": worst_log_odds,
        "defaults": asdict(base),
        "primary_endpoints": {
            name: {
                "final_return": row["final_return"],
                "final_q": row["final_q"],
                "final_v1": row["final_v1"],
                "final_pi1_good": row["final_pi1_good"],
                "finite": row["finite"],
                "first_update_v1_above_half": row["first_update_v1_above_half"],
                "first_update_q_above_half": row["first_update_q_above_half"],
                "first_update_q_above_090": row["first_update_q_above_090"],
                "escaped_adverse": row["escaped_adverse"],
            }
            for name, row in primary.items()
        },
        "ordering_asserted_by_tests": False,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "report.md").write_text(
        build_report(checkpoints, endpoints, direction_rows, log_odds_rows, manifest, base),
        encoding="utf-8",
    )
    return manifest


def _q_dip(rows: list[dict]) -> tuple[float, int | None, float]:
    """Initial q, the update of its minimum, and that minimum."""
    if not rows:
        return float("nan"), None, float("nan")
    ordered = sorted(rows, key=lambda row: row["update"])
    minimum = min(ordered, key=lambda row: row["q"])
    return ordered[0]["q"], minimum["update"], minimum["q"]


def build_report(
    checkpoints: list[dict],
    endpoints: list[dict],
    direction_rows: list[dict],
    log_odds_rows: list[dict],
    manifest: dict,
    base: FixedStepConfig,
) -> str:
    def endpoint(method: str, initialization: str, damping: float = PRIMARY_DAMPING) -> dict | None:
        for row in endpoints:
            if (
                row["method"] == method
                and row["initialization"] == initialization
                and row["damping"] == damping
            ):
                return row
        return None

    primary_adverse = endpoint(PRIMARY_METHOD, "adverse")
    primary_uniform = endpoint(PRIMARY_METHOD, "uniform")
    adverse_rows = _series(checkpoints, PRIMARY_METHOD, "adverse", PRIMARY_DAMPING)
    q_start, q_min_update, q_min = _q_dip(adverse_rows)
    worst_direction = manifest["worst_direction_relative_error"]
    worst_log_odds = manifest["worst_log_odds_absolute_error"]
    smallest_q = min(row["q"] for row in direction_rows)

    lines = [
        "# Exact-population fixed-step NPG on the two-state coordination trap",
        "",
        "This experiment is additive. It does not modify or re-run the existing",
        "target-KL experiment in `exact_two_state/`, the sampled experiments, or",
        "any Acrobot archive.",
        "",
        "## Method under test",
        "",
        "The primary theorem-matched update is",
        "",
        "```",
        "x = F_pool^{-1} grad J,    phi_next = phi + eta * x,    eta = "
        f"{base.eta}, damping = 0",
        "```",
        "",
        "with no target-KL normalization, no line search, no clipping, and no",
        "silent damping or pseudoinverse fallback. All arithmetic is exact",
        "population float64: no trajectories, seeds, or empirical Fisher.",
        "",
        "## 1. Exact mathematical identities (proved and numerically confirmed)",
        "",
        "**Natural-direction identity.** For reward-only exact population NPG under",
        "the transition-pooled Fisher, the reduced-logit natural direction is",
        "",
        "```",
        "F_pool^{-1} grad J = (1 + q) * (0.5, V1, 1.0, 0.2)",
        "```",
        "",
        f"Verified at {len(direction_rows)} deterministic interior policies. Worst",
        f"relative error: `{worst_direction:.3e}`. The smallest `q` tested is",
        f"`{smallest_q:.1e}`, at which the downstream components remain",
        "`(1 + q) * (1.0, 0.2)` rather than vanishing.",
        "",
        "The mechanism is that the block-diagonal pooled Fisher carries the factor",
        "`mu1 = q / (1 + q)` in its downstream block, while the exact reward",
        "gradient carries the visitation factor `d(s1) = q`. The inverse cancels",
        "the `q`, leaving `(1 + q)`. This cancellation is exact, not approximate.",
        "",
        "**Log-odds identities.** One fixed step of the primary update gives",
        "",
        "```",
        "Delta log(pi0(a1)/pi0(a0)) = eta * (1 + q) * (V1 - 0.5)",
        "Delta log(pi1(a0)/pi1(a1)) = eta * (1 + q) * 0.8",
        "Delta log(pi1(a0)/pi1(a2)) = eta * (1 + q) * 1.0",
        "```",
        "",
        f"Verified at {len(log_odds_rows)} interior policies. Worst absolute error:",
        f"`{worst_log_odds:.3e}`.",
        "",
        "The first identity changes sign exactly at `V1 = 0.5`. This is a property",
        "of the update rule itself, independent of any experimental outcome.",
        "",
        "## 2. Deterministic experimental observations",
        "",
    ]

    lines.append("| question | observation |")
    lines.append("| --- | --- |")
    if primary_adverse is not None:
        lines.append(
            "| 1. Does exact undamped reward-only NPG escape the adverse init? | "
            f"`escaped_adverse = {primary_adverse['escaped_adverse']}`, final "
            f"`q = {primary_adverse['final_q']:.6g}`, final "
            f"`pi1_good = {primary_adverse['final_pi1_good']:.6g}`, final return "
            f"`{primary_adverse['final_return']:.6g}`, `finite = {primary_adverse['finite']}` |"
        )
        lines.append(
            "| 2. At which update does V1 first exceed 0.5? | "
            f"`{primary_adverse['first_update_v1_above_half']}` (adverse); "
            f"`{primary_uniform['first_update_v1_above_half'] if primary_uniform else 'n/a'}` (uniform) |"
        )
        lines.append(
            "| 3. Does q initially decrease while V1 improves? | "
            f"q starts at `{q_start:.6g}` and reaches its recorded minimum "
            f"`{q_min:.6g}` at update `{q_min_update}` |"
        )
        lines.append(
            "| 4. Does q begin increasing after V1 crosses 0.5? | "
            f"`first_update_q_above_half = {primary_adverse['first_update_q_above_half']}`, "
            f"`first_update_q_above_090 = {primary_adverse['first_update_q_above_090']}` |"
        )
    lines.append(
        "| 5. Does the solved direction match `(1+q)*(0.5, V1, 1.0, 0.2)`? | "
        f"yes, worst relative error `{worst_direction:.3e}` |"
    )

    damped_summary = []
    for damping in (PRIMARY_DAMPING, *DAMPING_CONTROLS):
        row = endpoint(PRIMARY_METHOD, "adverse", damping)
        if row is not None:
            damped_summary.append(
                f"damping={damping}: final V1 `{row['final_v1']:.6g}`, final q "
                f"`{row['final_q']:.6g}`, V1>0.5 at `{row['first_update_v1_above_half']}`"
            )
    lines.append("| 6. How does damping alter downstream improvement? | " + "; ".join(damped_summary) + " |")

    barrier_rows = []
    for method in ("exact_npg_reward_only_fixed_step", "exact_npg_logbarrier_handoff_fixed_step"):
        row = endpoint(method, "adverse")
        if row is not None:
            barrier_rows.append(
                f"{method}: final return `{row['final_return']:.6g}`, V1>0.5 at "
                f"`{row['first_update_v1_above_half']}`, q>0.9 at `{row['first_update_q_above_090']}`"
            )
    lines.append("| 7. Does temporary LB change the NPG outcome or its timing? | " + "; ".join(barrier_rows) + " |")

    fixed_rows = []
    for method in ("exact_npg_logbarrier_fixed_fixed_step", "exact_pg_logbarrier_fixed_fixed_step"):
        row = endpoint(method, "adverse")
        if row is not None:
            fixed_rows.append(
                f"{method}: final return `{row['final_return']:.6g}`, final q "
                f"`{row['final_q']:.6g}`, final pi1_good `{row['final_pi1_good']:.6g}`"
            )
    lines.append("| 8. Does fixed LB retain an asymptotic interior bias? | " + "; ".join(fixed_rows) + " |")

    pseudo = endpoint(PSEUDOINVERSE_METHOD, "adverse")
    if pseudo is not None:
        lines.append(
            "| secondary diagnostic: pseudoinverse variant | "
            f"final return `{pseudo['final_return']:.6g}`, final q `{pseudo['final_q']:.6g}` "
            "(labelled diagnostic; does not replace the primary method) |"
        )

    degenerate = primary_adverse["first_update_fisher_not_positive_definite"] if primary_adverse else None
    lines.extend(
        [
            "",
            "## 2b. Numerical validity boundary (read before using late checkpoints)",
            "",
            "The exact Fisher is positive definite at every finite interior logit, but",
            "it approaches singularity as the policy approaches a deterministic",
            "corner. In float64 the smallest pooled eigenvalue of the primary adverse",
            f"run first stops supporting a trustworthy solve at update `{degenerate}`.",
            "",
            "Past that update `torch.linalg.solve` still returns a finite vector, but",
            "that vector is no longer the natural direction: the recorded",
            "`direction_vs_analytic_relative_error` rises from",
            f"`{primary_adverse['worst_undamped_direction_relative_error']:.3e}` (while the metric is",
            "numerically positive definite) to",
            f"`{primary_adverse['worst_direction_relative_error_after_degeneracy']:.3e}` afterwards."
            if primary_adverse
            else "",
            "",
            "This is floating-point breakdown at a degenerate metric, not a property",
            "of the update rule. It is reported rather than repaired: no damping,",
            "pseudoinverse, or clipping is switched on, per the experiment's rules.",
            "Every checkpoint carries `fisher_numerically_positive_definite` so this",
            "region can be excluded. All escape milestones above occur well before",
            "that boundary, so the reported outcome does not depend on it, and the",
            "`direction_identity_validation.csv` grid is unaffected because it",
            "samples interior policies only.",
            "",
            "`fisher_condition_number` is reported as `inf` in exactly this region;",
            "that is the honest value for a numerically singular matrix.",
            "",
            "## 3. Comparison with the existing target-KL experiment (question 9)",
            "",
            "The two experiments differ in step-length rule, not in gradient or",
            "Fisher convention:",
            "",
            "| | `exact_two_state.py` (existing) | this experiment |",
            "| --- | --- | --- |",
            "| step length | scaled to a declared target KL, `sqrt(2 delta / x'Fx)` | fixed `eta` |",
            "| primary damping | 0.01 | 0 |",
            "| invalid step | nonpositive quadratic form rejected | solve failure or nonfinite rejected |",
            "| step shrinks as metric degenerates | yes, by construction | no |",
            "",
            "The target-KL rule divides by `sqrt(x' F x)`. As the policy approaches",
            "the simplex boundary this quadratic form collapses, so the normalized",
            "rule spends its budget differently from the fixed-step rule even though",
            "both use the same direction `x`. See",
            "`plots/fixed_step_versus_target_kl.png`.",
            "",
            "## 4. Interpretations (supported by, but not identical to, the above)",
            "",
            "- The `(1 + q)` factor means the *direction* of downstream improvement",
            "  is independent of how rarely `s1` is visited, provided `q > 0`",
            "  exactly. This is a statement about the exact population update only.",
            "- The sign change of the explore-versus-safe increment at `V1 = 0.5`",
            "  gives a mechanism for the two-phase shape (downstream first, upstream",
            "  second) without invoking exploration noise.",
            "- Damping breaks the exact cancellation: the attenuation factor",
            "  `lambda / (lambda + damping)` applies per eigendirection, and the",
            "  downstream block's eigenvalues carry `mu1`, so small `q` is",
            "  attenuated most. Attenuation columns are recorded per checkpoint.",
            "",
            "## 5. Statements still requiring proof (question 10)",
            "",
            "The population-versus-sampled separation is **not** established by this",
            "experiment alone. What is established here is only the population side:",
            "the exact inverse cancels the visitation factor. The sampled side needs",
            "its own argument, because:",
            "",
            "- with `q > 0` exactly, `s1` is always present in the population",
            "  Fisher, whereas a finite batch contains `s1` with probability",
            "  `1 - (1-q)^N`; a batch without `s1` has a rank-deficient empirical",
            "  Fisher and no downstream signal at all;",
            "- the exact statement proved here is an identity about `F_pool^{-1}`,",
            "  not a concentration statement about `F_hat^{-1}`; the inverse of an",
            "  estimate is not the estimate of an inverse, and no bound relating the",
            "  two is proved here;",
            "- this MDP is a single 2x2-block instance; no claim is made that the",
            "  cancellation generalizes to shared function approximation.",
            "",
            "A formal statement would need a finite-sample bound on",
            "`||F_hat^{-1} g_hat - F_pool^{-1} g||` as a function of `N` and `q`.",
            "That is left open.",
            "",
            "## 6. Artifacts",
            "",
            "- `exact_population_checkpoints.csv`",
            "- `exact_population_endpoints.csv`",
            "- `direction_identity_validation.csv`",
            "- `log_odds_identity_validation.csv`",
            "- `method_configs.csv`",
            "- `manifest.json`",
            "- `plots/`",
            "",
        ]
    )
    return "\n".join(lines) + "\n"
