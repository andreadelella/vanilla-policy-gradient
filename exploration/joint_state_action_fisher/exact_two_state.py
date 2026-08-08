"""Deterministic Euclidean-gradient experiment for exact joint Fisher geometry."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Literal

import torch

from exploration.tabular_mdp.geometry import reduced_categorical_fisher
from exploration.tabular_mdp.model import (
    DTYPE,
    TwoStepTrap,
    as_phi,
    probabilities_from_reduced_logits,
    transition_pool_weights,
)

from .definitions import q_gradient
from .geometry import (
    joint_logdet_closed_form,
    joint_logdet_gradient_analytic,
    joint_state_action_fisher_decomposed,
    joint_visitation_contribution,
    pooled_policy_fisher_closed_form,
    pooled_policy_logdet,
    pooled_policy_logdet_gradient_analytic,
    state_distribution_fisher_closed_form,
)


MethodName = Literal[
    "reward_only",
    "statewise_conditional_barrier",
    "pooled_policy_logdet",
    "joint_state_action_logdet",
    "state_distribution_only",
    "joint_correction_only",
]

ALL_METHODS: tuple[MethodName, ...] = (
    "reward_only",
    "statewise_conditional_barrier",
    "pooled_policy_logdet",
    "joint_state_action_logdet",
    "state_distribution_only",
    "joint_correction_only",
)
REGULARIZED_METHODS: tuple[MethodName, ...] = ALL_METHODS[1:]
PRIMARY_METHODS: tuple[MethodName, ...] = ALL_METHODS[:4]

INITIALIZATIONS: dict[str, tuple[float, float, float, float]] = {
    "uniform": (0.0, 0.0, 0.0, 0.0),
    "adverse": (2.0, -2.0, -2.0, 2.0),
}


@dataclass(frozen=True)
class ExactRunConfig:
    method: MethodName
    protocol: str
    initialization: str
    alpha: float
    beta: float
    updates: int
    kappa: float | None = None

    def validate(self) -> None:
        if self.method not in ALL_METHODS:
            raise ValueError(f"unknown method: {self.method}")
        if self.initialization not in INITIALIZATIONS:
            raise ValueError(f"unknown initialization: {self.initialization}")
        if not math.isfinite(self.alpha) or self.alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if not math.isfinite(self.beta) or self.beta < 0:
            raise ValueError("beta must be finite and nonnegative")
        if not isinstance(self.updates, int) or isinstance(self.updates, bool) or self.updates < 1:
            raise ValueError("updates must be a positive integer")
        if self.method == "reward_only" and self.beta != 0.0:
            raise ValueError("reward_only requires beta=0")

    @property
    def run_id(self) -> str:
        beta = format(self.beta, ".12g").replace("-", "m").replace(".", "p")
        return f"{self.protocol}__{self.initialization}__{self.method}__beta_{beta}"


@dataclass(frozen=True)
class ExactRunResult:
    config: ExactRunConfig
    checkpoints: tuple[dict[str, object], ...]
    finite: bool

    @property
    def endpoint(self) -> dict[str, object]:
        return self.checkpoints[-1]


def _barrier_primitives(phi: torch.Tensor) -> dict[str, torch.Tensor]:
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    mu0, mu1 = transition_pool_weights(phi)
    b0 = torch.log(pi0).sum()
    b1 = torch.log(pi1).sum()
    zeros = torch.zeros(2, dtype=DTYPE)
    grad_b0 = torch.cat((1.0 - 3.0 * pi0[:2], zeros))
    grad_b1 = torch.cat((zeros, 1.0 - 3.0 * pi1[:2]))
    detached_conditional = mu0 * grad_b0 + mu1 * grad_b1
    q = pi0[1]
    v = q_gradient(phi)
    grad_log_mu0 = -v / (1.0 + q)
    visitation_gradient = ((1.0 - 1.5 * q) / (q * (1.0 + q))) * v
    return {
        "b0": b0,
        "b1": b1,
        "detached_conditional_value": mu0 * b0 + mu1 * b1,
        "detached_conditional_gradient": detached_conditional,
        "pooled_policy_value": pooled_policy_logdet(phi),
        "pooled_policy_gradient": pooled_policy_logdet_gradient_analytic(phi),
        "joint_value": joint_logdet_closed_form(phi),
        "joint_gradient": joint_logdet_gradient_analytic(phi),
        "joint_visitation_value": joint_visitation_contribution(phi),
        "joint_visitation_gradient": visitation_gradient,
        "joint_correction_value": 0.5 * torch.log(mu0),
        "joint_correction_gradient": 0.5 * grad_log_mu0,
    }


def regularizer_gradient(phi, method: MethodName) -> torch.Tensor:
    if method == "reward_only":
        return torch.zeros(4, dtype=DTYPE)
    terms = _barrier_primitives(phi)
    mapping = {
        "statewise_conditional_barrier": terms["detached_conditional_gradient"],
        "pooled_policy_logdet": terms["pooled_policy_gradient"],
        "joint_state_action_logdet": terms["joint_gradient"],
        # This is the explicitly derived visitation contribution to the joint
        # logdet. It is not logdet(F_mu), since F_mu has rank one in 4D.
        "state_distribution_only": terms["joint_visitation_gradient"],
        "joint_correction_only": terms["joint_correction_gradient"],
    }
    return mapping[method]


def regularizer_value(phi, method: MethodName) -> torch.Tensor:
    if method == "reward_only":
        return torch.zeros((), dtype=DTYPE)
    terms = _barrier_primitives(phi)
    mapping = {
        "statewise_conditional_barrier": terms["detached_conditional_value"],
        "pooled_policy_logdet": terms["pooled_policy_value"],
        "joint_state_action_logdet": terms["joint_value"],
        "state_distribution_only": terms["joint_visitation_value"],
        "joint_correction_only": terms["joint_correction_value"],
    }
    return mapping[method]


def _cosine(left: torch.Tensor, right: torch.Tensor) -> tuple[float, bool]:
    left_norm = torch.linalg.vector_norm(left)
    right_norm = torch.linalg.vector_norm(right)
    if float(left_norm.item()) == 0.0 or float(right_norm.item()) == 0.0:
        return math.nan, False
    return float(torch.dot(left, right).item() / (left_norm.item() * right_norm.item())), True


def _matrix_metrics(prefix: str, matrix: torch.Tensor, *, rank_one: bool = False) -> dict[str, object]:
    eigenvalues = torch.linalg.eigvalsh(matrix)
    threshold = 1e-12 * max(1.0, float(eigenvalues[-1].item()))
    positive = eigenvalues[eigenvalues > threshold]
    rank = int(positive.numel())
    smallest_positive = float(positive[0].item()) if rank else math.nan
    condition = float((positive[-1] / positive[0]).item()) if rank == matrix.shape[0] else math.nan
    sign, logabsdet = torch.linalg.slogdet(matrix)
    logdet = float(logabsdet.item()) if float(sign.item()) == 1.0 and rank == matrix.shape[0] else math.nan
    pseudologdet = float(torch.log(positive).sum().item()) if rank else math.nan
    values: dict[str, object] = {
        f"{prefix}_rank": rank,
        f"{prefix}_trace": float(torch.trace(matrix).item()),
        f"{prefix}_smallest_positive_eigenvalue": smallest_positive,
        f"{prefix}_condition_number": condition,
        f"{prefix}_logdet": logdet,
        f"{prefix}_pseudologdet": pseudologdet if rank_one else math.nan,
    }
    for index, eigenvalue in enumerate(eigenvalues):
        values[f"{prefix}_eigenvalue_{index}"] = float(eigenvalue.item())
    return values


def _batched_matrix_metrics(prefix: str, matrix: torch.Tensor, *, rank_one: bool = False) -> dict[str, torch.Tensor]:
    """Vectorized counterpart of ``_matrix_metrics`` for complete trajectories."""
    eigenvalues = torch.linalg.eigvalsh(matrix)
    threshold = 1e-12 * torch.maximum(torch.ones_like(eigenvalues[..., -1]), eigenvalues[..., -1])
    positive_mask = eigenvalues > threshold[..., None]
    rank = positive_mask.sum(dim=-1)
    infinity = torch.full_like(eigenvalues, torch.inf)
    smallest = torch.where(positive_mask, eigenvalues, infinity).min(dim=-1).values
    smallest = torch.where(rank > 0, smallest, torch.full_like(smallest, torch.nan))
    largest = eigenvalues[..., -1]
    condition = torch.where(rank == matrix.shape[-1], largest / smallest, torch.full_like(smallest, torch.nan))
    sign, logabsdet = torch.linalg.slogdet(matrix)
    logdet = torch.where(
        (sign == 1) & (rank == matrix.shape[-1]), logabsdet, torch.full_like(logabsdet, torch.nan)
    )
    safe_logs = torch.where(positive_mask, torch.log(torch.clamp_min(eigenvalues, torch.finfo(DTYPE).tiny)), torch.zeros_like(eigenvalues))
    pseudologdet = safe_logs.sum(dim=-1)
    result = {
        f"{prefix}_rank": rank,
        f"{prefix}_trace": torch.diagonal(matrix, dim1=-2, dim2=-1).sum(dim=-1),
        f"{prefix}_smallest_positive_eigenvalue": smallest,
        f"{prefix}_condition_number": condition,
        f"{prefix}_logdet": logdet,
        f"{prefix}_pseudologdet": pseudologdet if rank_one else torch.full_like(logdet, torch.nan),
    }
    result.update({f"{prefix}_eigenvalue_{index}": eigenvalues[..., index] for index in range(matrix.shape[-1])})
    return result


def _batch_geometry(phi: torch.Tensor, config: ExactRunConfig) -> dict[str, torch.Tensor]:
    """Compute every per-update diagnostic in vectorized float64 form."""
    mdp = TwoStepTrap()
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    mu0, mu1 = transition_pool_weights(phi)
    q = pi0[..., 1]
    zeros2 = torch.zeros_like(pi0[..., :2])
    v = torch.cat(
        (
            torch.stack((-q * pi0[..., 0], q * (1.0 - q)), dim=-1),
            zeros2,
        ),
        dim=-1,
    )
    b0 = torch.log(pi0).sum(dim=-1)
    b1 = torch.log(pi1).sum(dim=-1)
    grad_b0 = torch.cat((1.0 - 3.0 * pi0[..., :2], zeros2), dim=-1)
    grad_b1 = torch.cat((zeros2, 1.0 - 3.0 * pi1[..., :2]), dim=-1)
    conditional_gradient = mu0[..., None] * grad_b0 + mu1[..., None] * grad_b1
    grad_log_mu0 = -v / (1.0 + q)[..., None]
    grad_log_mu1 = v / (q * (1.0 + q))[..., None]
    pooled_gradient = 0.5 * (grad_b0 + grad_b1) + grad_log_mu0 + grad_log_mu1
    correction_gradient = 0.5 * grad_log_mu0
    joint_gradient = pooled_gradient + correction_gradient
    visitation_gradient = ((1.0 - 1.5 * q) / (q * (1.0 + q)))[..., None] * v
    zero_gradient = torch.zeros_like(phi)
    gradient_mapping = {
        "reward_only": zero_gradient,
        "statewise_conditional_barrier": conditional_gradient,
        "pooled_policy_logdet": pooled_gradient,
        "joint_state_action_logdet": joint_gradient,
        "state_distribution_only": visitation_gradient,
        "joint_correction_only": correction_gradient,
    }
    method_gradient = gradient_mapping[config.method]
    reward_gradient = mdp.exact_reward_gradient(phi)
    applied_gradient = config.beta * method_gradient
    total_gradient = reward_gradient + applied_gradient

    rewards1 = torch.tensor(mdp.state1_rewards, dtype=DTYPE)
    value1 = (pi1 * rewards1).sum(dim=-1)
    pooled_value = 0.5 * (b0 + b1) + torch.log(mu0) + torch.log(mu1)
    joint_value = pooled_value + 0.5 * torch.log(2.0 * mu0)
    visitation_value = 1.5 * torch.log(mu0) + torch.log(mu1)
    correction_value = 0.5 * torch.log(mu0)
    method_value_mapping = {
        "reward_only": torch.zeros_like(q),
        "statewise_conditional_barrier": mu0 * b0 + mu1 * b1,
        "pooled_policy_logdet": pooled_value,
        "joint_state_action_logdet": joint_value,
        "state_distribution_only": visitation_value,
        "joint_correction_only": correction_value,
    }

    f0 = reduced_categorical_fisher(pi0)
    f1 = reduced_categorical_fisher(pi1)
    f_policy = torch.zeros((phi.shape[0], 4, 4), dtype=DTYPE)
    f_policy[..., :2, :2] = mu0[..., None, None] * f0
    f_policy[..., 2:, 2:] = mu1[..., None, None] * f1
    f_state = v[..., :, None] * v[..., None, :] / (q * (1.0 + q).square())[..., None, None]
    f_joint = f_policy + f_state

    def norm(vector: torch.Tensor) -> torch.Tensor:
        return torch.linalg.vector_norm(vector, dim=-1)

    def cosine(left: torch.Tensor, right: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        denominator = norm(left) * norm(right)
        defined = denominator > 0
        value = torch.where(
            defined,
            (left * right).sum(dim=-1) / torch.where(defined, denominator, torch.ones_like(denominator)),
            torch.full_like(denominator, torch.nan),
        )
        return value, defined

    reward_pooled_cosine, reward_pooled_defined = cosine(reward_gradient, pooled_gradient)
    reward_joint_cosine, reward_joint_defined = cosine(reward_gradient, joint_gradient)
    reward_correction_cosine, reward_correction_defined = cosine(reward_gradient, correction_gradient)
    reward_method_cosine, reward_method_defined = cosine(reward_gradient, method_gradient)
    pooled_joint_cosine, pooled_joint_defined = cosine(pooled_gradient, joint_gradient)
    old_norm = norm(pooled_gradient)
    correction_norm = norm(correction_gradient)

    data: dict[str, torch.Tensor] = {
        **{f"phi_{index}": phi[..., index] for index in range(4)},
        "return": mdp.exact_return(phi),
        "q": q,
        "p_good": pi1[..., 0],
        "value_s1": value1,
        "value_s1_minus_safe": value1 - mdp.safe_reward,
        **{f"pi0_a{action}": pi0[..., action] for action in range(3)},
        **{f"pi1_a{action}": pi1[..., action] for action in range(3)},
        "min_pi0": pi0.min(dim=-1).values,
        "min_pi1": pi1.min(dim=-1).values,
        "mu0": mu0,
        "mu1": mu1,
        "b0": b0,
        "b1": b1,
        "pooled_policy_logdet_objective": pooled_value,
        "joint_state_action_logdet_objective": joint_value,
        "joint_visitation_component": visitation_value,
        "joint_correction": correction_value,
        "method_regularizer_value": method_value_mapping[config.method],
        "reward_gradient_norm": norm(reward_gradient),
        "pooled_policy_gradient_norm": old_norm,
        "joint_gradient_norm": norm(joint_gradient),
        "joint_correction_gradient_norm": correction_norm,
        "joint_correction_to_pooled_norm_ratio": torch.where(
            old_norm > 0, correction_norm / old_norm, torch.full_like(old_norm, torch.nan)
        ),
        "method_regularizer_gradient_norm": norm(method_gradient),
        "applied_regularizer_gradient_norm": norm(applied_gradient),
        "total_gradient_norm": norm(total_gradient),
        "cosine_reward_pooled": reward_pooled_cosine,
        "cosine_reward_pooled_defined": reward_pooled_defined,
        "cosine_reward_joint": reward_joint_cosine,
        "cosine_reward_joint_defined": reward_joint_defined,
        "cosine_reward_correction": reward_correction_cosine,
        "cosine_reward_correction_defined": reward_correction_defined,
        "cosine_reward_method": reward_method_cosine,
        "cosine_reward_method_defined": reward_method_defined,
        "cosine_pooled_joint": pooled_joint_cosine,
        "cosine_pooled_joint_defined": pooled_joint_defined,
        "finite": torch.isfinite(phi).all(dim=-1) & torch.isfinite(total_gradient).all(dim=-1),
    }
    for name, gradient in (
        ("reward_gradient", reward_gradient),
        ("pooled_policy_gradient", pooled_gradient),
        ("joint_gradient", joint_gradient),
        ("joint_correction_gradient", correction_gradient),
        ("method_regularizer_gradient", method_gradient),
        ("total_gradient", total_gradient),
    ):
        data.update({f"{name}_{index}": gradient[..., index] for index in range(4)})
    data.update(_batched_matrix_metrics("pooled_policy_fisher", f_policy))
    data.update(_batched_matrix_metrics("state_distribution_fisher", f_state, rank_one=True))
    data.update(_batched_matrix_metrics("joint_state_action_fisher", f_joint))
    return data


def checkpoint_metrics_batch(phi: torch.Tensor, config: ExactRunConfig) -> tuple[dict[str, object], ...]:
    data = _batch_geometry(phi, config)
    rows: list[dict[str, object]] = []
    for update in range(phi.shape[0]):
        row: dict[str, object] = {
            "run_id": config.run_id,
            "protocol": config.protocol,
            "initialization": config.initialization,
            "method": config.method,
            "alpha": config.alpha,
            "beta": config.beta,
            "kappa": config.kappa if config.kappa is not None else math.nan,
            "updates": config.updates,
            "update": update,
        }
        for key, values in data.items():
            item = values[update]
            if item.dtype == torch.bool:
                row[key] = bool(item.item())
            elif not item.dtype.is_floating_point:
                row[key] = int(item.item())
            else:
                row[key] = float(item.item())
        rows.append(row)
    return tuple(rows)


def checkpoint_metrics(phi: torch.Tensor, config: ExactRunConfig, update: int) -> dict[str, object]:
    mdp = TwoStepTrap()
    pi0, pi1 = probabilities_from_reduced_logits(phi)
    mu0, mu1 = transition_pool_weights(phi)
    rewards1 = torch.tensor(mdp.state1_rewards, dtype=DTYPE)
    value1 = torch.dot(pi1, rewards1)
    reward_gradient = mdp.exact_reward_gradient(phi)
    terms = _barrier_primitives(phi)
    method_gradient = regularizer_gradient(phi, config.method)
    applied_gradient = config.beta * method_gradient
    total_gradient = reward_gradient + applied_gradient

    reward_old_cosine, reward_old_defined = _cosine(reward_gradient, terms["pooled_policy_gradient"])
    reward_joint_cosine, reward_joint_defined = _cosine(reward_gradient, terms["joint_gradient"])
    reward_correction_cosine, reward_correction_defined = _cosine(
        reward_gradient, terms["joint_correction_gradient"]
    )
    reward_method_cosine, reward_method_defined = _cosine(reward_gradient, method_gradient)
    old_joint_cosine, old_joint_defined = _cosine(
        terms["pooled_policy_gradient"], terms["joint_gradient"]
    )

    f_policy = pooled_policy_fisher_closed_form(phi)
    f_state = state_distribution_fisher_closed_form(phi)
    f_joint = joint_state_action_fisher_decomposed(phi)
    correction_norm = float(torch.linalg.vector_norm(terms["joint_correction_gradient"]).item())
    old_norm = float(torch.linalg.vector_norm(terms["pooled_policy_gradient"]).item())
    correction_ratio = correction_norm / old_norm if old_norm > 0 else math.nan

    row: dict[str, object] = {
        "run_id": config.run_id,
        "protocol": config.protocol,
        "initialization": config.initialization,
        "method": config.method,
        "alpha": config.alpha,
        "beta": config.beta,
        "kappa": config.kappa if config.kappa is not None else math.nan,
        "updates": config.updates,
        "update": update,
        **{f"phi_{index}": float(component.item()) for index, component in enumerate(phi)},
        "return": float(mdp.exact_return(phi).item()),
        "q": float(pi0[1].item()),
        "p_good": float(pi1[0].item()),
        "value_s1": float(value1.item()),
        "value_s1_minus_safe": float((value1 - mdp.safe_reward).item()),
        **{f"pi0_a{action}": float(pi0[action].item()) for action in range(3)},
        **{f"pi1_a{action}": float(pi1[action].item()) for action in range(3)},
        "min_pi0": float(pi0.min().item()),
        "min_pi1": float(pi1.min().item()),
        "mu0": float(mu0.item()),
        "mu1": float(mu1.item()),
        "b0": float(terms["b0"].item()),
        "b1": float(terms["b1"].item()),
        "pooled_policy_logdet_objective": float(terms["pooled_policy_value"].item()),
        "joint_state_action_logdet_objective": float(terms["joint_value"].item()),
        "joint_visitation_component": float(terms["joint_visitation_value"].item()),
        "joint_correction": float(terms["joint_correction_value"].item()),
        "method_regularizer_value": float(regularizer_value(phi, config.method).item()),
        "reward_gradient_norm": float(torch.linalg.vector_norm(reward_gradient).item()),
        "pooled_policy_gradient_norm": old_norm,
        "joint_gradient_norm": float(torch.linalg.vector_norm(terms["joint_gradient"]).item()),
        "joint_correction_gradient_norm": correction_norm,
        "joint_correction_to_pooled_norm_ratio": correction_ratio,
        "method_regularizer_gradient_norm": float(torch.linalg.vector_norm(method_gradient).item()),
        "applied_regularizer_gradient_norm": float(torch.linalg.vector_norm(applied_gradient).item()),
        "total_gradient_norm": float(torch.linalg.vector_norm(total_gradient).item()),
        "cosine_reward_pooled": reward_old_cosine,
        "cosine_reward_pooled_defined": reward_old_defined,
        "cosine_reward_joint": reward_joint_cosine,
        "cosine_reward_joint_defined": reward_joint_defined,
        "cosine_reward_correction": reward_correction_cosine,
        "cosine_reward_correction_defined": reward_correction_defined,
        "cosine_reward_method": reward_method_cosine,
        "cosine_reward_method_defined": reward_method_defined,
        "cosine_pooled_joint": old_joint_cosine,
        "cosine_pooled_joint_defined": old_joint_defined,
        "finite": bool(torch.isfinite(phi).all() and torch.isfinite(total_gradient).all()),
    }
    for name, gradient in (
        ("reward_gradient", reward_gradient),
        ("pooled_policy_gradient", terms["pooled_policy_gradient"]),
        ("joint_gradient", terms["joint_gradient"]),
        ("joint_correction_gradient", terms["joint_correction_gradient"]),
        ("method_regularizer_gradient", method_gradient),
        ("total_gradient", total_gradient),
    ):
        row.update({f"{name}_{index}": float(component.item()) for index, component in enumerate(gradient)})
    row.update(_matrix_metrics("pooled_policy_fisher", f_policy))
    row.update(_matrix_metrics("state_distribution_fisher", f_state, rank_one=True))
    row.update(_matrix_metrics("joint_state_action_fisher", f_joint))
    return row


def train_exact(config: ExactRunConfig) -> ExactRunResult:
    config.validate()
    phi = as_phi(INITIALIZATIONS[config.initialization])
    mdp = TwoStepTrap()
    trajectory = [phi]
    finite = True
    for _ in range(config.updates):
        direction = mdp.exact_reward_gradient(phi) + config.beta * regularizer_gradient(phi, config.method)
        candidate = phi + config.alpha * direction
        if not bool(torch.isfinite(candidate).all()):
            finite = False
            break
        phi = candidate
        trajectory.append(phi)
    rows = checkpoint_metrics_batch(torch.stack(trajectory), config)
    finite = finite and all(bool(row["finite"]) for row in rows)
    return ExactRunResult(config, rows, finite)


def magnitude_matched_betas(
    initialization: str = "adverse", *, kappa: float = 1.0
) -> dict[MethodName, float]:
    if initialization not in INITIALIZATIONS:
        raise ValueError(f"unknown initialization: {initialization}")
    if not math.isfinite(kappa) or kappa <= 0:
        raise ValueError("kappa must be finite and positive")
    phi = as_phi(INITIALIZATIONS[initialization])
    reward_norm = torch.linalg.vector_norm(TwoStepTrap().exact_reward_gradient(phi))
    coefficients: dict[MethodName, float] = {}
    for method in REGULARIZED_METHODS:
        norm = torch.linalg.vector_norm(regularizer_gradient(phi, method))
        if float(norm.item()) == 0.0:
            raise ValueError(f"cannot magnitude-match zero gradient for {method}")
        coefficients[method] = float((kappa * reward_norm / norm).item())
    return coefficients


def build_run_configs(*, smoke: bool = False) -> tuple[ExactRunConfig, ...]:
    alpha = 0.05
    updates = 40 if smoke else 2000
    betas = (0.1,) if smoke else (0.01, 0.1, 0.2)
    configs: list[ExactRunConfig] = []
    for initialization in INITIALIZATIONS:
        configs.append(ExactRunConfig("reward_only", "same_beta", initialization, alpha, 0.0, updates))
        for beta in betas:
            for method in REGULARIZED_METHODS:
                configs.append(ExactRunConfig(method, "same_beta", initialization, alpha, beta, updates))
    configs.append(ExactRunConfig("reward_only", "magnitude_matched", "adverse", alpha, 0.0, updates, 1.0))
    for method, beta in magnitude_matched_betas().items():
        configs.append(ExactRunConfig(method, "magnitude_matched", "adverse", alpha, beta, updates, 1.0))
    return tuple(configs)


def vector_field_rows(*, beta: float = 0.1, grid_size: int = 17) -> tuple[dict[str, object], ...]:
    from exploration.tabular_mdp.model import phi_from_q_and_good

    if grid_size < 3:
        raise ValueError("grid_size must be at least three")
    mdp = TwoStepTrap()
    grid = torch.linspace(0.03, 0.97, grid_size, dtype=DTYPE)
    rows: list[dict[str, object]] = []
    for method in ("reward_only", "pooled_policy_logdet", "joint_state_action_logdet"):
        coefficient = 0.0 if method == "reward_only" else beta
        for q in grid:
            for good in grid:
                phi = phi_from_q_and_good(q, good)
                direction = mdp.exact_reward_gradient(phi) + coefficient * regularizer_gradient(phi, method)
                pi0, pi1 = probabilities_from_reduced_logits(phi)
                q_grad = q_gradient(phi)
                p = pi1[0]
                p_grad = torch.tensor((0.0, 0.0, p * (1.0 - p), -p * pi1[1]), dtype=DTYPE)
                dq = torch.dot(q_grad, direction)
                dp = torch.dot(p_grad, direction)
                pooled_gradient = regularizer_gradient(phi, "pooled_policy_logdet")
                joint_gradient = regularizer_gradient(phi, "joint_state_action_logdet")
                correction_gradient = regularizer_gradient(phi, "joint_correction_only")
                pooled_joint_cosine, cosine_defined = _cosine(pooled_gradient, joint_gradient)
                rows.append(
                    {
                        "method": method,
                        "beta": coefficient,
                        "q": float(q.item()),
                        "p_good": float(good.item()),
                        "dq_dt": float(dq.item()),
                        "dp_good_dt": float(dp.item()),
                        "speed": float(torch.sqrt(dq.square() + dp.square()).item()),
                        "pooled_regularizer_dq_dt": float(torch.dot(q_grad, pooled_gradient).item()),
                        "joint_regularizer_dq_dt": float(torch.dot(q_grad, joint_gradient).item()),
                        "joint_correction_dq_dt": float(torch.dot(q_grad, correction_gradient).item()),
                        "cosine_pooled_joint_regularizer": pooled_joint_cosine,
                        "cosine_pooled_joint_defined": cosine_defined,
                    }
                )
    return tuple(rows)
