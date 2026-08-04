"""Target-KL-normalized natural-gradient primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class NaturalStepResult:
    direction: torch.Tensor
    step: torch.Tensor
    valid: bool
    invalid_reason: str | None
    target_kl: float
    predicted_kl: float
    scale_factor: float
    gradient_norm: float
    natural_direction_norm: float
    quadratic_form: float
    damping: float
    solve_residual: float
    fisher_condition_number: float
    damped_condition_number: float

    def diagnostics(self) -> dict[str, float | bool | str | None]:
        result = asdict(self)
        result.pop("direction")
        result.pop("step")
        return result


def _condition_number(matrix: torch.Tensor) -> float:
    eigenvalues = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    largest = float(eigenvalues[-1]) if eigenvalues.numel() else 0.0
    threshold = 1e-12 * max(1.0, largest)
    positive = eigenvalues[eigenvalues > threshold]
    if positive.numel() < 1:
        return float("inf")
    return float(positive[-1] / positive[0])


def target_kl_natural_step(
    gradient: torch.Tensor,
    fisher: torch.Tensor,
    *,
    damping: float,
    target_kl: float,
) -> NaturalStepResult:
    """Solve a damped natural direction and scale it to a target local KL.

    The solve uses ``F + damping I``. The quadratic form and predicted KL use
    the undamped Fisher. Nonpositive or nonfinite forms are returned as invalid;
    no absolute value, line search, or silent fallback is used.
    """

    gradient = torch.as_tensor(gradient)
    fisher = torch.as_tensor(fisher, dtype=gradient.dtype, device=gradient.device)
    if gradient.ndim != 1 or fisher.shape != (gradient.numel(), gradient.numel()):
        raise ValueError("gradient and Fisher shapes are incompatible")
    if damping < 0.0 or target_kl <= 0.0:
        raise ValueError("damping must be nonnegative and target_kl positive")
    if not torch.isfinite(gradient).all() or not torch.isfinite(fisher).all():
        raise ValueError("gradient and Fisher must be finite")

    fisher = 0.5 * (fisher + fisher.T)
    identity = torch.eye(fisher.shape[0], dtype=fisher.dtype, device=fisher.device)
    damped = fisher + damping * identity
    try:
        direction = torch.linalg.solve(damped, gradient)
    except torch.linalg.LinAlgError:
        zero = torch.zeros_like(gradient)
        return NaturalStepResult(
            zero, zero, False, "damped_fisher_solve_failed", target_kl,
            float("nan"), float("nan"), float(gradient.norm()), float("nan"),
            float("nan"), damping, float("inf"), _condition_number(fisher),
            _condition_number(damped),
        )

    residual = float((damped @ direction - gradient).norm())
    quadratic = torch.dot(direction, fisher @ direction)
    quadratic_value = float(quadratic)
    base = dict(
        direction=direction.detach(),
        target_kl=target_kl,
        gradient_norm=float(gradient.norm()),
        natural_direction_norm=float(direction.norm()),
        quadratic_form=quadratic_value,
        damping=damping,
        solve_residual=residual,
        fisher_condition_number=_condition_number(fisher),
        damped_condition_number=_condition_number(damped),
    )
    if not torch.isfinite(quadratic) or quadratic_value <= 0.0:
        return NaturalStepResult(
            step=torch.zeros_like(direction), valid=False,
            invalid_reason="nonpositive_or_nonfinite_undamped_quadratic_form",
            predicted_kl=float("nan"), scale_factor=float("nan"), **base,
        )

    scale = torch.sqrt(torch.as_tensor(2.0 * target_kl, dtype=fisher.dtype) / quadratic)
    step = scale * direction
    predicted = 0.5 * torch.dot(step, fisher @ step)
    if not torch.isfinite(step).all() or not torch.isfinite(predicted):
        return NaturalStepResult(
            step=torch.zeros_like(direction), valid=False,
            invalid_reason="nonfinite_scaled_step", predicted_kl=float("nan"),
            scale_factor=float(scale), **base,
        )
    return NaturalStepResult(
        step=step.detach(), valid=True, invalid_reason=None,
        predicted_kl=float(predicted), scale_factor=float(scale), **base,
    )


def cosine(left: torch.Tensor, right: torch.Tensor) -> tuple[float, bool]:
    left = torch.as_tensor(left)
    right = torch.as_tensor(right, dtype=left.dtype, device=left.device)
    left_norm, right_norm = left.norm(), right.norm()
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        return float("nan"), False
    return float(torch.dot(left, right) / (left_norm * right_norm)), True


def flatten_parameters(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([parameter.detach().reshape(-1) for parameter in model.parameters()])


def set_parameters(model: torch.nn.Module, vector: torch.Tensor) -> None:
    vector = torch.as_tensor(vector)
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            count = parameter.numel()
            parameter.copy_(vector[offset:offset + count].reshape_as(parameter).to(parameter))
            offset += count
    if offset != vector.numel():
        raise ValueError("parameter vector has the wrong size")

