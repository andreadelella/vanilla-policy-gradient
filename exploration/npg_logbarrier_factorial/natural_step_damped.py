"""Natural step whose KL scaling uses the damped Fisher.

``natural_step.target_kl_natural_step`` solves with ``F + lambda I`` but measures
the quadratic form with the undamped ``F``. On a softmax policy that saturates,
``F -> 0`` while the form sits in the denominator of

    scale = sqrt(2 * target_kl / quad)

so a vanishing metric *inflates* the step: bigger step -> sharper policy ->
smaller ``F`` -> bigger step. The observed run reached ``|phi| ~ 1e6`` and a true
KL of ``inf`` while ``predicted_kl`` still read exactly ``1e-3``.

Using the same matrix in both places removes that inversion. As ``F -> 0`` the
direction tends to ``g / lambda`` and ``quad -> |g|^2 / lambda``, so

    |step| -> sqrt(2 * target_kl / lambda)

which is bounded and independent of the gradient.

This changes only the step *length*. The direction is the same solve, so the
geometry under study -- the direction of ``F^{-1} g``, its angle to the Euclidean
gradient, the conditioning of ``F`` -- is untouched; the cosine between the
undamped and damped steps is 1 to machine precision. There is no line search and
no trust region: this stays a natural-gradient method, not TRPO.

The returned object is the original :class:`NaturalStepResult`, so every existing
diagnostic column keeps its meaning. ``predicted_kl`` is still reported against
the undamped ``F``, which is the honest second-order KL estimate for the step
actually taken.
"""

from __future__ import annotations

import torch

from .natural_step import NaturalStepResult, _condition_number


def damped_target_kl_natural_step(
    gradient: torch.Tensor,
    fisher: torch.Tensor,
    *,
    damping: float,
    target_kl: float,
) -> NaturalStepResult:
    """Scale the damped natural direction to a target KL under ``F + lambda I``.

    Drop-in replacement for :func:`natural_step.target_kl_natural_step` with an
    identical signature and return type.
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
    # The one substantive difference: the scaling form is measured under the same
    # damped matrix used for the solve.
    quadratic = torch.dot(direction, damped @ direction)
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
    # Reachable only when the gradient itself has underflowed to zero, i.e. the
    # policy is fully saturated and there is no signal left to follow.
    if not torch.isfinite(quadratic) or quadratic_value <= 0.0:
        return NaturalStepResult(
            step=torch.zeros_like(direction), valid=False,
            invalid_reason="nonpositive_or_nonfinite_damped_quadratic_form",
            predicted_kl=float("nan"), scale_factor=float("nan"), **base,
        )

    scale = torch.sqrt(torch.as_tensor(2.0 * target_kl, dtype=fisher.dtype) / quadratic)
    step = scale * direction
    # Reported against the undamped Fisher: this is the second-order KL estimate
    # for the step taken, and comparing it to the realized KL shows how far the
    # quadratic model can be trusted.
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
