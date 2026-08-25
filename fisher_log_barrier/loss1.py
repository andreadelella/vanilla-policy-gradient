"""Loss-Function (1) surrogate for a trajectory-score Fisher log barrier.

For complete trajectories ``tau_k``, this module constructs

``ell_k = sum_t log pi(a_kt | s_kt)`` and ``z_k = grad ell_k``,

then estimates ``F = mean_k z_k z_k.T``.  The strict barrier domain is
``F - mu I > 0``.  This is not the statewise categorical action barrier in
``log_barrier.acrobot.barrier``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.func import functional_call, grad_and_value, vmap


SCORE_BACKENDS = ("vmap", "loop")


@dataclass(frozen=True)
class FisherLogDetDiagnostics:
    parameter_count: int
    trajectory_count: int
    gradient_trajectory_count: int
    separate_fisher_batch: bool
    score_backend: str
    rank: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    mu: float
    logdet_margin: float
    surrogate_value: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FisherInverseEstimate:
    """Detached inverse margin and serializable Fisher diagnostics."""

    inverse_margin: torch.Tensor
    parameter_count: int
    trajectory_count: int
    rank: int
    minimum_eigenvalue: float
    maximum_eigenvalue: float
    mu: float
    logdet_margin: float
    score_backend: str

    def to_dict(self) -> dict:
        return {
            "parameter_count": self.parameter_count,
            "trajectory_count": self.trajectory_count,
            "rank": self.rank,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "maximum_eigenvalue": self.maximum_eigenvalue,
            "mu": self.mu,
            "logdet_margin": self.logdet_margin,
            "score_backend": self.score_backend,
        }


class FisherLogDetDomainError(ValueError):
    """Raised when the empirical Fisher is outside ``F - mu I > 0``."""

    def __init__(
        self,
        *,
        parameter_count: int,
        trajectory_count: int,
        rank: int,
        minimum_eigenvalue: float,
        maximum_eigenvalue: float,
        mu: float,
    ) -> None:
        self.parameter_count = parameter_count
        self.trajectory_count = trajectory_count
        self.rank = rank
        self.minimum_eigenvalue = minimum_eigenvalue
        self.maximum_eigenvalue = maximum_eigenvalue
        self.mu = mu
        super().__init__(
            "trajectory Fisher is outside the strict log-barrier domain "
            "F_hat - mu I > 0: "
            f"parameters={parameter_count}, trajectories={trajectory_count}, "
            f"rank={rank}, lambda_min={minimum_eigenvalue:.9e}, "
            f"lambda_max={maximum_eigenvalue:.9e}, mu={mu:.9e}"
        )

    def to_dict(self) -> dict:
        return {
            "parameter_count": self.parameter_count,
            "trajectory_count": self.trajectory_count,
            "rank": self.rank,
            "minimum_eigenvalue": self.minimum_eigenvalue,
            "maximum_eigenvalue": self.maximum_eigenvalue,
            "mu": self.mu,
            "failure_reason": str(self),
        }


def _trajectory_log_probabilities(
    policy: torch.nn.Module,
    trajectories,
    device: torch.device,
) -> torch.Tensor:
    """Return one summed policy log-probability for each trajectory."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    totals = []
    for index, trajectory in enumerate(trajectories):
        if not trajectory.states or not trajectory.actions:
            raise ValueError(f"trajectory {index} is empty")
        if len(trajectory.states) != len(trajectory.actions):
            raise ValueError(f"trajectory {index} has mismatched states and actions")
        states = torch.as_tensor(
            np.asarray(trajectory.states, dtype=np.float32),
            dtype=next(policy.parameters()).dtype,
            device=device,
        )
        actions = torch.as_tensor(np.asarray(trajectory.actions), device=device)
        totals.append(policy.log_prob(states, actions).sum())
    return torch.stack(totals)


def _flatten_parameter_gradients(
    gradients: tuple[torch.Tensor | None, ...],
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
) -> torch.Tensor:
    """Flatten gradients in the stable order from ``policy.named_parameters``."""

    pieces = []
    for gradient, (name, parameter) in zip(gradients, named_parameters, strict=True):
        if gradient is None:
            raise RuntimeError(f"trajectory log-probability does not depend on parameter {name!r}")
        if gradient.shape != parameter.shape:
            raise RuntimeError(f"gradient shape mismatch for parameter {name!r}")
        pieces.append(gradient.reshape(-1))
    return torch.cat(pieces)


def _detached_spectrum(
    fisher: torch.Tensor,
    scores: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    matrix = fisher.detach().to(device="cpu", dtype=torch.float64)
    eigenvalues = torch.linalg.eigvalsh(0.5 * (matrix + matrix.T))
    score_matrix = scores.detach().to(device="cpu", dtype=torch.float64)
    rank = int(torch.linalg.matrix_rank(score_matrix))
    return eigenvalues, rank


def _trajectory_scores_loop(
    policy: torch.nn.Module,
    trajectories,
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
    device: torch.device,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return trajectory log probabilities and their parameter scores."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    parameters = tuple(parameter for _, parameter in named_parameters)
    if not create_graph:
        detached_log_probs = []
        detached_scores = []
        for trajectory in trajectories:
            trajectory_log_prob = _trajectory_log_probabilities(
                policy,
                [trajectory],
                device,
            )[0]
            gradients = torch.autograd.grad(
                trajectory_log_prob,
                parameters,
                create_graph=False,
                retain_graph=False,
                allow_unused=True,
            )
            detached_log_probs.append(trajectory_log_prob.detach())
            detached_scores.append(
                _flatten_parameter_gradients(gradients, named_parameters).detach()
            )
        scores = torch.stack(detached_scores)
        if not bool(torch.isfinite(scores).all()):
            raise FloatingPointError("non-finite trajectory policy scores")
        return torch.stack(detached_log_probs), scores

    trajectory_log_probs = _trajectory_log_probabilities(policy, trajectories, device)
    score_rows = []
    for trajectory_log_prob in trajectory_log_probs:
        gradients = torch.autograd.grad(
            trajectory_log_prob,
            parameters,
            create_graph=create_graph,
            retain_graph=create_graph,
            allow_unused=True,
        )
        score_rows.append(_flatten_parameter_gradients(gradients, named_parameters))
    scores = torch.stack(score_rows)
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("non-finite trajectory policy scores")
    return trajectory_log_probs, scores


class _PolicyLogProbability(torch.nn.Module):
    """Expose ``policy.log_prob`` as ``forward`` for ``functional_call``."""

    def __init__(self, policy: torch.nn.Module) -> None:
        super().__init__()
        self.policy = policy

    def forward(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.policy.log_prob(states, actions)


def _padded_trajectory_tensors(
    trajectories,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pad complete trajectories for batched score computation."""

    if not trajectories:
        raise ValueError("at least one trajectory is required")
    for index, trajectory in enumerate(trajectories):
        if not trajectory.states or not trajectory.actions:
            raise ValueError(f"trajectory {index} is empty")
        if len(trajectory.states) != len(trajectory.actions):
            raise ValueError(f"trajectory {index} has mismatched states and actions")

    trajectory_count = len(trajectories)
    maximum_length = max(len(trajectory.states) for trajectory in trajectories)
    state_shape = np.asarray(trajectories[0].states[0]).shape
    action_example = np.asarray(trajectories[0].actions[0])
    action_shape = action_example.shape
    states = np.zeros(
        (trajectory_count, maximum_length, *state_shape),
        dtype=np.float64 if dtype == torch.float64 else np.float32,
    )
    actions = np.zeros(
        (trajectory_count, maximum_length, *action_shape),
        dtype=action_example.dtype,
    )
    mask = np.zeros((trajectory_count, maximum_length), dtype=np.bool_)
    for index, trajectory in enumerate(trajectories):
        length = len(trajectory.states)
        states[index, :length] = np.asarray(trajectory.states)
        actions[index, :length] = np.asarray(trajectory.actions)
        mask[index, :length] = True
    return (
        torch.as_tensor(states, dtype=dtype, device=device),
        torch.as_tensor(actions, device=device),
        torch.as_tensor(mask, device=device),
    )


def _trajectory_scores_vmap(
    policy: torch.nn.Module,
    trajectories,
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
    device: torch.device,
    *,
    create_graph: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorize complete-trajectory parameter scores over a padded batch."""

    dtype = named_parameters[0][1].dtype
    states, actions, mask = _padded_trajectory_tensors(
        trajectories,
        dtype=dtype,
        device=device,
    )
    wrapper = _PolicyLogProbability(policy)
    parameters = {
        f"policy.{name}": parameter if create_graph else parameter.detach()
        for name, parameter in named_parameters
    }
    buffers = {
        f"policy.{name}": buffer.detach()
        for name, buffer in policy.named_buffers()
    }

    def trajectory_log_probability(functional_parameters, states_i, actions_i, mask_i):
        log_probabilities = functional_call(
            wrapper,
            {**functional_parameters, **buffers},
            (states_i, actions_i),
        )
        return (log_probabilities * mask_i).sum()

    score_function = grad_and_value(trajectory_log_probability, argnums=0)
    per_trajectory_gradients, trajectory_log_probs = vmap(
        score_function,
        in_dims=(None, 0, 0, 0),
    )(parameters, states, actions, mask)
    scores = torch.cat(
        [
            per_trajectory_gradients[f"policy.{name}"].reshape(len(trajectories), -1)
            for name, _ in named_parameters
        ],
        dim=1,
    )
    if not create_graph:
        trajectory_log_probs = trajectory_log_probs.detach()
        scores = scores.detach()
    if not bool(torch.isfinite(scores).all()):
        raise FloatingPointError("non-finite trajectory policy scores")
    return trajectory_log_probs, scores


def _trajectory_scores(
    policy: torch.nn.Module,
    trajectories,
    named_parameters: tuple[tuple[str, torch.nn.Parameter], ...],
    device: torch.device,
    *,
    create_graph: bool,
    backend: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if backend == "vmap":
        return _trajectory_scores_vmap(
            policy,
            trajectories,
            named_parameters,
            device,
            create_graph=create_graph,
        )
    if backend == "loop":
        return _trajectory_scores_loop(
            policy,
            trajectories,
            named_parameters,
            device,
            create_graph=create_graph,
        )
    raise ValueError(f"score backend must be one of {SCORE_BACKENDS}, got {backend!r}")


def estimate_trajectory_fisher_inverse(
    policy: torch.nn.Module,
    trajectories,
    *,
    mu: float,
    score_backend: str = "vmap",
    device: torch.device | str | None = None,
) -> FisherInverseEstimate:
    """Estimate and invert ``F_hat - mu I`` without retaining derivative graphs."""

    if not np.isfinite(mu) or mu < 0.0:
        raise ValueError("mu must be finite and non-negative")
    named_parameters = tuple(
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    )
    if not named_parameters:
        raise ValueError("policy has no trainable parameters")
    inferred_device = named_parameters[0][1].device
    device = inferred_device if device is None else torch.device(device)
    if any(parameter.device != device for _, parameter in named_parameters):
        raise ValueError("all trainable policy parameters must be on the requested device")

    _, fisher_scores = _trajectory_scores(
        policy,
        trajectories,
        named_parameters,
        device,
        create_graph=False,
        backend=score_backend,
    )
    fisher_scores = fisher_scores.detach().to(torch.float64)
    trajectory_count, parameter_count = fisher_scores.shape
    fisher = fisher_scores.T @ fisher_scores / trajectory_count
    fisher = 0.5 * (fisher + fisher.T)
    eigenvalues, rank = _detached_spectrum(fisher, fisher_scores)
    minimum_eigenvalue = float(eigenvalues[0])
    maximum_eigenvalue = float(eigenvalues[-1])
    identity = torch.eye(parameter_count, dtype=fisher.dtype, device=fisher.device)
    margin = fisher - float(mu) * identity
    factor, info = torch.linalg.cholesky_ex(margin)
    if (
        rank < parameter_count
        or minimum_eigenvalue <= float(mu)
        or int(info.item()) != 0
    ):
        raise FisherLogDetDomainError(
            parameter_count=parameter_count,
            trajectory_count=trajectory_count,
            rank=rank,
            minimum_eigenvalue=minimum_eigenvalue,
            maximum_eigenvalue=maximum_eigenvalue,
            mu=float(mu),
        )
    inverse_margin = torch.cholesky_inverse(factor).detach()
    logdet_margin = float(
        (2.0 * torch.log(torch.diagonal(factor))).sum().detach().cpu()
    )
    return FisherInverseEstimate(
        inverse_margin=inverse_margin,
        parameter_count=parameter_count,
        trajectory_count=trajectory_count,
        rank=rank,
        minimum_eigenvalue=minimum_eigenvalue,
        maximum_eigenvalue=maximum_eigenvalue,
        mu=float(mu),
        logdet_margin=logdet_margin,
        score_backend=score_backend,
    )


def trajectory_fisher_logdet_surrogate(
    policy: torch.nn.Module,
    trajectories,
    *,
    mu: float,
    fisher_trajectories=None,
    score_backend: str = "vmap",
    device: torch.device | str | None = None,
) -> tuple[torch.Tensor, FisherLogDetDiagnostics]:
    """Return the differentiable Loss-Function (1) barrier surrogate.

    ``fisher_trajectories`` estimates the Fisher and may be a larger batch than
    ``trajectories``, which estimates the outer Loss-Function (1) expectation.
    The Fisher scores do not retain a second-order graph because the inverse of
    ``F_hat - mu I`` is detached. The outer scores do retain it so the direct
    derivative of ``Tr(A B(theta))`` is preserved.
    """

    named_parameters = tuple(
        (name, parameter)
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad
    )
    if not named_parameters:
        raise ValueError("policy has no trainable parameters")
    inferred_device = named_parameters[0][1].device
    device = inferred_device if device is None else torch.device(device)
    if any(parameter.device != device for _, parameter in named_parameters):
        raise ValueError("all trainable policy parameters must be on the requested device")

    fisher_samples = trajectories if fisher_trajectories is None else fisher_trajectories
    separate_fisher_batch = fisher_samples is not trajectories
    estimate = estimate_trajectory_fisher_inverse(
        policy,
        fisher_samples,
        mu=mu,
        score_backend=score_backend,
        device=device,
    )
    trajectory_log_probs, scores = _trajectory_scores(
        policy,
        trajectories,
        named_parameters,
        device,
        create_graph=True,
        backend=score_backend,
    )
    scores = scores.to(torch.float64)
    trace_inverse = torch.trace(estimate.inverse_margin)
    quadratic = torch.einsum(
        "ni,ij,nj->n",
        scores,
        estimate.inverse_margin,
        scores,
    )
    b_values = quadratic - float(mu) * trace_inverse
    surrogate = (trajectory_log_probs * b_values.detach() + b_values).mean()
    if not bool(torch.isfinite(surrogate)):
        raise FloatingPointError("non-finite Fisher log-determinant surrogate")
    diagnostics = FisherLogDetDiagnostics(
        parameter_count=estimate.parameter_count,
        trajectory_count=estimate.trajectory_count,
        gradient_trajectory_count=len(trajectories),
        separate_fisher_batch=separate_fisher_batch,
        score_backend=score_backend,
        rank=estimate.rank,
        minimum_eigenvalue=estimate.minimum_eigenvalue,
        maximum_eigenvalue=estimate.maximum_eigenvalue,
        mu=estimate.mu,
        logdet_margin=estimate.logdet_margin,
        surrogate_value=float(surrogate.detach().cpu()),
    )
    return surrogate, diagnostics
