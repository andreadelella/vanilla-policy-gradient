"""Categorical and Gaussian policy-Fisher validation gate."""

from __future__ import annotations

import copy
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.distributions import Categorical, Normal
from torch.func import functional_call

from vpg.policy import GaussianPolicy, MLPSoftmaxPolicy

from .natural_step import cosine


DTYPE = torch.float64
SAMPLE_COUNTS = (1, 4, 16, 64, 256)


@dataclass(frozen=True)
class ParameterLayout:
    names: tuple[str, ...]
    shapes: tuple[torch.Size, ...]
    counts: tuple[int, ...]


def parameter_layout(model: torch.nn.Module) -> ParameterLayout:
    named = tuple(model.named_parameters())
    return ParameterLayout(
        tuple(name for name, _ in named),
        tuple(parameter.shape for _, parameter in named),
        tuple(parameter.numel() for _, parameter in named),
    )


def parameter_vector(model: torch.nn.Module) -> torch.Tensor:
    return torch.cat([
        parameter.detach().reshape(-1) for _, parameter in model.named_parameters()
    ])


def vector_parameters(vector: torch.Tensor, layout: ParameterLayout) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}
    offset = 0
    for name, shape, count in zip(layout.names, layout.shapes, layout.counts):
        result[name] = vector[offset:offset + count].reshape(shape)
        offset += count
    if offset != vector.numel():
        raise ValueError("parameter vector does not match layout")
    return result


def _functional_output(model, vector, layout, states):
    parameters = vector_parameters(vector, layout)
    buffers = dict(model.named_buffers())
    return functional_call(model, {**parameters, **buffers}, (states,))


def _score(vector, scalar_log_probability) -> torch.Tensor:
    return torch.autograd.grad(scalar_log_probability, vector, create_graph=False)[0]


def categorical_score_fisher(
    policy: torch.nn.Module,
    states: torch.Tensor,
    actions: torch.Tensor,
) -> torch.Tensor:
    model = copy.deepcopy(policy).to(dtype=DTYPE, device="cpu")
    states = states.detach().cpu().to(DTYPE)
    actions = actions.detach().cpu().long()
    if states.shape[0] != actions.shape[0]:
        raise ValueError("states and actions must have equal row counts")
    layout = parameter_layout(model)
    base = parameter_vector(model)
    rows = []
    for state, action in zip(states, actions):
        vector = base.detach().clone().requires_grad_(True)
        logits = _functional_output(model, vector, layout, state.unsqueeze(0)).squeeze(0)
        rows.append(_score(vector, Categorical(logits=logits).log_prob(action)))
    score_matrix = torch.stack(rows)
    return score_matrix.T @ score_matrix / states.shape[0]


def categorical_enumerated_fisher(policy: torch.nn.Module, states: torch.Tensor) -> torch.Tensor:
    model = copy.deepcopy(policy).to(dtype=DTYPE, device="cpu")
    states = states.detach().cpu().to(DTYPE)
    layout = parameter_layout(model)
    base = parameter_vector(model)
    with torch.no_grad():
        probabilities = torch.softmax(model(states), dim=-1)
    result = torch.zeros((base.numel(), base.numel()), dtype=DTYPE)
    for index, state in enumerate(states):
        for action in range(probabilities.shape[1]):
            vector = base.detach().clone().requires_grad_(True)
            logits = _functional_output(model, vector, layout, state.unsqueeze(0)).squeeze(0)
            score = _score(vector, Categorical(logits=logits).log_prob(torch.tensor(action)))
            result += probabilities[index, action] * torch.outer(score, score)
    return result / states.shape[0]


def categorical_kl_hessian(policy: torch.nn.Module, states: torch.Tensor) -> torch.Tensor:
    model = copy.deepcopy(policy).to(dtype=DTYPE, device="cpu")
    states = states.detach().cpu().to(DTYPE)
    layout = parameter_layout(model)
    base = parameter_vector(model)
    with torch.no_grad():
        reference_log_probabilities = torch.log_softmax(model(states), dim=-1)
        reference_probabilities = reference_log_probabilities.exp()

    def mean_forward_kl(vector):
        current_log = torch.log_softmax(_functional_output(model, vector, layout, states), dim=-1)
        return (reference_probabilities * (reference_log_probabilities - current_log)).sum(-1).mean()

    return torch.autograd.functional.hessian(mean_forward_kl, base)


def sampled_categorical_fisher(policy, states, samples_per_state: int, generator) -> torch.Tensor:
    model = copy.deepcopy(policy).to(dtype=DTYPE, device="cpu")
    states = states.detach().cpu().to(DTYPE)
    with torch.no_grad():
        probabilities = torch.softmax(model(states), dim=-1)
        actions = torch.multinomial(probabilities, samples_per_state, replacement=True, generator=generator)
    repeated_states = states.repeat_interleave(samples_per_state, dim=0)
    return categorical_score_fisher(model, repeated_states, actions.reshape(-1))


def gaussian_score_fisher(policy, states, raw_actions) -> torch.Tensor:
    model = copy.deepcopy(policy).to(dtype=DTYPE, device="cpu")
    states = states.detach().cpu().to(DTYPE)
    raw_actions = raw_actions.detach().cpu().to(DTYPE)
    layout = parameter_layout(model)
    base = parameter_vector(model)
    rows = []
    for state, action in zip(states, raw_actions):
        vector = base.detach().clone().requires_grad_(True)
        mean, std = _functional_output(model, vector, layout, state.unsqueeze(0))
        log_probability = Normal(mean.squeeze(0), std).log_prob(action).sum()
        rows.append(_score(vector, log_probability))
    scores = torch.stack(rows)
    return scores.T @ scores / states.shape[0]


def gaussian_kl_hessian(policy, states) -> torch.Tensor:
    model = copy.deepcopy(policy).to(dtype=DTYPE, device="cpu")
    states = states.detach().cpu().to(DTYPE)
    layout = parameter_layout(model)
    base = parameter_vector(model)
    with torch.no_grad():
        mean0, std0 = model(states)

    def mean_forward_kl(vector):
        mean, std = _functional_output(model, vector, layout, states)
        kl = torch.log(std / std0) + (std0.square() + (mean0 - mean).square()) / (2.0 * std.square()) - 0.5
        return kl.sum(-1).mean()

    return torch.autograd.functional.hessian(mean_forward_kl, base)


def gaussian_analytic_fisher(policy, states) -> torch.Tensor:
    """Action-integrated diagonal-Gaussian Fisher in parameter coordinates."""

    model = copy.deepcopy(policy).to(dtype=DTYPE, device="cpu")
    states = states.detach().cpu().to(DTYPE)
    layout = parameter_layout(model)
    base = parameter_vector(model)
    result = torch.zeros((base.numel(), base.numel()), dtype=DTYPE)
    for state in states:
        vector = base.detach().clone().requires_grad_(True)
        mean, std = _functional_output(model, vector, layout, state.unsqueeze(0))
        mean, std = mean.squeeze(0), std.reshape(-1)
        for coordinate in range(mean.numel()):
            mean_gradient = torch.autograd.grad(mean[coordinate], vector, retain_graph=True)[0]
            log_std_gradient = torch.autograd.grad(torch.log(std[coordinate]), vector, retain_graph=True)[0]
            result += torch.outer(mean_gradient, mean_gradient) / std[coordinate].square()
            result += 2.0 * torch.outer(log_std_gradient, log_std_gradient)
    return result / states.shape[0]


def sampled_gaussian_fisher(policy, states, samples_per_state, generator):
    model = copy.deepcopy(policy).to(dtype=DTYPE, device="cpu")
    states = states.detach().cpu().to(DTYPE)
    with torch.no_grad():
        mean, std = model(states)
        noise = torch.randn(
            (states.shape[0], samples_per_state, mean.shape[-1]),
            generator=generator, dtype=DTYPE,
        )
        raw_actions = mean[:, None, :] + std.reshape(1, 1, -1) * noise
    repeated_states = states.repeat_interleave(samples_per_state, dim=0)
    return gaussian_score_fisher(model, repeated_states, raw_actions.reshape(-1, mean.shape[-1]))


def _metrics(estimate, reference, probe_gradient):
    estimate = (0.5 * (estimate + estimate.T)).detach()
    reference = (0.5 * (reference + reference.T)).detach()
    difference = estimate - reference
    ref_norm = float(reference.norm())
    trace_ref = float(torch.trace(reference))
    eigen_est = torch.linalg.eigvalsh(estimate).flip(0)
    eigen_ref = torch.linalg.eigvalsh(reference).flip(0)
    leading = min(5, eigen_ref.numel())
    denom = torch.maximum(eigen_ref[:leading].abs(), torch.full((leading,), 1e-15, dtype=DTYPE))
    leading_error = float(torch.max(torch.abs(eigen_est[:leading] - eigen_ref[:leading]) / denom))
    damping = 1e-3
    identity = torch.eye(reference.shape[0], dtype=DTYPE)
    direction_est = torch.linalg.solve(estimate + damping * identity, probe_gradient)
    direction_ref = torch.linalg.solve(reference + damping * identity, probe_gradient)
    direction_cosine, _ = cosine(direction_est, direction_ref)
    return {
        "relative_frobenius_error": float(difference.norm()) / ref_norm if ref_norm else float("nan"),
        "trace_relative_error": abs(float(torch.trace(estimate)) - trace_ref) / abs(trace_ref) if trace_ref else float("nan"),
        "leading_eigenvalue_max_relative_error": leading_error,
        "natural_direction_cosine": direction_cosine,
        "maximum_absolute_entry_error": float(difference.abs().max()),
        "symmetry_residual": float((estimate - estimate.T).abs().max()),
        "minimum_eigenvalue": float(torch.linalg.eigvalsh(estimate)[0]),
        "psd_within_1e_minus_10": bool(float(torch.linalg.eigvalsh(estimate)[0]) >= -1e-10),
    }


def _write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot(rows, family, path):
    sampled = [row for row in rows if row["family"] == family and row["estimator"] == "sampled_score"]
    figure, axis = plt.subplots(figsize=(6, 4))
    axis.loglog(
        [row["samples_per_state"] for row in sampled],
        [row["relative_frobenius_error"] for row in sampled],
        marker="o",
    )
    axis.set_xlabel("action samples per state")
    axis.set_ylabel("relative Frobenius error")
    axis.set_title(f"{family} sampled-score Fisher convergence")
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_validation(output_directory: str | Path) -> dict:
    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        report = output / "validation_result.json"
        if report.exists():
            return json.loads(report.read_text(encoding="utf-8"))
        raise FileExistsError(f"nonempty incomplete validation directory: {output}")
    (output / "plots").mkdir(parents=True, exist_ok=True)
    torch.manual_seed(1729)

    categorical = MLPSoftmaxPolicy(3, 3, hidden_sizes=(4,)).to(DTYPE)
    categorical_states = torch.tensor(
        [[-0.7, 0.2, 1.1], [0.3, -1.2, 0.4], [1.0, 0.5, -0.2]], dtype=DTYPE,
    )
    categorical_before = parameter_vector(categorical).clone()
    categorical_kl = categorical_kl_hessian(categorical, categorical_states)
    categorical_exact = categorical_enumerated_fisher(categorical, categorical_states)
    probe = torch.linspace(-1.0, 1.0, categorical_kl.shape[0], dtype=DTYPE)
    categorical_rows = [{
        "family": "categorical", "estimator": "action_enumerated", "samples_per_state": "exact",
        **_metrics(categorical_exact, categorical_kl, probe),
    }]
    generator = torch.Generator(device="cpu").manual_seed(1730)
    for count in SAMPLE_COUNTS:
        estimate = sampled_categorical_fisher(categorical, categorical_states, count, generator)
        categorical_rows.append({
            "family": "categorical", "estimator": "sampled_score", "samples_per_state": count,
            **_metrics(estimate, categorical_kl, probe),
        })
    if not torch.equal(categorical_before, parameter_vector(categorical)):
        raise AssertionError("categorical Fisher construction changed policy parameters")

    gaussian = GaussianPolicy(3, 2, hidden_sizes=(4,), init_log_std=-0.4).to(DTYPE)
    gaussian_states = categorical_states.clone()
    gaussian_before = parameter_vector(gaussian).clone()
    gaussian_kl = gaussian_kl_hessian(gaussian, gaussian_states)
    gaussian_exact = gaussian_analytic_fisher(gaussian, gaussian_states)
    probe_g = torch.linspace(-0.5, 0.5, gaussian_kl.shape[0], dtype=DTYPE)
    gaussian_rows = [{
        "family": "gaussian", "estimator": "analytic_action_integrated", "samples_per_state": "exact",
        **_metrics(gaussian_exact, gaussian_kl, probe_g),
    }]
    generator_g = torch.Generator(device="cpu").manual_seed(1731)
    for count in SAMPLE_COUNTS:
        estimate = sampled_gaussian_fisher(gaussian, gaussian_states, count, generator_g)
        gaussian_rows.append({
            "family": "gaussian", "estimator": "sampled_score", "samples_per_state": count,
            **_metrics(estimate, gaussian_kl, probe_g),
        })
    if not torch.equal(gaussian_before, parameter_vector(gaussian)):
        raise AssertionError("Gaussian Fisher construction changed policy parameters")

    _write_csv(output / "categorical_validation.csv", categorical_rows)
    _write_csv(output / "gaussian_validation.csv", gaussian_rows)
    _plot(categorical_rows, "categorical", output / "plots" / "categorical_convergence.png")
    _plot(gaussian_rows, "gaussian", output / "plots" / "gaussian_convergence.png")

    categorical_exact_error = categorical_rows[0]["relative_frobenius_error"]
    gaussian_exact_error = gaussian_rows[0]["relative_frobenius_error"]
    passed = bool(categorical_exact_error <= 1e-9 and gaussian_exact_error <= 1e-9)
    result = {
        "schema_version": 1,
        "passed": passed,
        "dtype": "float64",
        "categorical_exact_relative_frobenius_error": categorical_exact_error,
        "gaussian_exact_relative_frobenius_error": gaussian_exact_error,
        "sample_counts": list(SAMPLE_COUNTS),
        "raw_gaussian_actions_used": True,
        "clipped_actions_used_for_likelihood": False,
        "policy_parameters_unchanged": True,
    }
    (output / "validation_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    (output / "validation_report.md").write_text(
        "# Policy-Fisher validation\n\n"
        f"Gate status: **{'PASS' if passed else 'FAIL'}**.\n\n"
        "The categorical action-enumerated Fisher and the analytic diagonal-Gaussian "
        "Fisher are compared with the Hessian of mean forward KL at the reference "
        "parameters. Sampled score Fishers use raw policy actions and are reported "
        "over increasing action samples per state. Main factorial runners must refuse "
        "to execute unless `validation_result.json` has `passed=true`.\n",
        encoding="utf-8",
    )
    return result
