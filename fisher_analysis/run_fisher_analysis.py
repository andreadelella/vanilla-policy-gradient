"""Estimate undamped empirical Fisher spectra for fixed random policies."""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import gymnasium as gym
import matplotlib
import numpy as np
import torch
from torch.distributions import Categorical, Normal
from torch.func import functional_call, grad as functional_grad, vmap

from policy import build_policy

matplotlib.use("Agg")
from matplotlib import pyplot as plt


DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parent / "results" / "hopper_width_sweep"
)
SCORE_BATCH_SIZE = 1024


@dataclass
class RolloutBatch:
    states: np.ndarray
    actions: np.ndarray
    episode_returns: np.ndarray
    episode_lengths: np.ndarray
    invalid_sample_count: int


def parameter_layout(policy: torch.nn.Module) -> list[dict[str, Any]]:
    """Return the named-parameter order used to flatten score gradients."""
    layout: list[dict[str, Any]] = []
    offset = 0
    for name, parameter in policy.named_parameters():
        end = offset + parameter.numel()
        layout.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "start": offset,
                "stop": end,
                "numel": parameter.numel(),
                "dtype": str(parameter.dtype),
            }
        )
        offset = end
    return layout


def _state_dict_snapshot(policy: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in policy.state_dict().items()
    }


def _assert_policy_unchanged(
    policy: torch.nn.Module,
    initial_state: dict[str, torch.Tensor],
) -> None:
    current_state = policy.state_dict()
    if current_state.keys() != initial_state.keys():
        raise AssertionError("Policy state layout changed during Fisher analysis")
    for name, initial_value in initial_state.items():
        if not torch.equal(current_state[name].detach().cpu(), initial_value):
            raise AssertionError(f"Policy tensor {name!r} changed during Fisher analysis")
    if any(parameter.grad is not None for parameter in policy.parameters()):
        raise AssertionError("Fisher analysis unexpectedly populated parameter gradients")


def _policy_seed(base_seed: int, width: int) -> int:
    return int((base_seed + 10_007 * width) % (2**63 - 1))


def _environment_seed(
    base_seed: int,
    iteration: int,
    trajectory_index: int,
    env_index: int,
    trajectories_per_env: int,
    n_envs: int,
) -> int:
    episode_index = (
        (iteration * trajectories_per_env + trajectory_index) * n_envs
        + env_index
    )
    return int((base_seed + 1_000_003 + episode_index) % (2**32))


def _action_seed(base_seed: int, width: int, iteration: int) -> int:
    return int(
        (base_seed + 2_000_003 + 100_003 * width + iteration) % (2**63 - 1)
    )


def _make_env_factory(env_id: str):
    def make_env():
        return gym.make(env_id)

    return make_env


def _multiprocessing_context() -> str | None:
    # Fork keeps a 32-worker MuJoCo vector environment memory-efficient on
    # Unix. Fall back to the platform default where fork is unavailable.
    return (
        "fork"
        if "fork" in multiprocessing.get_all_start_methods()
        else None
    )


def _sample_policy_actions(
    policy: torch.nn.Module,
    observations: Sequence[np.ndarray],
    generator: torch.Generator,
) -> np.ndarray:
    parameter = next(policy.parameters())
    state_tensor = torch.as_tensor(
        np.asarray(observations),
        dtype=parameter.dtype,
        device=parameter.device,
    )
    with torch.no_grad():
        output = policy(state_tensor)
        if isinstance(output, tuple):
            mean, std = output
            noise = torch.randn(
                mean.shape,
                dtype=mean.dtype,
                device=mean.device,
                generator=generator,
            )
            actions = mean + std * noise
        else:
            probabilities = torch.softmax(output, dim=-1)
            actions = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            ).squeeze(-1)
    return actions.detach().cpu().numpy()


def collect_fixed_policy_batch(
    policy: torch.nn.Module,
    envs: gym.vector.VectorEnv,
    *,
    iteration: int,
    trajectories_per_env: int,
    horizon: int,
    base_seed: int,
    width: int,
) -> RolloutBatch:
    """Collect fresh seeded trajectories from asynchronously stepped environments."""
    action_generator = torch.Generator(device="cpu")
    action_generator.manual_seed(_action_seed(base_seed, width, iteration))

    collected_states: list[np.ndarray] = []
    collected_actions: list[np.ndarray] = []
    episode_returns: list[float] = []
    episode_lengths: list[int] = []
    invalid_sample_count = 0
    n_envs = envs.num_envs
    action_space = envs.single_action_space
    continuous_actions = isinstance(action_space, gym.spaces.Box)

    for trajectory_index in range(trajectories_per_env):
        env_seeds = [
            _environment_seed(
                base_seed,
                iteration,
                trajectory_index,
                env_index,
                trajectories_per_env,
                n_envs,
            )
            for env_index in range(n_envs)
        ]
        observations, _ = envs.reset(seed=env_seeds)
        active = np.ones(n_envs, dtype=bool)
        running_returns = np.zeros(n_envs, dtype=np.float64)
        running_lengths = np.zeros(n_envs, dtype=np.int64)

        if continuous_actions:
            full_actions = np.zeros(
                (n_envs, *action_space.shape),
                dtype=action_space.dtype,
            )
        else:
            full_actions = np.zeros(n_envs, dtype=np.int64)

        while np.any(active):
            active_indices = np.flatnonzero(active)
            active_observations = observations[active_indices]
            raw_actions = _sample_policy_actions(
                policy,
                active_observations,
                action_generator,
            )

            finite_states = np.all(
                np.isfinite(active_observations),
                axis=tuple(range(1, active_observations.ndim)),
            )
            if raw_actions.ndim == 1:
                finite_actions = np.isfinite(raw_actions)
            else:
                finite_actions = np.all(
                    np.isfinite(raw_actions),
                    axis=tuple(range(1, raw_actions.ndim)),
                )
            valid_samples = finite_states & finite_actions
            if np.any(valid_samples):
                collected_states.append(
                    np.asarray(
                        active_observations[valid_samples],
                        dtype=np.float64,
                    ).copy()
                )
                collected_actions.append(
                    np.asarray(
                        raw_actions[valid_samples],
                        dtype=np.float64 if continuous_actions else np.int64,
                    ).copy()
                )
            invalid_sample_count += int(np.count_nonzero(~valid_samples))

            full_actions[...] = 0
            if continuous_actions:
                full_actions[active_indices] = np.clip(
                    raw_actions,
                    action_space.low,
                    action_space.high,
                ).astype(action_space.dtype, copy=False)
            else:
                full_actions[active_indices] = raw_actions.astype(
                    np.int64,
                    copy=False,
                )

            next_observations, rewards, terminated, truncated, _ = envs.step(
                full_actions
            )
            running_returns[active_indices] += rewards[active_indices]
            running_lengths[active_indices] += 1

            reached_horizon = np.zeros(n_envs, dtype=bool)
            if horizon > 0:
                reached_horizon = running_lengths >= horizon
            finished = active & (terminated | truncated | reached_horizon)
            if np.any(finished):
                episode_returns.extend(running_returns[finished].tolist())
                episode_lengths.extend(running_lengths[finished].tolist())
                active[finished] = False
            observations = next_observations

    if not collected_states:
        raise RuntimeError("Rollout collection produced no finite state/action samples")

    action_dtype = np.float64 if continuous_actions else np.int64
    return RolloutBatch(
        states=np.concatenate(collected_states, axis=0, dtype=np.float64),
        actions=np.concatenate(collected_actions, axis=0, dtype=action_dtype),
        episode_returns=np.asarray(episode_returns, dtype=np.float64),
        episode_lengths=np.asarray(episode_lengths, dtype=np.int64),
        invalid_sample_count=invalid_sample_count,
    )


def compute_empirical_fisher(
    policy: torch.nn.Module,
    states: np.ndarray | torch.Tensor,
    actions: np.ndarray | torch.Tensor,
    *,
    score_batch_size: int = SCORE_BATCH_SIZE,
) -> torch.Tensor:
    """Compute the undamped empirical Fisher in float64.

    The implementation accumulates score outer products in chunks, avoiding
    storage of the full M-by-P score matrix for larger experiments.
    """
    if score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive")

    parameter_items = list(policy.named_parameters())
    if not parameter_items:
        raise ValueError("policy has no trainable parameters")
    if any(parameter.dtype != torch.float64 for _, parameter in parameter_items):
        raise ValueError("policy parameters must be float64")
    if any(parameter.device.type != "cpu" for _, parameter in parameter_items):
        raise ValueError("Fisher analysis currently requires a CPU policy")

    state_tensor = torch.as_tensor(states, dtype=torch.float64, device="cpu")
    action_tensor = torch.as_tensor(actions, device="cpu")
    if state_tensor.ndim != 2:
        raise ValueError("states must have shape [samples, state_dim]")
    if action_tensor.shape[0] != state_tensor.shape[0]:
        raise ValueError("states and actions must have the same sample count")
    if state_tensor.shape[0] == 0:
        raise ValueError("at least one state/action sample is required")

    params = dict(parameter_items)
    buffers = dict(policy.named_buffers())
    merged_state = {**params, **buffers}
    parameter_count = sum(parameter.numel() for parameter in params.values())
    fisher_sum = torch.zeros(
        (parameter_count, parameter_count),
        dtype=torch.float64,
        device="cpu",
    )

    def single_log_prob(
        functional_params: dict[str, torch.Tensor],
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        state_and_buffers = {**functional_params, **buffers}
        output = functional_call(
            policy,
            state_and_buffers,
            (state.unsqueeze(0),),
        )
        if isinstance(output, tuple):
            mean, std = output
            return Normal(mean.squeeze(0), std).log_prob(action).sum()
        return Categorical(logits=output.squeeze(0)).log_prob(action.long())

    # Materialize the merged mapping once to catch duplicate parameter/buffer
    # names before entering torch.func's transformed function.
    if len(merged_state) != len(params) + len(buffers):
        raise ValueError("policy parameter and buffer names overlap")

    sample_count = state_tensor.shape[0]
    score_function = vmap(
        functional_grad(single_log_prob, argnums=0),
        in_dims=(None, 0, 0),
    )
    for start in range(0, sample_count, score_batch_size):
        stop = min(start + score_batch_size, sample_count)
        batch_states = state_tensor[start:stop]
        batch_actions = action_tensor[start:stop]
        if batch_actions.ndim == 1:
            batch_actions = batch_actions.long()
        else:
            batch_actions = batch_actions.to(torch.float64)

        per_parameter_scores = score_function(
            params,
            batch_states,
            batch_actions,
        )
        batch_size = stop - start
        score_matrix = torch.cat(
            [
                per_parameter_scores[name].reshape(batch_size, -1)
                for name, _ in parameter_items
            ],
            dim=1,
        ).detach().to(torch.float64)
        fisher_sum.addmm_(score_matrix.T, score_matrix)

    return fisher_sum / sample_count


def analyze_fisher(
    fisher: np.ndarray | torch.Tensor,
    *,
    sample_count: int,
) -> tuple[np.ndarray, dict[str, float | int], float, float]:
    """Validate a Fisher matrix and derive its descending eigenspectrum metrics."""
    matrix = (
        fisher.detach().cpu().numpy()
        if isinstance(fisher, torch.Tensor)
        else np.asarray(fisher)
    )
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("fisher must be a square matrix")
    if not np.all(np.isfinite(matrix)):
        raise AssertionError("Fisher matrix contains non-finite values")

    dimension = matrix.shape[0]
    matrix_scale = max(1.0, float(np.max(np.abs(matrix), initial=0.0)))
    symmetry_tolerance = 64.0 * dimension * np.finfo(np.float64).eps * matrix_scale
    symmetry_error = float(np.max(np.abs(matrix - matrix.T), initial=0.0))
    if symmetry_error > symmetry_tolerance:
        raise AssertionError(
            "Fisher matrix is not symmetric: "
            f"max error {symmetry_error:.3e} > {symmetry_tolerance:.3e}"
        )

    eigenvalues = np.linalg.eigvalsh(matrix)[::-1].copy()
    spectral_scale = float(np.max(np.abs(eigenvalues), initial=0.0))
    rank_tolerance = max(
        dimension * np.finfo(np.float64).eps * spectral_scale,
        np.finfo(np.float64).tiny,
    )
    psd_tolerance = max(
        100.0 * rank_tolerance,
        1e-12 * max(1.0, spectral_scale),
    )
    minimum_eigenvalue = float(eigenvalues[-1]) if dimension else 0.0
    if minimum_eigenvalue < -psd_tolerance:
        raise AssertionError(
            "Fisher matrix is not positive semidefinite: "
            f"minimum eigenvalue {minimum_eigenvalue:.3e} "
            f"< {-psd_tolerance:.3e}"
        )

    trace = float(np.trace(matrix))
    eigenvalue_sum = float(np.sum(eigenvalues))
    if not np.isclose(
        trace,
        eigenvalue_sum,
        rtol=1e-10,
        atol=psd_tolerance * max(1, dimension),
    ):
        raise AssertionError(
            f"Fisher trace {trace:.16e} != eigenvalue sum {eigenvalue_sum:.16e}"
        )

    positive = eigenvalues[eigenvalues > rank_tolerance]
    numerical_rank = int(positive.size)
    positive_trace = float(np.sum(positive))
    if numerical_rank:
        probabilities = positive / positive_trace
        effective_rank = float(
            np.exp(-np.sum(probabilities * np.log(probabilities)))
        )
        stable_rank = float(np.sum(positive**2) / positive[0] ** 2)
        condition_number = float(positive[0] / positive[-1])
        cumulative = np.cumsum(positive) / positive_trace

        def components_for(fraction: float) -> int:
            return int(np.searchsorted(cumulative, fraction, side="left") + 1)

    else:
        effective_rank = 0.0
        stable_rank = 0.0
        condition_number = math.nan

        def components_for(fraction: float) -> int:
            del fraction
            return 0

    metrics: dict[str, float | int] = {
        "matrix_dimension": dimension,
        "sample_count": int(sample_count),
        "trace": trace,
        "numerical_rank": numerical_rank,
        "effective_rank": effective_rank,
        "stable_rank": stable_rank,
        "positive_condition_number": condition_number,
        "components_90": components_for(0.90),
        "components_95": components_for(0.95),
        "components_99": components_for(0.99),
        "minimum_eigenvalue": minimum_eigenvalue,
        "symmetry_error": symmetry_error,
    }
    return eigenvalues, metrics, rank_tolerance, psd_tolerance


def _plot_spectra(
    spectra: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    colors = ["#176B87", "#C2410C", "#3F6212", "#7E22CE", "#374151"]

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for index, spectrum in enumerate(spectra):
        eigenvalues = spectrum["eigenvalues"]
        floor = max(spectrum["rank_tolerance"], np.finfo(np.float64).tiny)
        plotted = np.maximum(eigenvalues, floor)
        axis.semilogy(
            np.arange(1, eigenvalues.size + 1),
            plotted,
            color=colors[index % len(colors)],
            label=f"width {spectrum['width']} (P={eigenvalues.size})",
        )
    axis.set_xlabel("Principal-component index")
    axis.set_ylabel("Eigenvalue")
    axis.set_title("Undamped empirical Fisher eigenspectrum")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "raw_eigenspectrum.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for index, spectrum in enumerate(spectra):
        eigenvalues = spectrum["eigenvalues"]
        trace = spectrum["trace"]
        normalized = eigenvalues / trace if trace > 0 else np.zeros_like(eigenvalues)
        normalized_floor = (
            spectrum["rank_tolerance"] / trace
            if trace > 0
            else np.finfo(np.float64).tiny
        )
        axis.semilogy(
            np.arange(1, eigenvalues.size + 1),
            np.maximum(normalized, normalized_floor),
            color=colors[index % len(colors)],
            label=f"width {spectrum['width']}",
        )
    axis.set_xlabel("Principal-component index")
    axis.set_ylabel("Eigenvalue / Fisher trace")
    axis.set_title("Trace-normalized Fisher eigenspectrum")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "trace_normalized_eigenspectrum.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for index, spectrum in enumerate(spectra):
        eigenvalues = spectrum["eigenvalues"].copy()
        tolerance_level = np.abs(eigenvalues) <= spectrum["psd_tolerance"]
        eigenvalues[tolerance_level] = np.maximum(eigenvalues[tolerance_level], 0.0)
        total = float(np.sum(eigenvalues))
        cumulative = (
            np.cumsum(eigenvalues) / total
            if total > 0
            else np.zeros_like(eigenvalues)
        )
        axis.plot(
            np.arange(1, eigenvalues.size + 1),
            cumulative,
            color=colors[index % len(colors)],
            label=f"width {spectrum['width']}",
        )
    for threshold in (0.90, 0.95, 0.99):
        axis.axhline(threshold, color="#6B7280", linewidth=0.8, linestyle="--")
    axis.set_ylim(0.0, 1.01)
    axis.set_xlabel("Number of principal components")
    axis.set_ylabel("Cumulative explained Fisher trace")
    axis.set_title("Cumulative Fisher trace")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "cumulative_explained_trace.png", dpi=180)
    plt.close(figure)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write empty CSV file: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _validate_args(args: argparse.Namespace) -> None:
    if not args.widths or any(width <= 0 for width in args.widths):
        raise ValueError("--widths must contain positive integers")
    if len(set(args.widths)) != len(args.widths):
        raise ValueError("--widths must not contain duplicates")
    if args.depth <= 0:
        raise ValueError("--depth must be positive")
    for name in ("iterations", "n_envs", "trajectories_per_env"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.horizon < 0:
        raise ValueError("--horizon must be non-negative")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze undamped empirical Fisher spectra of fixed policies.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-id", default="Hopper-v5")
    parser.add_argument("--widths", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--trajectories-per-env", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser


def run_analysis(args: argparse.Namespace) -> list[dict[str, Any]]:
    _validate_args(args)
    output_dir = args.output_dir.expanduser().resolve()
    checkpoint_dir = output_dir / "checkpoints"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    vector_context = _multiprocessing_context()
    config: dict[str, Any] = {
        "schema_version": 1,
        "experiment": "fixed_policy_undamped_empirical_fisher",
        "env_id": args.env_id,
        "widths": list(args.widths),
        "depth": args.depth,
        "iterations": args.iterations,
        "n_envs": args.n_envs,
        "trajectories_per_env": args.trajectories_per_env,
        "horizon": args.horizon,
        "seed": args.seed,
        "output_dir": str(output_dir),
        "dtype": "float64",
        "damping": 0.0,
        "init_log_std": -0.5,
        "learn_std": True,
        "score_batch_size": SCORE_BATCH_SIZE,
        "vectorization": "gymnasium.vector.AsyncVectorEnv",
        "multiprocessing_context": vector_context or "platform_default",
        "rank_definition": "eigenvalues > P * eps(float64) * max(abs(eigenvalues))",
        "effective_rank_definition": "exp(entropy(positive_eigenvalues / positive_trace))",
        "stable_rank_definition": "sum(positive_eigenvalues^2) / largest_eigenvalue^2",
        "seed_schedule": {
            "policy": "(seed + 10007 * width) mod (2^63 - 1)",
            "environment": (
                "(seed + 1000003 + flattened episode index) mod 2^32"
            ),
            "action": (
                "(seed + 2000003 + 100003 * width + iteration) "
                "mod (2^63 - 1)"
            ),
        },
        "policy_seeds": {
            str(width): _policy_seed(args.seed, width) for width in args.widths
        },
    }
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)

    summary_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    parameter_dimensions: dict[str, int] = {}

    vector_env_kwargs: dict[str, Any] = {
        "autoreset_mode": gym.vector.AutoresetMode.NEXT_STEP,
    }
    if vector_context is not None:
        vector_env_kwargs["context"] = vector_context
    parallel_envs = gym.vector.AsyncVectorEnv(
        [_make_env_factory(args.env_id) for _ in range(args.n_envs)],
        **vector_env_kwargs,
    )
    probe_env = gym.make(args.env_id)
    try:
        for width in args.widths:
            policy_seed = _policy_seed(args.seed, width)
            torch.manual_seed(policy_seed)
            policy_config = {
                "hidden_sizes": [width] * args.depth,
                "policy": "mlp",
                "init_log_std": -0.5,
                "learn_std": True,
            }
            policy = build_policy(policy_config, probe_env).to(
                device="cpu",
                dtype=torch.float64,
            )
            policy.eval()
            initial_state = _state_dict_snapshot(policy)
            layout = parameter_layout(policy)
            dimension = sum(item["numel"] for item in layout)
            parameter_dimensions[str(width)] = dimension
            torch.save(initial_state, checkpoint_dir / f"policy_width_{width}.pt")

            width_states: list[np.ndarray] = []
            width_actions: list[np.ndarray] = []
            sample_counts: list[int] = []
            print(
                f"Width {width}: P={dimension}, collecting "
                f"{args.iterations} seeded rollout batches"
            )
            for iteration in range(args.iterations):
                start = time.perf_counter()
                batch = collect_fixed_policy_batch(
                    policy,
                    parallel_envs,
                    iteration=iteration,
                    trajectories_per_env=args.trajectories_per_env,
                    horizon=args.horizon,
                    base_seed=args.seed,
                    width=width,
                )
                elapsed = time.perf_counter() - start
                _assert_policy_unchanged(policy, initial_state)
                width_states.append(batch.states)
                width_actions.append(batch.actions)
                sample_counts.append(batch.states.shape[0])
                iteration_rows.append(
                    {
                        "width": width,
                        "iteration": iteration,
                        "trajectory_count": int(batch.episode_returns.size),
                        "sample_count": int(batch.states.shape[0]),
                        "invalid_sample_count": batch.invalid_sample_count,
                        "mean_return": float(np.mean(batch.episode_returns)),
                        "std_return": float(np.std(batch.episode_returns)),
                        "min_return": float(np.min(batch.episode_returns)),
                        "max_return": float(np.max(batch.episode_returns)),
                        "mean_length": float(np.mean(batch.episode_lengths)),
                        "min_length": int(np.min(batch.episode_lengths)),
                        "max_length": int(np.max(batch.episode_lengths)),
                        "rollout_seconds": elapsed,
                    }
                )
                print(
                    f"  iteration {iteration + 1:02d}/{args.iterations}: "
                    f"{batch.states.shape[0]} samples, "
                    f"mean return {np.mean(batch.episode_returns):.2f}"
                )

            states = np.concatenate(width_states, axis=0)
            actions = np.concatenate(width_actions, axis=0)
            print(f"  accumulating undamped Fisher from {states.shape[0]} scores")
            fisher_tensor = compute_empirical_fisher(policy, states, actions)
            _assert_policy_unchanged(policy, initial_state)
            eigenvalues, metrics, rank_tolerance, psd_tolerance = analyze_fisher(
                fisher_tensor,
                sample_count=states.shape[0],
            )
            fisher = fisher_tensor.detach().cpu().numpy()
            trace = float(metrics["trace"])
            normalized = (
                eigenvalues / trace if trace > 0 else np.zeros_like(eigenvalues)
            )
            cleaned_for_cumulative = eigenvalues.copy()
            tolerance_level = np.abs(cleaned_for_cumulative) <= psd_tolerance
            cleaned_for_cumulative[tolerance_level] = np.maximum(
                cleaned_for_cumulative[tolerance_level],
                0.0,
            )
            cumulative = np.cumsum(cleaned_for_cumulative)
            if cumulative.size and cumulative[-1] > 0:
                cumulative /= cumulative[-1]

            names = np.asarray([item["name"] for item in layout])
            offsets = np.asarray(
                [[item["start"], item["stop"]] for item in layout],
                dtype=np.int64,
            )
            shapes_json = np.asarray(
                [json.dumps(item["shape"]) for item in layout]
            )
            np.savez_compressed(
                output_dir / f"fisher_width_{width}.npz",
                fisher=fisher,
                eigenvalues=eigenvalues,
                trace_normalized_eigenvalues=normalized,
                cumulative_explained_trace=cumulative,
                parameter_names=names,
                parameter_offsets=offsets,
                parameter_shapes_json=shapes_json,
                parameter_layout_json=np.asarray(json.dumps(layout)),
                sample_counts=np.asarray(sample_counts, dtype=np.int64),
                total_sample_count=np.asarray(states.shape[0], dtype=np.int64),
                rank_tolerance=np.asarray(rank_tolerance, dtype=np.float64),
                psd_tolerance=np.asarray(psd_tolerance, dtype=np.float64),
            )

            summary_row = {
                "width": width,
                "depth": args.depth,
                **metrics,
                "rank_tolerance": rank_tolerance,
                "psd_tolerance": psd_tolerance,
            }
            summary_rows.append(summary_row)
            spectra.append(
                {
                    "width": width,
                    "eigenvalues": eigenvalues,
                    "trace": trace,
                    "rank_tolerance": rank_tolerance,
                    "psd_tolerance": psd_tolerance,
                }
            )
            print(
                f"  trace={trace:.6g}, rank={metrics['numerical_rank']}/{dimension}, "
                f"condition={metrics['positive_condition_number']:.3e}"
            )
    finally:
        probe_env.close()
        parallel_envs.close()

    config["parameter_dimensions"] = parameter_dimensions
    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
    _write_csv(output_dir / "iteration_stats.csv", iteration_rows)
    _write_csv(output_dir / "summary.csv", summary_rows)
    _plot_spectra(spectra, output_dir)
    print(f"Saved Fisher analysis to {output_dir}")
    return summary_rows


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run_analysis(args)
    except (ValueError, AssertionError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
