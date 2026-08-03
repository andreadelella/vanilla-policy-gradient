"""Training primitives for neural categorical-policy barrier experiments."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch

from vpg.data_collection import Trajectory
from vpg.data_collection import collect_parallel_trajectories
from vpg.gpomdp import (
    apply_npg_preconditioning,
    compute_gpomdp_loss,
    trajectories_to_tensors,
)
from vpg.policy import MLPSoftmaxPolicy

from .barrier import categorical_log_barrier


MethodName = Literal[
    "gpomdp_reward_only",
    "gpomdp_entropy_fixed",
    "gpomdp_logbarrier_fixed",
    "gpomdp_logbarrier_handoff",
    "npg_reward_only",
]


@dataclass(frozen=True)
class NeuralTrainingConfig:
    environment: str
    method: MethodName
    seed: int
    hidden_sizes: tuple[int, ...] = (8, 8)
    learning_rate: float = 1e-3
    gamma: float = 0.99
    updates: int = 60
    batch_steps: int = 512
    horizon: int = 500
    center_returns: bool = False
    normalize_returns: bool = False
    beta: float = 0.0
    entropy_coefficient: float = 0.0
    handoff_fraction: float | None = None
    npg_damping: float = 1e-2
    evaluation_episodes: int = 5
    checkpoint_fractions: tuple[float, ...] = tuple(i / 10 for i in range(11))
    dtype: str = "float32"
    collector_mode: str = "fixed_steps"
    parallel_environments: int = 8
    episodes_per_update: int | None = None

    def validate(self) -> None:
        if self.updates < 1 or self.batch_steps < 1 or self.horizon < 1:
            raise ValueError("updates, batch_steps, and horizon must be positive")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1]")
        if self.method == "gpomdp_logbarrier_handoff":
            if self.handoff_fraction is None or not 0.0 < self.handoff_fraction < 1.0:
                raise ValueError("handoff method requires a fraction in (0, 1)")
        if self.method != "gpomdp_logbarrier_handoff" and self.handoff_fraction is not None:
            raise ValueError("handoff_fraction is only valid for the handoff method")
        if self.method == "gpomdp_entropy_fixed" and self.entropy_coefficient <= 0.0:
            raise ValueError("entropy method requires a positive coefficient")
        if "logbarrier" in self.method and self.beta <= 0.0:
            raise ValueError("log-barrier methods require beta > 0")
        if self.collector_mode not in (
            "fixed_steps",
            "complete_episodes",
            "complete_episodes_by_update",
        ):
            raise ValueError(
                "collector_mode must be fixed_steps, complete_episodes, "
                "or complete_episodes_by_update"
            )
        if self.parallel_environments < 1:
            raise ValueError("parallel_environments must be positive")
        if self.collector_mode == "complete_episodes_by_update":
            if self.episodes_per_update is None or self.episodes_per_update < 1:
                raise ValueError(
                    "complete_episodes_by_update requires episodes_per_update > 0"
                )
            if self.episodes_per_update % self.parallel_environments != 0:
                raise ValueError(
                    "episodes_per_update must be divisible by parallel_environments"
                )
        elif self.episodes_per_update is not None:
            raise ValueError(
                "episodes_per_update is only valid for complete_episodes_by_update"
            )

    @property
    def total_environment_steps(self) -> int:
        return self.updates * self.batch_steps

    @property
    def total_training_episodes(self) -> int | None:
        if self.collector_mode != "complete_episodes_by_update":
            return None
        return self.updates * int(self.episodes_per_update)

    @property
    def handoff_update(self) -> int | None:
        if self.handoff_fraction is None:
            return None
        return int(round(self.updates * self.handoff_fraction))

    def beta_at_update(self, update: int) -> float:
        if self.method == "gpomdp_logbarrier_fixed":
            return self.beta
        if self.method == "gpomdp_logbarrier_handoff":
            return self.beta if update < int(self.handoff_update) else 0.0
        return 0.0

    def beta_at_environment_step(self, environment_step: int) -> float:
        if self.method == "gpomdp_logbarrier_fixed":
            return self.beta
        if self.method == "gpomdp_logbarrier_handoff":
            boundary = int(round(self.total_environment_steps * float(self.handoff_fraction)))
            return self.beta if environment_step < boundary else 0.0
        return 0.0

    def to_dict(self) -> dict:
        result = asdict(self)
        result["total_environment_steps"] = (
            None
            if self.collector_mode == "complete_episodes_by_update"
            else self.total_environment_steps
        )
        result["total_training_episodes"] = self.total_training_episodes
        result["handoff_update"] = self.handoff_update
        return result


def initial_weight_identifier(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        digest.update(name.encode("utf-8"))
        digest.update(state_dict[name].detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def build_seeded_policy(config: NeuralTrainingConfig) -> tuple[MLPSoftmaxPolicy, str]:
    probe = gym.make(config.environment)
    try:
        state_dim = int(probe.observation_space.shape[0])
        action_dim = int(probe.action_space.n)
    finally:
        probe.close()
    torch.manual_seed(config.seed)
    policy = MLPSoftmaxPolicy(state_dim, action_dim, config.hidden_sizes)
    return policy, initial_weight_identifier(policy.state_dict())


def _sample_categorical(probabilities: torch.Tensor, generator: torch.Generator) -> int:
    u = torch.rand((), generator=generator, dtype=torch.float64).item()
    cumulative = torch.cumsum(probabilities.detach().cpu().to(torch.float64), dim=-1)
    return int(torch.searchsorted(cumulative, torch.tensor(u, dtype=torch.float64), right=False).clamp_max(probabilities.numel() - 1))


def collect_fixed_step_trajectories(
    environment: str,
    policy: torch.nn.Module,
    *,
    step_count: int,
    horizon: int,
    reset_seed_base: int,
    action_generator: torch.Generator,
    maximum_parallel_environments: int = 16,
) -> list[Trajectory]:
    """Collect exactly ``step_count`` interactions with batched policy inference.

    Independent ordinary Gym environments are stepped in lockstep.  This keeps
    environment semantics transparent while avoiding one neural forward pass
    per interaction.  The number of environments is chosen as the largest
    divisor of ``step_count`` no greater than 16, so no padding or discarded
    transition is needed.
    """

    environment_count = max(
        candidate
        for candidate in range(1, min(maximum_parallel_environments, step_count) + 1)
        if step_count % candidate == 0
    )
    environments = [gym.make(environment) for _ in range(environment_count)]
    trajectories: list[Trajectory] = []
    episode_indices = [0 for _ in range(environment_count)]
    states = []
    current = []
    for index, env in enumerate(environments):
        state, _ = env.reset(seed=reset_seed_base + index)
        states.append(state)
        current.append(Trajectory(states=[], actions=[], rewards=[], dones=[]))
    rounds = step_count // environment_count
    try:
        for round_index in range(rounds):
            state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32)
            with torch.no_grad():
                batch_probabilities = policy.distribution(state_tensor).probs
            actions = [
                _sample_categorical(batch_probabilities[index], action_generator)
                for index in range(environment_count)
            ]
            final_round = round_index + 1 == rounds
            for index, (env, action) in enumerate(zip(environments, actions)):
                next_state, reward, terminated, truncated, _ = env.step(action)
                trajectory = current[index]
                trajectory.states.append(np.asarray(states[index], dtype=np.float32).copy())
                trajectory.actions.append(np.asarray(action, dtype=np.int64))
                trajectory.executed_actions.append(np.asarray(action, dtype=np.int64))
                trajectory.rewards.append(float(reward))
                reached_horizon = len(trajectory.rewards) >= horizon
                done = bool(terminated or truncated or reached_horizon or final_round)
                trajectory.dones.append(done)
                if done:
                    trajectories.append(trajectory)
                    if not final_round:
                        episode_indices[index] += 1
                        next_state, _ = env.reset(
                            seed=reset_seed_base
                            + index
                            + episode_indices[index] * environment_count
                        )
                        current[index] = Trajectory(states=[], actions=[], rewards=[], dones=[])
                states[index] = next_state
    finally:
        for env in environments:
            env.close()
    if sum(len(item.rewards) for item in trajectories) != step_count:
        raise AssertionError("fixed-step collector returned the wrong interaction count")
    return trajectories


def _flatten_valid_states(trajectories: list[Trajectory]) -> torch.Tensor:
    states, _, _, mask = trajectories_to_tensors(trajectories, device="cpu")
    flat = states.reshape(-1, states.shape[-1])
    return flat[mask.reshape(-1).bool()].detach()


def _flat_gradient(objective: torch.Tensor, policy: torch.nn.Module, *, retain_graph: bool) -> torch.Tensor:
    parameters = tuple(policy.parameters())
    if not objective.requires_grad:
        return torch.cat([torch.zeros_like(parameter).reshape(-1) for parameter in parameters])
    gradients = torch.autograd.grad(
        objective,
        parameters,
        retain_graph=retain_graph,
        allow_unused=True,
    )
    return torch.cat([
        torch.zeros_like(parameter).reshape(-1) if gradient is None else gradient.reshape(-1)
        for parameter, gradient in zip(parameters, gradients)
    ])


def _cosine(left: torch.Tensor, right: torch.Tensor) -> tuple[float, bool]:
    left_norm = left.norm()
    right_norm = right.norm()
    if float(left_norm) == 0.0 or float(right_norm) == 0.0:
        return float("nan"), False
    return float(torch.dot(left, right) / (left_norm * right_norm)), True


def evaluate_policy(
    environment: str,
    policy: torch.nn.Module,
    *,
    episodes: int,
    horizon: int,
    seed_base: int,
    deterministic: bool,
) -> tuple[float, float, list[np.ndarray]]:
    metrics, states = evaluate_policy_detailed(
        environment,
        policy,
        episodes=episodes,
        horizon=horizon,
        seed_base=seed_base,
        deterministic=deterministic,
    )
    return metrics["mean_return"], metrics["mean_episode_length"], states


def evaluate_policy_detailed(
    environment: str,
    policy: torch.nn.Module,
    *,
    episodes: int,
    horizon: int,
    seed_base: int,
    deterministic: bool,
) -> tuple[dict[str, float | int], list[np.ndarray]]:
    """Evaluate complete episodes and retain goal termination separately from timeout."""

    env = gym.make(environment)
    returns: list[float] = []
    lengths: list[int] = []
    states: list[np.ndarray] = []
    terminations = 0
    truncations = 0
    generator = torch.Generator(device="cpu").manual_seed(seed_base + 900_000)
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=seed_base + episode)
            total = 0.0
            for step in range(horizon):
                state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
                with torch.no_grad():
                    probabilities = policy.distribution(state_tensor).probs.squeeze(0)
                action = int(probabilities.argmax()) if deterministic else _sample_categorical(probabilities, generator)
                states.append(np.asarray(state, dtype=np.float32).copy())
                state, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                if terminated or truncated:
                    terminations += int(terminated)
                    truncations += int(truncated)
                    lengths.append(step + 1)
                    break
            else:
                lengths.append(horizon)
            returns.append(total)
    finally:
        env.close()
    metrics: dict[str, float | int] = {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "return_standard_deviation": float(np.std(returns, ddof=1)) if episodes > 1 else 0.0,
        "mean_episode_length": float(np.mean(lengths)),
        "termination_rate": float(terminations / episodes),
        "truncation_rate": float(truncations / episodes),
    }
    return metrics, states


def checkpoint_updates(config: NeuralTrainingConfig) -> list[int]:
    checkpoints = {int(round(config.updates * value)) for value in config.checkpoint_fractions}
    if config.handoff_update is not None:
        checkpoints.update({
            max(0, config.handoff_update - 1),
            config.handoff_update,
            min(config.updates, config.handoff_update + 1),
            min(config.updates, config.handoff_update + max(1, int(round(0.05 * config.updates)))),
        })
    return sorted(checkpoints)


def checkpoint_environment_steps(config: NeuralTrainingConfig) -> list[int]:
    checkpoints = {
        int(round(config.total_environment_steps * value))
        for value in config.checkpoint_fractions
    }
    if config.handoff_fraction is not None:
        handoff = int(round(config.total_environment_steps * config.handoff_fraction))
        checkpoints.update({
            handoff,
            min(config.total_environment_steps, handoff + max(1, int(round(0.05 * config.total_environment_steps)))),
        })
    return sorted(checkpoints)


def _checkpoint_row(
    config: NeuralTrainingConfig,
    policy: torch.nn.Module,
    update: int,
    *,
    beta: float,
    environment_steps: int | None = None,
) -> dict[str, float | int | bool | str]:
    if environment_steps is None:
        environment_steps = update * config.batch_steps
    deterministic_metrics, _ = evaluate_policy_detailed(
        config.environment,
        policy,
        episodes=config.evaluation_episodes,
        horizon=config.horizon,
        seed_base=7_000_000 + config.seed * 10_000 + update * 31,
        deterministic=True,
    )
    stochastic_metrics, evaluation_states = evaluate_policy_detailed(
        config.environment,
        policy,
        episodes=config.evaluation_episodes,
        horizon=config.horizon,
        seed_base=8_000_000 + config.seed * 10_000 + update * 31,
        deterministic=False,
    )
    states = torch.as_tensor(np.asarray(evaluation_states), dtype=torch.float32)
    with torch.no_grad():
        logits = policy(states)
        _, diagnostics = categorical_log_barrier(logits)
    return {
        "environment": config.environment,
        "method": config.method,
        "seed": config.seed,
        "update": update,
        "environment_steps": environment_steps,
        "deterministic_return": deterministic_metrics["mean_return"],
        "stochastic_return": stochastic_metrics["mean_return"],
        "deterministic_return_standard_deviation": deterministic_metrics["return_standard_deviation"],
        "stochastic_return_standard_deviation": stochastic_metrics["return_standard_deviation"],
        "episode_length": deterministic_metrics["mean_episode_length"],
        "deterministic_termination_rate": deterministic_metrics["termination_rate"],
        "stochastic_termination_rate": stochastic_metrics["termination_rate"],
        "entropy": diagnostics.mean_entropy,
        "mean_min_probability": diagnostics.mean_min_probability,
        "global_min_probability": diagnostics.global_min_probability,
        "barrier_value": diagnostics.barrier_value,
        "beta": beta,
        "barrier_active": bool(beta > 0.0),
    }


def train_policy(config: NeuralTrainingConfig, output_directory: str | Path) -> dict:
    """Train one seed/method and write a self-contained, non-overwriting archive."""

    config.validate()
    output_directory = Path(output_directory)
    config_path = output_directory / "config.json"
    # Canonicalize tuples to JSON arrays before compatibility comparison so a
    # completed archive can be resumed without treating serialization itself
    # as a scientific configuration change.
    requested = json.loads(json.dumps(config.to_dict()))
    if output_directory.exists() and any(output_directory.iterdir()):
        if not config_path.exists() or json.loads(config_path.read_text(encoding="utf-8")) != requested:
            raise FileExistsError(f"incompatible nonempty output directory: {output_directory}")
        summary_path = output_directory / "summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
        raise RuntimeError(f"incomplete existing run requires explicit cleanup: {output_directory}")

    output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = output_directory / "checkpoints"
    checkpoint_directory.mkdir()
    config_path.write_text(json.dumps(requested, indent=2, sort_keys=True), encoding="utf-8")

    policy, weight_id = build_seeded_policy(config)
    if config.method == "npg_reward_only":
        optimizer = torch.optim.SGD(policy.parameters(), lr=config.learning_rate)
        optimizer_name = "SGD_after_sampled_action_NPG"
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
        optimizer_name = "Adam"

    action_generator = torch.Generator(device="cpu").manual_seed(config.seed + 50_000)
    checkpoint_set = set(checkpoint_updates(config))
    checkpoint_step_targets = set(checkpoint_environment_steps(config))
    behavior_rows: list[dict] = []
    gradient_rows: list[dict] = []
    training_rows: list[dict] = []
    finite = True

    environment_steps = 0
    training_episodes = 0
    update = 0
    vector_environments = None
    if config.collector_mode in ("complete_episodes", "complete_episodes_by_update"):
        def make_factory(index: int):
            def factory():
                env = gym.make(config.environment)
                env.reset(seed=config.seed * 100_000 + index)
                env.action_space.seed(config.seed * 100_000 + index + 50_000)
                return env
            return factory
        vector_environments = gym.vector.SyncVectorEnv([
            make_factory(index) for index in range(config.parallel_environments)
        ])

    if 0 in (checkpoint_step_targets if config.collector_mode == "complete_episodes" else checkpoint_set):
        filename = "checkpoint_step_000000000_target_000000000.pt" if config.collector_mode == "complete_episodes" else "checkpoint_update_000000.pt"
        torch.save(policy.state_dict(), checkpoint_directory / filename)
        initial_beta = config.beta_at_environment_step(0) if config.collector_mode == "complete_episodes" else config.beta_at_update(0)
        behavior_rows.append(_checkpoint_row(config, policy, 0, beta=initial_beta, environment_steps=0))

    try:
      while (
          update < config.updates
          if config.collector_mode == "complete_episodes_by_update"
          else environment_steps < config.total_environment_steps
      ):
        if config.collector_mode == "complete_episodes":
            remaining = config.total_environment_steps - environment_steps
            maximum_full_batch = config.parallel_environments * config.horizon
            if remaining >= maximum_full_batch:
                trajectories = collect_parallel_trajectories(
                    vector_environments,
                    policy,
                    n_trajectories_per_env=1,
                    clip_actions=False,
                    device="cpu",
                )
            else:
                trajectories = collect_fixed_step_trajectories(
                    config.environment,
                    policy,
                    step_count=remaining,
                    horizon=config.horizon,
                    reset_seed_base=config.seed * 10_000_000 + update * 10_000,
                    action_generator=action_generator,
                )
        elif config.collector_mode == "complete_episodes_by_update":
            trajectories = collect_parallel_trajectories(
                vector_environments,
                policy,
                n_trajectories_per_env=(
                    int(config.episodes_per_update) // config.parallel_environments
                ),
                clip_actions=False,
                device="cpu",
            )
        else:
            trajectories = collect_fixed_step_trajectories(
                config.environment,
                policy,
                step_count=config.batch_steps,
                horizon=config.horizon,
                reset_seed_base=config.seed * 10_000_000 + update * 10_000,
                action_generator=action_generator,
            )
        batch_interactions = sum(len(item.rewards) for item in trajectories)
        states = _flatten_valid_states(trajectories)
        reward_loss = compute_gpomdp_loss(
            policy,
            trajectories,
            gamma=config.gamma,
            center_returns=config.center_returns,
            normalize_returns=config.normalize_returns,
            entropy_coeff=0.0,
            device="cpu",
        )
        reward_objective = -reward_loss
        logits = policy(states.detach())
        barrier, barrier_diagnostics = categorical_log_barrier(logits)
        entropy = policy.distribution(states.detach()).entropy().mean()
        beta = (
            config.beta_at_environment_step(environment_steps)
            if config.collector_mode == "complete_episodes"
            else config.beta_at_update(update)
        )
        regularizer = torch.zeros((), dtype=reward_objective.dtype)
        if "logbarrier" in config.method:
            regularizer = beta * barrier
        elif config.method == "gpomdp_entropy_fixed":
            regularizer = config.entropy_coefficient * entropy

        reward_gradient = _flat_gradient(reward_objective, policy, retain_graph=True)
        barrier_gradient = _flat_gradient(barrier, policy, retain_graph=True)
        regularizer_gradient = _flat_gradient(regularizer, policy, retain_graph=True)
        cosine, cosine_defined = _cosine(reward_gradient, regularizer_gradient)
        total_objective = reward_objective + regularizer

        optimizer.zero_grad()
        (-total_objective).backward()
        if config.method == "npg_reward_only":
            apply_npg_preconditioning(policy, trajectories, config.npg_damping, device="cpu")
        optimizer.step()

        parameter_vector = torch.cat([parameter.detach().reshape(-1) for parameter in policy.parameters()])
        if not torch.isfinite(parameter_vector).all():
            finite = False
            break
        train_returns = np.asarray([sum(item.rewards) for item in trajectories], dtype=np.float64)
        environment_steps += batch_interactions
        training_episodes += len(trajectories)
        completed = update + 1
        training_rows.append({
            "update": update + 1,
            "environment_steps": environment_steps,
            "batch_training_episodes": len(trajectories),
            "cumulative_training_episodes": training_episodes,
            "training_return": float(train_returns.mean()),
            "training_episode_length": float(np.mean([len(item.rewards) for item in trajectories])),
            "beta": beta,
            "barrier_active": bool(beta > 0.0),
        })
        gradient_rows.append({
            "update": update,
            "environment_steps": environment_steps - batch_interactions,
            "reward_gradient_norm": float(reward_gradient.norm()),
            "barrier_gradient_norm": float(barrier_gradient.norm()),
            "regularizer_gradient_norm": float(regularizer_gradient.norm()),
            "reward_regularizer_cosine": cosine,
            "cosine_defined": cosine_defined,
            "beta": beta,
        })

        if config.collector_mode == "complete_episodes":
            crossed = sorted(target for target in checkpoint_step_targets if environment_steps >= target)
            for target in crossed:
                if target == 0:
                    checkpoint_step_targets.remove(target)
                    continue
                torch.save(
                    policy.state_dict(),
                    checkpoint_directory / f"checkpoint_step_{environment_steps:09d}_target_{target:09d}.pt",
                )
                behavior_rows.append(_checkpoint_row(
                    config,
                    policy,
                    completed,
                    beta=config.beta_at_environment_step(environment_steps),
                    environment_steps=environment_steps,
                ))
                checkpoint_step_targets.remove(target)
        elif completed in checkpoint_set:
            torch.save(policy.state_dict(), checkpoint_directory / f"checkpoint_update_{completed:06d}.pt")
            behavior_rows.append(_checkpoint_row(
                config,
                policy,
                completed,
                beta=config.beta_at_update(completed),
                environment_steps=environment_steps,
            ))
        update += 1
    finally:
        if vector_environments is not None:
            vector_environments.close()

    def write_csv(path: Path, rows: list[dict]) -> None:
        if not rows:
            path.write_text("", encoding="utf-8")
            return
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(output_directory / "training.csv", training_rows)
    write_csv(output_directory / "checkpoint_behavior.csv", behavior_rows)
    write_csv(output_directory / "checkpoint_gradients.csv", gradient_rows)
    final_behavior = behavior_rows[-1] if behavior_rows else {}
    summary = {
        "schema_version": 1,
        "finite": finite,
        "environment": config.environment,
        "method": config.method,
        "seed": config.seed,
        "initial_weight_identifier": weight_id,
        "optimizer": optimizer_name,
        "learning_rate": config.learning_rate,
        "return_convention": {
            "gamma": config.gamma,
            "centered": config.center_returns,
            "normalized": config.normalize_returns,
            "trajectory_outer_mean": True,
        },
        "budget_unit": (
            "complete_training_episodes_and_optimizer_updates"
            if config.collector_mode == "complete_episodes_by_update"
            else "environment_steps"
        ),
        "declared_optimizer_updates": config.updates,
        "actual_optimizer_updates": update,
        "declared_training_episodes": config.total_training_episodes,
        "actual_training_episodes": training_episodes,
        "declared_environment_step_budget": (
            None
            if config.collector_mode == "complete_episodes_by_update"
            else config.total_environment_steps
        ),
        "total_environment_steps": environment_steps,
        "policy_architecture": list(config.hidden_sizes),
        "final": final_behavior,
    }
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def restore_policy(config: NeuralTrainingConfig, checkpoint: str | Path) -> MLPSoftmaxPolicy:
    policy, _ = build_seeded_policy(config)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    policy.load_state_dict(state)
    policy.eval()
    return policy
