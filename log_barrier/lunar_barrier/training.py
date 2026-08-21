"""GPOMDP training and held-out evaluation for discrete LunarLander."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import gymnasium as gym
import numpy as np
import torch

from log_barrier.acrobot.barrier import categorical_log_barrier
from vpg.data_collection import collect_parallel_trajectories
from vpg.gpomdp import compute_gpomdp_loss, trajectories_to_tensors
from vpg.policy import build_policy


METHODS = ("reward_only", "log_barrier")


def _make_env(env_id: str, seed: int, horizon: int):
    """Return an isolated environment factory without importing the full trainer."""

    def thunk():
        env = gym.make(env_id, max_episode_steps=horizon)
        env.reset(seed=seed)
        return env

    return thunk


@dataclass(frozen=True)
class LunarLanderConfig:
    method: str
    seed: int
    learning_rate: float
    beta: float = 0.0
    env_id: str = "LunarLander-v3"
    hidden_sizes: tuple[int, ...] = (8, 8)
    gamma: float = 0.99
    updates: int = 1000
    episodes_per_update: int = 8
    horizon: int = 1000
    center_returns: bool = True
    normalize_returns: bool = False
    evaluation_episodes: int = 32
    handoff_fraction: float = 0.25
    device: str = "cpu"

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError(f"unknown method: {self.method}")
        if self.learning_rate <= 0.0 or self.beta < 0.0:
            raise ValueError("learning_rate must be positive and beta non-negative")
        if not 0.0 <= self.gamma <= 1.0:
            raise ValueError("gamma must be between zero and one")
        if min(self.updates, self.episodes_per_update, self.evaluation_episodes) < 1:
            raise ValueError("updates, episodes_per_update, and evaluation_episodes must be positive")
        if self.horizon < 1:
            raise ValueError("horizon must be positive")
        if not 0.0 < self.handoff_fraction <= 1.0:
            raise ValueError("handoff_fraction must be in (0, 1]")
        if not self.hidden_sizes or any(width < 1 for width in self.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive widths")

    def to_dict(self) -> dict:
        result = asdict(self)
        result["hidden_sizes"] = list(self.hidden_sizes)
        result["optimizer"] = "Adam"
        return result


def _valid_states(trajectories, device: torch.device) -> torch.Tensor:
    states, _, _, mask = trajectories_to_tensors(trajectories, device=device)
    return states.reshape(-1, states.shape[-1])[mask.reshape(-1).bool()].detach()


def _evaluate(policy, config: LunarLanderConfig) -> tuple[list[float], list[float]]:
    """Evaluate with common environment seeds and common action uniforms."""

    stochastic_returns: list[float] = []
    deterministic_returns: list[float] = []
    device = next(policy.parameters()).device
    policy.eval()
    for deterministic, destination in (
        (False, stochastic_returns),
        (True, deterministic_returns),
    ):
        env = gym.make(config.env_id, max_episode_steps=config.horizon)
        try:
            for episode in range(config.evaluation_episodes):
                evaluation_seed = 1_000_000 + config.seed * 100 + episode
                state, _ = env.reset(seed=evaluation_seed)
                uniform_rng = np.random.default_rng(evaluation_seed + 50_000_000)
                total = 0.0
                terminated = truncated = False
                while not (terminated or truncated):
                    with torch.no_grad():
                        logits = policy(torch.as_tensor(state, dtype=torch.float32, device=device))
                        if deterministic:
                            action = int(logits.argmax())
                        else:
                            probabilities = torch.softmax(logits, dim=-1).cpu().numpy()
                            action = int(np.searchsorted(np.cumsum(probabilities), uniform_rng.random()))
                            action = min(action, probabilities.size - 1)
                    state, reward, terminated, truncated, _ = env.step(action)
                    total += float(reward)
                destination.append(total)
        finally:
            env.close()
    policy.train()
    return stochastic_returns, deterministic_returns


def train_and_evaluate(config: LunarLanderConfig) -> dict:
    """Train one independently seeded policy and return compact behavioral results."""

    config.validate()
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    device = torch.device(config.device)
    probe = gym.make(config.env_id)
    try:
        if not isinstance(probe.action_space, gym.spaces.Discrete):
            raise ValueError("the categorical barrier requires a discrete action space")
        policy = build_policy(
            {"hidden_sizes": config.hidden_sizes, "policy": "mlp"}, probe
        ).to(device)
    finally:
        probe.close()

    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    envs = gym.vector.SyncVectorEnv(
        [
            _make_env(config.env_id, config.seed + worker, config.horizon)
            for worker in range(config.episodes_per_update)
        ]
    )
    training_returns: list[float] = []
    minimum_probabilities: list[float] = []
    environment_steps = 0
    handoff_update = int(round(config.updates * config.handoff_fraction))
    try:
        for update in range(config.updates):
            trajectories = collect_parallel_trajectories(envs, policy, 1, device=device)
            states = _valid_states(trajectories, device)
            reward_loss = compute_gpomdp_loss(
                policy,
                trajectories,
                gamma=config.gamma,
                center_returns=config.center_returns,
                normalize_returns=config.normalize_returns,
                device=device,
            )
            loss = reward_loss
            barrier_active = (
                config.method == "log_barrier"
                and config.beta > 0.0
                and update < handoff_update
            )
            if barrier_active:
                barrier, diagnostics = categorical_log_barrier(policy(states))
                loss = loss - config.beta * barrier
                minimum_probabilities.append(diagnostics.mean_min_probability)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            episode_returns = [sum(trajectory.rewards) for trajectory in trajectories]
            training_returns.append(float(np.mean(episode_returns)))
            environment_steps += sum(len(trajectory.rewards) for trajectory in trajectories)
            if not np.isfinite(training_returns[-1]):
                raise FloatingPointError("non-finite training return")
    finally:
        envs.close()

    stochastic, deterministic = _evaluate(policy, config)
    tail_size = max(1, min(len(training_returns), max(10, config.updates // 10)))
    return {
        "schema_version": 1,
        "config": config.to_dict(),
        "seed": config.seed,
        "method": config.method,
        "environment_steps": environment_steps,
        "handoff_fraction": config.handoff_fraction,
        "handoff_update": handoff_update,
        "training_returns": training_returns,
        "final_training_mean": float(np.mean(training_returns[-tail_size:])),
        "mean_min_probability": (
            float(np.mean(minimum_probabilities[-tail_size:]))
            if minimum_probabilities
            else None
        ),
        "stochastic_evaluation_returns": stochastic,
        "deterministic_evaluation_returns": deterministic,
        "stochastic_evaluation_mean": float(np.mean(stochastic)),
        "deterministic_evaluation_mean": float(np.mean(deterministic)),
    }
