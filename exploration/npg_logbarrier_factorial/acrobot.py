"""Frozen Acrobot 2×2 factorial with target-KL-normalized NPG."""

from __future__ import annotations

import copy
import csv
import json
import math
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from vpg.data_collection import collect_parallel_trajectories
from vpg.gpomdp import _compute_empirical_fisher, compute_gpomdp_loss, trajectories_to_tensors
from vpg.stats import mean_confidence_interval

from exploration.neural_discrete_log_barrier.barrier import categorical_log_barrier
from exploration.neural_discrete_log_barrier.fisher import (
    action_enumerated_fisher_spectrum,
    reward_gradient_alignment,
)
from exploration.neural_discrete_log_barrier.training import (
    _flat_gradient,
    _flatten_valid_states,
    build_seeded_policy,
    collect_fixed_step_trajectories,
    evaluate_policy_detailed,
    initial_weight_identifier,
)

from .natural_step import cosine, flatten_parameters, set_parameters, target_kl_natural_step


METHODS = (
    "gpomdp_reward_only",
    "gpomdp_logbarrier_handoff",
    "npg_reward_only",
    "npg_logbarrier_handoff",
)
CHECKPOINTS = (0, 50, 100, 150, 200, 249, 250, 251, 300, 350, 400, 450, 500, 600, 750, 850, 1000)
EXISTING_BETA = 546.4135158976487
PILOT_SEEDS = (611, 612, 613, 614, 615)
CONFIRMATORY_SEEDS = tuple(range(701, 731))


@dataclass(frozen=True)
class AcrobotFactorialConfig:
    method: str
    seed: int
    updates: int = 1000
    episodes_per_update: int = 8
    hidden_sizes: tuple[int, ...] = (8, 8)
    gamma: float = 0.99
    horizon: int = 500
    learning_rate: float = 0.003
    center_returns: bool = True
    normalize_returns: bool = False
    beta: float = EXISTING_BETA
    handoff_update: int = 250
    damping: float = 0.01
    target_kl: float = 1e-3
    evaluation_episodes: int = 32
    parallel_environments: int = 8
    coefficient_mode: str = "same_beta"

    def validate(self) -> None:
        if self.method not in METHODS:
            raise ValueError("unknown Acrobot method")
        if self.updates < 1 or self.episodes_per_update < 1 or self.horizon != 500:
            raise ValueError("invalid frozen Acrobot counts")
        if self.episodes_per_update % self.parallel_environments:
            raise ValueError("episodes per update must divide across workers")
        if self.hidden_sizes != (8, 8) or self.gamma != 0.99:
            raise ValueError("the frozen architecture and discount may not change")
        if not self.center_returns or self.normalize_returns:
            raise ValueError("frozen Acrobot uses centered, unnormalized returns")
        if self.method.startswith("npg") and (self.damping < 0 or self.target_kl <= 0):
            raise ValueError("NPG requires nonnegative damping and positive target KL")
        if not 0 < self.handoff_update < self.updates:
            raise ValueError("handoff must occur inside training")

    @property
    def natural(self) -> bool:
        return self.method.startswith("npg")

    @property
    def barrier(self) -> bool:
        return "logbarrier" in self.method

    def beta_at(self, update: int) -> float:
        return self.beta if self.barrier and update < self.handoff_update else 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def _policy_config(config: AcrobotFactorialConfig):
    """Use the existing builder without changing its implementation."""
    from exploration.neural_discrete_log_barrier.training import NeuralTrainingConfig
    return NeuralTrainingConfig(
        environment="Acrobot-v1", method="gpomdp_reward_only", seed=config.seed,
        hidden_sizes=config.hidden_sizes, learning_rate=config.learning_rate,
        gamma=config.gamma, updates=config.updates, batch_steps=500,
        horizon=config.horizon, center_returns=config.center_returns,
        normalize_returns=config.normalize_returns,
        evaluation_episodes=config.evaluation_episodes,
        collector_mode="complete_episodes_by_update",
        parallel_environments=config.parallel_environments,
        episodes_per_update=config.episodes_per_update,
    )


def _vector_environments(config: AcrobotFactorialConfig):
    def factory(index):
        def build():
            env = gym.make("Acrobot-v1")
            env.reset(seed=config.seed * 100_000 + index)
            env.action_space.seed(config.seed * 100_000 + index + 50_000)
            return env
        return build
    return gym.vector.SyncVectorEnv([factory(i) for i in range(config.parallel_environments)])


def _empirical_fisher(policy, trajectories) -> torch.Tensor:
    states, actions, _, mask = trajectories_to_tensors(trajectories, device="cpu")
    n, horizon = mask.shape
    flat_states = states.reshape(n * horizon, -1)
    flat_actions = actions.reshape(n * horizon).long()
    return _compute_empirical_fisher(
        policy, flat_states, flat_actions, mask.reshape(-1), damping=0.0
    )


def _mean_forward_kl(policy, old_policy, states) -> float:
    with torch.no_grad():
        old = old_policy.distribution(states).probs
        new = policy.distribution(states).probs
        return float((old * (torch.log(old) - torch.log(new))).sum(-1).mean())


def _behavior_row(config, policy, update, environment_steps):
    deterministic, _ = evaluate_policy_detailed(
        "Acrobot-v1", policy, episodes=config.evaluation_episodes,
        horizon=config.horizon, seed_base=7_000_000 + config.seed * 10_000 + update * 31,
        deterministic=True,
    )
    stochastic, states = evaluate_policy_detailed(
        "Acrobot-v1", policy, episodes=config.evaluation_episodes,
        horizon=config.horizon, seed_base=8_000_000 + config.seed * 10_000 + update * 31,
        deterministic=False,
    )
    tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32)
    with torch.no_grad():
        probabilities = policy.distribution(tensor).probs
        log_probabilities = torch.log(probabilities)
        entropy = -(probabilities * log_probabilities).sum(-1)
        minima = probabilities.min(-1).values
        top = torch.topk(probabilities, 2, dim=-1).values
        margins = top[:, 0] - top[:, 1]
    return {
        "method": config.method, "seed": config.seed, "update": update,
        "true_environment_steps": environment_steps,
        "stochastic_return": stochastic["mean_return"],
        "deterministic_return": deterministic["mean_return"],
        "stochastic_termination_rate": stochastic["termination_rate"],
        "deterministic_termination_rate": deterministic["termination_rate"],
        "stochastic_episode_length": stochastic["mean_episode_length"],
        "entropy": float(entropy.mean()),
        "mean_min_probability": float(minima.mean()),
        "global_min_probability": float(minima.min()),
        "minimum_probability_q01": float(torch.quantile(minima, 0.01)),
        "minimum_probability_q05": float(torch.quantile(minima, 0.05)),
        "minimum_probability_q10": float(torch.quantile(minima, 0.10)),
        "action_margin_mean": float(margins.mean()),
        "action_margin_q10": float(torch.quantile(margins, 0.10)),
        "beta": config.beta_at(update),
        "barrier_active": config.beta_at(update) > 0.0,
    }


def _write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _environment_step_auc(behavior_rows) -> float:
    """Time-normalized trapezoidal AUC on true environment steps."""

    steps = np.asarray(
        [float(row["true_environment_steps"]) for row in behavior_rows],
        dtype=np.float64,
    )
    returns = np.asarray(
        [float(row["stochastic_return"]) for row in behavior_rows],
        dtype=np.float64,
    )
    if steps.size == 1 or steps[-1] == steps[0]:
        return float(returns[-1])
    return float(np.trapezoid(returns, steps) / (steps[-1] - steps[0]))


def train_acrobot(config: AcrobotFactorialConfig, output_directory: str | Path) -> dict:
    config.validate()
    output = Path(output_directory)
    stored = json.loads(json.dumps(config.to_dict()))
    if output.exists() and any(output.iterdir()):
        if json.loads((output / "config.json").read_text(encoding="utf-8")) != stored:
            raise FileExistsError(f"incompatible existing run: {output}")
        if (output / "summary.json").exists():
            return json.loads((output / "summary.json").read_text(encoding="utf-8"))
        raise RuntimeError(f"incomplete run needs explicit cleanup: {output}")
    output.mkdir(parents=True, exist_ok=True)
    checkpoints_directory = output / "checkpoints"; checkpoints_directory.mkdir()
    (output / "config.json").write_text(json.dumps(stored, indent=2, sort_keys=True), encoding="utf-8")

    policy, weight_id = build_seeded_policy(_policy_config(config))
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate) if not config.natural else None
    environments = _vector_environments(config)
    behavior_rows = [_behavior_row(config, policy, 0, 0)]
    gradient_rows, training_rows = [], []
    torch.save(policy.state_dict(), checkpoints_directory / "checkpoint_update_000000.pt")
    environment_steps = 0
    finite = True
    invalid_reason = None
    realized_kl_explosions = 0
    try:
        for update in range(config.updates):
            trajectories = collect_parallel_trajectories(
                environments, policy,
                n_trajectories_per_env=config.episodes_per_update // config.parallel_environments,
                clip_actions=False, device="cpu",
            )
            states = _flatten_valid_states(trajectories)
            reward_objective = -compute_gpomdp_loss(
                policy, trajectories, gamma=config.gamma,
                center_returns=config.center_returns,
                normalize_returns=config.normalize_returns,
                entropy_coeff=0.0, device="cpu",
            )
            barrier, _ = categorical_log_barrier(policy(states.detach()))
            beta = config.beta_at(update)
            reward_gradient = _flat_gradient(reward_objective, policy, retain_graph=True)
            barrier_gradient = _flat_gradient(barrier, policy, retain_graph=True)
            total_gradient = reward_gradient + beta * barrier_gradient
            fisher = _empirical_fisher(policy, trajectories)
            identity = torch.eye(fisher.shape[0], dtype=fisher.dtype)
            damped = fisher + config.damping * identity
            natural_reward = torch.linalg.solve(damped, reward_gradient)
            natural_barrier = torch.linalg.solve(damped, beta * barrier_gradient)
            cos_e, cos_e_defined = cosine(reward_gradient, barrier_gradient)
            cos_n, cos_n_defined = cosine(natural_reward, natural_barrier)
            old_policy = copy.deepcopy(policy)
            natural_result = None
            if config.natural:
                natural_result = target_kl_natural_step(
                    total_gradient, fisher, damping=config.damping,
                    target_kl=config.target_kl,
                )
                if not natural_result.valid:
                    finite = False; invalid_reason = natural_result.invalid_reason; break
                set_parameters(policy, flatten_parameters(policy) + natural_result.step)
            else:
                optimizer.zero_grad()
                (-(reward_objective + beta * barrier)).backward()
                optimizer.step()
            realized_kl = _mean_forward_kl(policy, old_policy, states)
            if config.natural and realized_kl > 10.0 * config.target_kl:
                realized_kl_explosions += 1
            batch_steps = sum(len(trajectory.rewards) for trajectory in trajectories)
            environment_steps += batch_steps
            reward_norm = float(reward_gradient.norm())
            natural_reward_norm = float(natural_reward.norm())
            row = {
                "method": config.method, "seed": config.seed, "update": update,
                "true_environment_steps_before_update": environment_steps - batch_steps,
                "beta": beta, "barrier_active": beta > 0.0,
                "reward_gradient_norm": reward_norm,
                "barrier_gradient_norm": float(barrier_gradient.norm()),
                "regularizer_gradient_norm": float((beta * barrier_gradient).norm()),
                "total_gradient_norm": float(total_gradient.norm()),
                "naturalized_reward_norm": natural_reward_norm,
                "naturalized_barrier_norm": float(natural_barrier.norm()),
                "euclidean_barrier_to_reward_ratio": float((beta * barrier_gradient).norm()) / reward_norm if reward_norm else float("nan"),
                "natural_barrier_to_reward_ratio": float(natural_barrier.norm()) / natural_reward_norm if natural_reward_norm else float("nan"),
                "euclidean_cosine": cos_e, "euclidean_cosine_defined": cos_e_defined,
                "natural_cosine": cos_n, "natural_cosine_defined": cos_n_defined,
                "realized_kl": realized_kl, "damping": config.damping,
            }
            if natural_result is not None:
                row.update(natural_result.diagnostics())
            gradient_rows.append(row)
            train_returns = np.asarray([sum(item.rewards) for item in trajectories])
            training_rows.append({
                "update": update + 1, "true_environment_steps": environment_steps,
                "training_return": float(train_returns.mean()),
                "training_episode_length": float(np.mean([len(item.rewards) for item in trajectories])),
                "beta": beta,
            })
            completed = update + 1
            if completed in CHECKPOINTS:
                torch.save(policy.state_dict(), checkpoints_directory / f"checkpoint_update_{completed:06d}.pt")
                behavior_rows.append(_behavior_row(config, policy, completed, environment_steps))
            if not torch.isfinite(flatten_parameters(policy)).all():
                finite = False; invalid_reason = "nonfinite_parameters"; break
    finally:
        environments.close()

    _write_csv(output / "acrobot_checkpoints.csv", behavior_rows)
    _write_csv(output / "gradient_scale_diagnostics.csv", gradient_rows)
    _write_csv(output / "kl_step_diagnostics.csv", gradient_rows)
    _write_csv(output / "training.csv", training_rows)
    final = behavior_rows[-1]
    failure = final["stochastic_return"] < -300 or final["stochastic_termination_rate"] < 0.8
    summary = {
        "schema_version": 1, "complete": finite and len(training_rows) == config.updates,
        "finite": finite, "invalid_reason": invalid_reason,
        "method": config.method, "seed": config.seed,
        "initial_weight_identifier": weight_id,
        "actual_updates": len(training_rows), "training_episodes": len(training_rows) * config.episodes_per_update,
        "true_environment_steps": environment_steps,
        "final_stochastic_return": final["stochastic_return"],
        "final_deterministic_return": final["deterministic_return"],
        "final_stochastic_termination_rate": final["stochastic_termination_rate"],
        "environment_step_return_auc": _environment_step_auc(behavior_rows),
        "failure": failure, "realized_kl_explosion_count": realized_kl_explosions,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _train_task(config, output):
    torch.set_num_threads(1)
    return train_acrobot(config, output)


def run_configs(configs, root: Path, parallel_workers: int):
    tasks = [(config, root / "runs" / config.method / f"seed_{config.seed:03d}") for config in configs]
    if parallel_workers <= 1:
        return [train_acrobot(config, path) for config, path in tasks]
    results = []
    with ProcessPoolExecutor(max_workers=parallel_workers) as executor:
        futures = {executor.submit(_train_task, config, path): config for config, path in tasks}
        for future in as_completed(futures):
            results.append(future.result())
    return results


def run_acrobot_pilot(output: str | Path, *, parallel_workers: int = 1) -> dict:
    root = Path(output); root.mkdir(parents=True, exist_ok=True)
    selection_path = root / "pilot_selection.json"
    if selection_path.exists():
        return json.loads(selection_path.read_text(encoding="utf-8"))
    reward_results = []
    reward_configs = []
    for damping in (0.01, 0.1):
        group_configs = [
            AcrobotFactorialConfig(
                "npg_reward_only", seed, updates=300,
                handoff_update=250, damping=damping,
            )
            for seed in PILOT_SEEDS
        ]
        reward_configs.extend(group_configs)
        reward_results.extend(run_configs(
            group_configs, root / f"damping_{str(damping).replace('.', 'p')}", parallel_workers
        ))
    viable = []
    for damping in (0.01, 0.1):
        group_root = root / f"damping_{str(damping).replace('.', 'p')}" / "runs" / "npg_reward_only"
        group = [
            json.loads((group_root / f"seed_{seed:03d}" / "summary.json").read_text(encoding="utf-8"))
            for seed in PILOT_SEEDS
        ]
        finite = all(row["finite"] for row in group)
        learned = sum(row["final_stochastic_return"] > -300 for row in group)
        explosions = sum(row["realized_kl_explosion_count"] for row in group)
        if finite and learned >= 2 and explosions <= len(group):
            viable.append(damping)
    if not viable:
        result = {"schema_version": 1, "gate_passed": False, "reason": "no stable learning NPG damping", "outcomes_used_for_confirmatory_selection": False}
        selection_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
        return result
    selected_damping = viable[0]
    barrier_configs = [
        AcrobotFactorialConfig(
            "npg_logbarrier_handoff", seed, updates=300,
            handoff_update=250, damping=selected_damping,
        )
        for seed in PILOT_SEEDS
    ]
    selected_root = root / "selected"
    run_configs(barrier_configs, selected_root, parallel_workers)
    ratios = []
    for config in barrier_configs:
        path = selected_root / "runs" / config.method / f"seed_{config.seed:03d}" / "gradient_scale_diagnostics.csv"
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if int(row["update"]) < min(50, config.handoff_update) and math.isfinite(float(row["natural_barrier_to_reward_ratio"])):
                    ratios.append(float(row["natural_barrier_to_reward_ratio"]))
    median_ratio = float(np.median(ratios))
    if median_ratio > 1.0:
        selected_beta = EXISTING_BETA * 0.3 / median_ratio
        coefficient_mode = "natural_scale_matched"
    else:
        selected_beta = EXISTING_BETA
        coefficient_mode = "same_beta"
    result = {
        "schema_version": 1, "gate_passed": True,
        "selected_damping": selected_damping, "selected_target_kl": 1e-3,
        "same_beta_median_early_natural_ratio": median_ratio,
        "selected_beta": selected_beta, "coefficient_mode": coefficient_mode,
        "selection_uses_final_confirmatory_outcomes": False,
    }
    selection_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def _wilson(count, n):
    z = 1.959963984540054
    center = (count / n + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt((count / n) * (1 - count / n) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return center - half, center + half


def summarize_confirmatory(root: Path, configs) -> dict:
    endpoints = []
    checkpoint_rows, gradient_rows, kl_rows, config_rows = [], [], [], []
    for config in configs:
        run = root / "runs" / config.method / f"seed_{config.seed:03d}"
        endpoints.append(json.loads((run / "summary.json").read_text(encoding="utf-8")))
        config_rows.append(config.to_dict())
        for filename, destination in (
            ("acrobot_checkpoints.csv", checkpoint_rows),
            ("gradient_scale_diagnostics.csv", gradient_rows),
            ("kl_step_diagnostics.csv", kl_rows),
        ):
            with (run / filename).open(newline="", encoding="utf-8") as handle:
                destination.extend(csv.DictReader(handle))
    _write_csv(root / "acrobot_endpoints.csv", endpoints)
    _write_csv(root / "acrobot_checkpoints.csv", checkpoint_rows)
    _write_csv(root / "gradient_scale_diagnostics.csv", gradient_rows)
    _write_csv(root / "kl_step_diagnostics.csv", kl_rows)
    _write_csv(root / "method_configs.csv", config_rows)
    failures, method_summaries, paired = [], [], []
    by_method = {method: [row for row in endpoints if row["method"] == method] for method in METHODS}
    for method, rows in by_method.items():
        count = sum(bool(row["failure"]) for row in rows); low, high = _wilson(count, len(rows))
        failures.append({"method": method, "n": len(rows), "failures": count, "failure_rate": count / len(rows), "wilson95_low": low, "wilson95_high": high})
        summary = {"method": method, "n": len(rows)}
        for metric in (
            "final_stochastic_return",
            "final_deterministic_return",
            "environment_step_return_auc",
        ):
            values = np.asarray([float(row[metric]) for row in rows])
            mean, ci_low, ci_high = mean_confidence_interval(values)
            summary.update({
                f"{metric}_mean": float(mean),
                f"{metric}_median": float(np.median(values)),
                f"{metric}_ci95_low": float(ci_low),
                f"{metric}_ci95_high": float(ci_high),
            })
        method_summaries.append(summary)
    comparisons = (
        ("gpomdp_logbarrier_handoff", "gpomdp_reward_only", "barrier_within_euclidean"),
        ("npg_logbarrier_handoff", "npg_reward_only", "barrier_within_natural"),
        ("npg_reward_only", "gpomdp_reward_only", "optimizer_reward_only"),
        ("npg_logbarrier_handoff", "gpomdp_logbarrier_handoff", "optimizer_with_barrier"),
    )
    for method, reference_method, comparison_family in comparisons:
        reference = {row["seed"]: row for row in by_method[reference_method]}
        current = {row["seed"]: row for row in by_method[method]}
        common = sorted(set(reference) & set(current))
        for metric in (
            "final_stochastic_return",
            "final_deterministic_return",
            "environment_step_return_auc",
            "failure",
        ):
            differences = np.asarray([float(current[s][metric]) - float(reference[s][metric]) for s in common])
            mean, low, high = mean_confidence_interval(differences)
            paired.append({"comparison_family": comparison_family, "method": method, "reference": reference_method, "metric": metric, "n": len(common), "mean_difference": float(mean), "ci95_low": float(low), "ci95_high": float(high)})
        method_failed_reference_succeeded = sum(bool(current[s]["failure"]) and not bool(reference[s]["failure"]) for s in common)
        method_succeeded_reference_failed = sum(not bool(current[s]["failure"]) and bool(reference[s]["failure"]) for s in common)
        discordant = method_failed_reference_succeeded + method_succeeded_reference_failed
        if discordant:
            tail = sum(math.comb(discordant, k) for k in range(min(method_failed_reference_succeeded, method_succeeded_reference_failed) + 1)) / (2 ** discordant)
            mcnemar_p = min(1.0, 2.0 * tail)
        else:
            mcnemar_p = 1.0
        paired.append({
            "comparison_family": comparison_family, "method": method,
            "reference": reference_method, "metric": "exact_mcnemar_p",
            "n": len(common), "mean_difference": mcnemar_p,
            "ci95_low": "", "ci95_high": "",
            "method_failed_reference_succeeded": method_failed_reference_succeeded,
            "method_succeeded_reference_failed": method_succeeded_reference_failed,
        })
    _write_csv(root / "acrobot_failure_table.csv", failures)
    _write_csv(root / "acrobot_method_summaries.csv", method_summaries)
    _write_csv(root / "acrobot_paired_differences.csv", paired)
    return {
        "endpoint_rows": len(endpoints),
        "failure_rows": len(failures),
        "method_summary_rows": len(method_summaries),
        "paired_rows": len(paired),
    }


def run_acrobot_confirmatory(output: str | Path, pilot_output: str | Path, *, parallel_workers: int = 1) -> dict:
    root = Path(output)
    pilot = json.loads((Path(pilot_output) / "pilot_selection.json").read_text(encoding="utf-8"))
    if not pilot.get("gate_passed"):
        raise RuntimeError("Acrobot NPG pilot gate has not passed")
    configs = []
    for seed in CONFIRMATORY_SEEDS:
        for method in METHODS:
            configs.append(AcrobotFactorialConfig(
                method, seed, damping=float(pilot["selected_damping"]),
                target_kl=float(pilot["selected_target_kl"]),
                beta=float(pilot["selected_beta"]),
                coefficient_mode=str(pilot["coefficient_mode"]),
            ))
    run_configs(configs, root, parallel_workers)
    summary = summarize_confirmatory(root, configs)
    manifest = {
        "schema_version": 1, "complete": True, "paired_seeds": len(CONFIRMATORY_SEEDS),
        "methods": list(METHODS), "pilot_selection": pilot, **summary,
        "historical_pg_archives_overwritten": False,
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def run_fisher_diagnostics(confirmatory_root: str | Path, output: str | Path) -> dict:
    source, destination = Path(confirmatory_root), Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    bank = Path("exploration/results/neural_discrete_log_barrier/state_banks/acrobot_reference_states.npz")
    with np.load(bank) as archive:
        fixed_states = torch.as_tensor(archive["states"], dtype=torch.float32)
    on_rows, fixed_rows = [], []
    for run in sorted((source / "runs").glob("*/seed_*")):
        config_data = json.loads((run / "config.json").read_text(encoding="utf-8"))
        config_data["hidden_sizes"] = tuple(config_data["hidden_sizes"])
        config = AcrobotFactorialConfig(**config_data)
        for checkpoint in sorted((run / "checkpoints").glob("checkpoint_update_*.pt")):
            update = int(checkpoint.stem.split("_")[-1])
            policy, _ = build_seeded_policy(_policy_config(config))
            policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
            generator = torch.Generator(device="cpu").manual_seed(93_000_000 + config.seed * 1000 + update)
            trajectories = collect_fixed_step_trajectories(
                "Acrobot-v1", policy, step_count=128, horizon=500,
                reset_seed_base=94_000_000 + config.seed * 1000 + update,
                action_generator=generator,
            )
            on_states = _flatten_valid_states(trajectories)
            reward_objective = -compute_gpomdp_loss(
                policy,
                trajectories,
                gamma=config.gamma,
                center_returns=config.center_returns,
                normalize_returns=config.normalize_returns,
                entropy_coeff=0.0,
                device="cpu",
            )
            reward_gradient = _flat_gradient(reward_objective, policy).detach()
            for label, states, target in (("on_policy", on_states, on_rows), ("fixed_reference", fixed_states, fixed_rows)):
                spectrum = action_enumerated_fisher_spectrum(copy.deepcopy(policy), states)
                alignment, alignment_arrays = reward_gradient_alignment(
                    spectrum, reward_gradient
                )
                target.append({
                    "method": config.method, "seed": config.seed, "update": update,
                    **spectrum.metrics.to_dict(),
                    **alignment.to_dict(),
                    "positive_eigenvalues_json": json.dumps(spectrum.eigenvalues.tolist()),
                    "gradient_coordinates_json": json.dumps(
                        alignment_arrays["gradient_coordinates"].tolist()
                    ),
                    "cumulative_euclidean_fraction_json": json.dumps(
                        alignment_arrays["cumulative_euclidean_fraction"].tolist()
                    ),
                    "cumulative_natural_fraction_json": json.dumps(
                        alignment_arrays["cumulative_natural_fraction"].tolist()
                    ),
                })
    _write_csv(destination / "fisher_on_policy.csv", on_rows)
    _write_csv(destination / "fisher_fixed_reference.csv", fixed_rows)
    result = {"schema_version": 1, "complete": True, "on_policy_rows": len(on_rows), "fixed_reference_rows": len(fixed_rows), "fisher_is_primary_endpoint": False}
    (destination / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result
