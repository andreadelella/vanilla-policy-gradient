"""Named experiment grids for the categorical-bandit reproduction."""

from dataclasses import dataclass
from typing import Any

from exploration.categorical_bandit.algorithms import AlgorithmSpec
from exploration.categorical_bandit.experiment import TrainingConfig, stable_seed


PAPER_ETA = {
    (10, 0.01): 1000.0,
    (10, 0.1): 1000.0,
    (100, 0.01): 2000.0,
    (100, 0.1): 2000.0,
    (1000, 0.01): 10000.0,
    (1000, 0.1): 5000.0,
}


@dataclass(frozen=True)
class PresetSuite:
    name: str
    base_seed: int
    device: str
    units: tuple[TrainingConfig, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "preset": self.name,
            "base_seed": self.base_seed,
            "device": self.device,
            "dtype": "torch.float64",
            "paired_unit": "bandit run",
            "confidence_interval": "two-sided 95% Student-t, df=N-1",
            "units": [unit.to_dict() for unit in self.units],
        }


def _standard_algorithms(alpha: float, eta: float) -> tuple[AlgorithmSpec, ...]:
    return (
        AlgorithmSpec("sgb", "sgb", alpha),
        AlgorithmSpec("entropy_sgb", "entropy_sgb", alpha, entropy_coefficient=1.0 / eta),
        AlgorithmSpec("npg", "npg", alpha),
        AlgorithmSpec("lb_sgb", "lb_sgb", alpha, eta=eta),
    )


def _unit(
    preset: str,
    num_actions: int,
    num_runs: int,
    horizon: int,
    alpha: float,
    algorithm: AlgorithmSpec,
    base_seed: int,
) -> TrainingConfig:
    return TrainingConfig(
        preset=preset,
        num_actions=num_actions,
        num_runs=num_runs,
        horizon=horizon,
        record_interval=25,
        reward_std=1.0,
        collapse_threshold=1e-12,
        seed=stable_seed(base_seed, "trajectory", preset, num_actions, alpha, algorithm.key),
        algorithm=algorithm,
    )


def build_preset(name: str, *, base_seed: int = 23, device: str = "cpu") -> PresetSuite:
    """Build a deterministic preset without launching it."""
    units: list[TrainingConfig] = []
    if name == "smoke":
        for algorithm in _standard_algorithms(0.01, 1000.0):
            units.append(_unit(name, 10, 4, 200, 0.01, algorithm, base_seed))
    elif name == "pilot":
        for num_actions in (10, 100):
            for alpha in (0.01, 0.1):
                eta = PAPER_ETA[(num_actions, alpha)]
                for algorithm in _standard_algorithms(alpha, eta):
                    units.append(
                        _unit(name, num_actions, 10, 2500, alpha, algorithm, base_seed)
                    )
    elif name == "eta":
        alpha = 0.1
        algorithms = [AlgorithmSpec("sgb", "sgb", alpha)]
        algorithms.extend(
            AlgorithmSpec(f"lb_sgb_eta_{int(eta)}", "lb_sgb", alpha, eta=eta)
            for eta in (100.0, 1000.0, 10000.0)
        )
        for algorithm in algorithms:
            units.append(_unit(name, 10, 100, 25000, alpha, algorithm, base_seed))
    elif name == "paper":
        for num_actions in (10, 100, 1000):
            for alpha in (0.01, 0.1):
                eta = PAPER_ETA[(num_actions, alpha)]
                for algorithm in _standard_algorithms(alpha, eta):
                    units.append(
                        _unit(name, num_actions, 100, 25000, alpha, algorithm, base_seed)
                    )
    else:
        raise ValueError(f"Unknown preset: {name}")
    return PresetSuite(name=name, base_seed=base_seed, device=device, units=tuple(units))


def configuration_key(config: TrainingConfig) -> str:
    alpha = f"{config.algorithm.learning_rate:g}".replace(".", "p")
    return f"K{config.num_actions:04d}_alpha{alpha}"


def unit_filename(config: TrainingConfig) -> str:
    return f"{configuration_key(config)}__{config.algorithm.key}.npz"
