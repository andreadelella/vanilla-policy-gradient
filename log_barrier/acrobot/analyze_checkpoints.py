"""Compute empirical policy-Fisher spectra from saved Acrobot checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from vpg.policy import build_policy

from .fisher import empirical_policy_fisher_spectrum
from .reporting import plot_checkpoint_behavior, plot_explained_trace, plot_fisher


def _checkpoint_path(root: Path, directory: str, seed: int, update: int) -> Path:
    candidates = (
        root / directory / f"seed_{seed}" / "checkpoints" / f"checkpoint_update_{update:06d}.pt",
        root / f"seed_{seed}__{directory}" / "checkpoints" / f"update_{update:04d}.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def _states_for_checkpoint(policy, seed: int, update: int, episodes: int, maximum_states: int) -> torch.Tensor:
    analysis_seed = seed * 100_000 + update
    env = gym.make("Acrobot-v1", max_episode_steps=500)
    collected = []
    try:
        for episode in range(episodes):
            state, _ = env.reset(seed=analysis_seed + episode)
            terminated = truncated = False
            while not (terminated or truncated):
                collected.append(np.asarray(state, dtype=np.float32))
                with torch.no_grad():
                    action = int(policy.sample_action_tensor(torch.as_tensor(state).float().unsqueeze(0))[0])
                state, _, terminated, truncated, _ = env.step(action)
    finally:
        env.close()
    valid = torch.as_tensor(np.asarray(collected), dtype=torch.float32)
    if maximum_states > 0 and valid.shape[0] > maximum_states:
        # Even spacing preserves the full time range and is deterministic.
        indices = torch.linspace(0, valid.shape[0] - 1, maximum_states).round().long()
        valid = valid[indices]
    return valid


def analyze(
    source: Path,
    output: Path,
    seeds: list[int],
    method_directories: dict[str, str],
    episodes: int,
    maximum_states: int,
) -> list[dict]:
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    probe = gym.make("Acrobot-v1")
    try:
        for seed in seeds:
            for method, directory in method_directories.items():
                for update in range(0, 1001, 100):
                    checkpoint = _checkpoint_path(source, directory, seed, update)
                    if not checkpoint.is_file():
                        raise FileNotFoundError(checkpoint)
                    policy = build_policy({"hidden_sizes": (8, 8), "policy": "mlp"}, probe).cpu()
                    policy.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
                    policy.eval()
                    states = _states_for_checkpoint(policy, seed, update, episodes, maximum_states)
                    spectrum = empirical_policy_fisher_spectrum(policy, states)
                    row = {
                        "seed": seed,
                        "method": method,
                        "update": update,
                        **spectrum.metrics.to_dict(),
                    }
                    row.update({f"eigenvalue_{index + 1}": float(value) for index, value in enumerate(spectrum.eigenvalues)})
                    rows.append(row)
                    _write_rows(output / "fisher_spectra.partial.csv", rows)
    finally:
        probe.close()
    _write_rows(output / "fisher_spectra.csv", rows)
    partial = output / "fisher_spectra.partial.csv"
    if partial.exists():
        partial.unlink()
    plot_fisher(rows, output / "fisher_spectra.png")
    plot_explained_trace(rows, output)
    behavior_rows = _load_behavior_rows(source, seeds, method_directories)
    if behavior_rows:
        _write_rows(output / "behavior_checkpoints.csv", behavior_rows)
        plot_checkpoint_behavior(behavior_rows, output / "fixed_barrier_behavior.png")
    provenance = {
        "schema_version": 1,
        "source": str(source),
        "seeds": seeds,
        "methods": list(method_directories),
        "checkpoint_directories": method_directories,
        "checkpoint_updates": list(range(0, 1001, 100)),
        "episodes_per_checkpoint": episodes,
        "maximum_states_per_checkpoint": maximum_states,
        "state_sampling": "fresh deterministic on-policy rollouts; evenly spaced truncation when necessary",
        "action_sampling": "none; all categorical actions enumerated exactly",
        "damping": 0.0,
        "spectral_floor": False,
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    return rows


def _write_rows(path: Path, rows: list[dict]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _load_behavior_rows(source: Path, seeds: list[int], method_directories: dict[str, str]) -> list[dict]:
    """Read the decile evaluation rows stored beside each checkpoint set."""

    rows = []
    for seed in seeds:
        for method, directory in method_directories.items():
            path = source / directory / f"seed_{seed}" / "checkpoint_behavior.csv"
            if not path.is_file():
                continue
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    update = int(row["update"])
                    if update % 100 == 0:
                        rows.append(
                            {
                                "seed": seed,
                                "method": method,
                                "update": update,
                                "stochastic_return": float(row["stochastic_return"]),
                                "deterministic_return": float(row["deterministic_return"]),
                                "entropy": float(row["entropy"]),
                                "mean_min_probability": float(row["mean_min_probability"]),
                            }
                        )
    return rows


def _method_directories(specification: str) -> dict[str, str]:
    """Parse ``label=directory`` pairs used for plot labels and input paths."""

    result = {}
    for item in specification.split(","):
        label, separator, directory = item.strip().partition("=")
        if not separator or not label or not directory:
            raise ValueError("--methods must contain comma-separated label=directory pairs")
        result[label] = directory
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("results/log_barrier/acrobot/checkpoints"),
        help="Directory containing the saved checkpoint sets",
    )
    parser.add_argument("--output", type=Path, default=Path("results/log_barrier/acrobot/fixed_barrier_fisher"))
    parser.add_argument("--seeds", default="401,402,403,404,405")
    parser.add_argument("--methods", default="reward_only=reward_only,log_barrier=logbarrier_fixed")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--maximum-states", type=int, default=32)
    args = parser.parse_args(argv)
    analyze(
        args.source,
        args.output,
        [int(value) for value in args.seeds.split(",")],
        _method_directories(args.methods),
        args.episodes,
        args.maximum_states,
    )
    print(f"Saved checkpoint Fisher analysis to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
