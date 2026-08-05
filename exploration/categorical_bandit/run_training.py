"""CLI for Step 2 categorical-bandit log-barrier training."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch

from exploration.categorical_bandit.environment import BanditBatch, generate_paired_bandits
from exploration.categorical_bandit.experiment import TrainingResult, run_training_unit, stable_seed
from exploration.categorical_bandit.presets import build_preset, unit_filename
from exploration.categorical_bandit.reporting import (
    make_configuration_plots,
    write_paired_differences,
    write_run_status,
    write_summary,
)


DEFAULT_OUTPUT_ROOT = Path("exploration/results/categorical_bandit")


def _resolved_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return requested


def _manifest(suite) -> dict[str, object]:
    data = suite.to_dict()
    data.update(
        {
            "torch_version": torch.__version__,
            "reward_model": "stationary Gaussian N(mean[action], 1)",
            "instance_construction": {
                "best_mean": 1.0,
                "second_best_mean": 0.9,
                "remaining_means": "Uniform[-1.0, 0.9)",
                "random_arm_permutation": True,
            },
            "paper_source": "https://arxiv.org/abs/2603.15001v2",
            "paper_grid_eta": {
                "K10_alpha0.01": 1000.0,
                "K10_alpha0.1": 1000.0,
                "K100_alpha0.01": 2000.0,
                "K100_alpha0.1": 2000.0,
                "K1000_alpha0.01": 10000.0,
                "K1000_alpha0.1": 5000.0,
            },
        }
    )
    return data


def _prepare_output(output_dir: Path, manifest: dict[str, object], resume: bool) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    contents = list(output_dir.iterdir())
    config_path = output_dir / "config.json"
    if contents:
        if not config_path.exists():
            raise ValueError(f"Non-empty output directory has no compatible config: {output_dir}")
        saved = json.loads(config_path.read_text(encoding="utf-8"))
        if saved != manifest:
            raise ValueError(f"Output directory contains an incompatible configuration: {output_dir}")
        if not resume:
            raise ValueError(f"Output directory is non-empty; use --resume to continue: {output_dir}")
        return
    config_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _load_or_save_bandits(
    output_dir: Path,
    num_actions: int,
    num_runs: int,
    base_seed: int,
    resume: bool,
) -> BanditBatch:
    path = output_dir / f"paired_bandits_K{num_actions:04d}.npz"
    seed = stable_seed(base_seed, "instances", num_actions)
    expected = generate_paired_bandits(num_runs, num_actions, seed)
    if path.exists():
        if not resume:
            raise ValueError(f"Bandit archive already exists: {path}")
        with np.load(path, allow_pickle=False) as archive:
            loaded = BanditBatch(
                mean_rewards=torch.from_numpy(archive["mean_rewards"].copy()),
                optimal_actions=torch.from_numpy(archive["optimal_actions"].copy()),
                second_best_actions=torch.from_numpy(archive["second_best_actions"].copy()),
            )
        if not (
            torch.equal(loaded.mean_rewards, expected.mean_rewards)
            and torch.equal(loaded.optimal_actions, expected.optimal_actions)
            and torch.equal(loaded.second_best_actions, expected.second_best_actions)
        ):
            raise ValueError(f"Saved paired bandits do not match their declared seed: {path}")
        return loaded
    np.savez_compressed(
        path,
        schema_version=np.asarray(1, dtype=np.int64),
        seed=np.asarray(seed, dtype=np.int64),
        mean_rewards=expected.mean_rewards.numpy(),
        optimal_actions=expected.optimal_actions.numpy(),
        second_best_actions=expected.second_best_actions.numpy(),
    )
    return expected


def run_preset(
    preset: str,
    *,
    output_dir: Path,
    base_seed: int = 23,
    device: str = "auto",
    resume: bool = False,
) -> list[TrainingResult]:
    """Run or resume a complete named preset."""
    resolved_device = _resolved_device(device)
    suite = build_preset(preset, base_seed=base_seed, device=resolved_device)
    manifest = _manifest(suite)
    _prepare_output(output_dir, manifest, resume)

    bandits: dict[int, BanditBatch] = {}
    for num_actions in sorted({unit.num_actions for unit in suite.units}):
        matching = next(unit for unit in suite.units if unit.num_actions == num_actions)
        bandits[num_actions] = _load_or_save_bandits(
            output_dir, num_actions, matching.num_runs, base_seed, resume
        )

    results: list[TrainingResult] = []
    for index, unit in enumerate(suite.units, start=1):
        path = output_dir / unit_filename(unit)
        print(
            f"[{index}/{len(suite.units)}] K={unit.num_actions}, "
            f"alpha={unit.algorithm.learning_rate:g}, {unit.algorithm.key}",
            flush=True,
        )
        if path.exists():
            if not resume:
                raise ValueError(f"Result unit already exists: {path}")
            result = TrainingResult.load(path, unit)
            print("  complete archive validated; skipped", flush=True)
        else:
            result = run_training_unit(
                unit, bandits[unit.num_actions], device=resolved_device, progress=True
            )
            result.save(path)
        results.append(result)

    write_run_status(output_dir / "run_status.csv", results, bandits)
    write_summary(output_dir / "summary.csv", results, bandits)
    write_paired_differences(output_dir / "paired_final_differences.csv", results)
    make_configuration_plots(output_dir, results)
    print(f"Completed {preset} preset: {output_dir}", flush=True)
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "pilot", "eta", "paper"), required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_dir = args.output_dir or DEFAULT_OUTPUT_ROOT / args.preset
    try:
        run_preset(
            args.preset,
            output_dir=output_dir,
            base_seed=args.seed,
            device=args.device,
            resume=args.resume,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
