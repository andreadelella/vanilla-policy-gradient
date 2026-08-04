"""Run the temporary sampled-conditional barrier handoff experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from .experiment import SampledTrainingConfig, SampledTrainingResult
from .reporting import (
    handoff_change_rows,
    handoff_checkpoint_rows,
    handoff_pairwise_rows,
    make_handoff_plots,
    paired_difference_rows,
    training_summary_rows,
    write_rows,
)
from .run_experiment import _run_unit, prepare_output
from .verify import TOLERANCES, run_verification


DEFAULT_OUTPUT_ROOT = Path(
    "exploration/results/tabular_mdp/two_step_trap_sampled/handoff"
)


def manifest(preset: str) -> dict:
    scales = {
        "smoke": {"n_seeds": 4, "updates": 20, "handoff_update": 10},
        "full": {"n_seeds": 100, "updates": 4000, "handoff_update": 2000},
    }
    if preset not in scales:
        raise ValueError(f"unknown preset: {preset}")
    scale = scales[preset]
    return {
        "schema_version": 1,
        "stage": "sampled_two_state_temporary_barrier_handoff",
        "preset": preset,
        "dtype": "torch.float64",
        "device": "cpu",
        "alpha": 0.05,
        "n_trajectories": 32,
        "record_interval": 10,
        "base_seed": 23,
        "initializations": ["uniform", "adverse"],
        "returns": "raw reward-to-go; uncentered and unnormalized",
        "rewards": "deterministic",
        "stream_derivation": (
            "one CPU torch.Generator seeded 23 per unit; identical tensor shapes "
            "give paired base-uniform streams across methods"
        ),
        "schedule_indexing": (
            "zero-based t; the first handoff_update optimizer updates use beta, "
            "then beta_after"
        ),
        "curves": [
            {
                "label": "reward_only",
                "method": "reward_only",
                "beta": 0.0,
                "beta_after": None,
                "role": "baseline",
            },
            {
                "label": "sampled_conditional_fixed",
                "method": "detached_conditional_sampled",
                "beta": 0.2,
                "beta_after": None,
                "role": "fixed practical barrier",
            },
            {
                "label": "sampled_conditional_handoff",
                "method": "detached_conditional_sampled",
                "beta": 0.2,
                "beta_after": 0.0,
                "handoff_update": scale["handoff_update"],
                "role": "candidate algorithm",
            },
            {
                "label": "full_oracle_handoff",
                "method": "full_pooled_fisher_oracle",
                "beta": 0.1,
                "beta_after": 0.0,
                "handoff_update": scale["handoff_update"],
                "role": "diagnostic upper reference only",
            },
        ],
        "scale": scale,
        "torch_version": torch.__version__,
    }


def _config_from_curve(
    curve: dict, initialization: str, suite: dict
) -> SampledTrainingConfig:
    scale = suite["scale"]
    return SampledTrainingConfig(
        method=curve["method"],
        initialization=initialization,
        n_trajectories=suite["n_trajectories"],
        n_seeds=scale["n_seeds"],
        alpha=suite["alpha"],
        beta=curve["beta"],
        beta_after=curve.get("beta_after"),
        handoff_update=curve.get("handoff_update"),
        updates=scale["updates"],
        record_interval=suite["record_interval"],
        base_seed=suite["base_seed"],
        reward_noise_std=0.0,
        center_returns=False,
        normalize_returns=False,
        label=curve["label"],
    )


def run_handoff_suite(
    output: Path, *, preset: str = "full", resume: bool = False
) -> list[tuple[str, SampledTrainingResult]]:
    suite = manifest(preset)
    prepare_output(output, suite, resume)
    verification = run_verification()
    (output / "verification.json").write_text(
        json.dumps(
            {"schema_version": 1, "tolerances": TOLERANCES, **verification.to_dict()},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    if not verification.passed:
        raise ValueError("sampled estimator verification failed")

    results: list[tuple[str, SampledTrainingResult]] = []
    status: list[dict] = []
    for initialization in suite["initializations"]:
        group: list[SampledTrainingResult] = []
        for curve in suite["curves"]:
            config = _config_from_curve(curve, initialization, suite)
            print(
                f"Running {initialization}: {config.label} "
                f"({config.n_seeds} seeds, {config.updates} updates)",
                flush=True,
            )
            result, skipped = _run_unit(output, "handoff", config, resume)
            results.append(("handoff", result))
            group.append(result)
            status.append(
                {
                    "experiment": "handoff",
                    "label": config.label,
                    "method": config.method,
                    "initialization": initialization,
                    "n": config.n_trajectories,
                    "beta": config.beta,
                    "beta_after": config.beta_after,
                    "handoff_update": config.handoff_update,
                    "skipped": int(skipped),
                    "finite_fraction": float(result.finite.mean()),
                }
            )
        make_handoff_plots(output / "plots" / initialization, group)

    write_rows(output / "summary.csv", training_summary_rows(results))
    write_rows(output / "paired_differences.csv", paired_difference_rows(results))
    checkpoints = (suite["scale"]["handoff_update"], suite["scale"]["updates"])
    write_rows(
        output / "checkpoint_summary.csv",
        handoff_checkpoint_rows(results, checkpoints),
    )
    write_rows(
        output / "post_handoff_changes.csv",
        handoff_change_rows(results, suite["scale"]["handoff_update"]),
    )
    write_rows(output / "handoff_pairwise.csv", handoff_pairwise_rows(results))
    write_rows(output / "run_status.csv", status)
    print(f"Completed {len(results)} handoff units in {output}", flush=True)
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "full"), default="full")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    output = args.output_dir or (DEFAULT_OUTPUT_ROOT / args.preset)
    try:
        run_handoff_suite(output, preset=args.preset, resume=args.resume)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
