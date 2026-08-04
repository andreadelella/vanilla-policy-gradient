"""Run resumable Step 4 estimator audits and sampled training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

from .experiment import (
    METHODS,
    EstimatorAuditConfig,
    EstimatorAuditResult,
    SampledTrainingConfig,
    SampledTrainingResult,
    run_estimator_audit,
    train_sampled,
)
from .reporting import (
    make_audit_plots,
    make_endpoint_heatmaps,
    make_training_group_plots,
    paired_difference_rows,
    training_summary_rows,
    write_audit_csv,
    write_rows,
)
from .verify import TOLERANCES, run_verification


DEFAULT_OUTPUT = Path("exploration/results/tabular_mdp/two_step_trap_sampled")


def manifest(preset: str) -> dict:
    scales = {
        "smoke": {"n_seeds": 4, "updates": 50, "repetitions": 200, "audit_policies": ["uniform", "adverse"], "audit_batches": [4, 32], "main_batches": [4], "betas": [0.1]},
        "pilot": {"n_seeds": 20, "updates": 2000, "repetitions": 5_000, "audit_policies": ["uniform", "adverse", "rare_good", "common_bad", "near_optimal"], "audit_batches": [1, 2, 4, 8, 16, 32, 64, 128, 256], "main_batches": [4, 32, 128], "betas": [0.01, 0.1, 0.2]},
        "full": {"n_seeds": 100, "updates": 2000, "repetitions": 50_000, "audit_policies": ["uniform", "adverse", "rare_good", "common_bad", "near_optimal"], "audit_batches": [1, 2, 4, 8, 16, 32, 64, 128, 256], "main_batches": [4, 32, 128], "betas": [0.01, 0.1, 0.2]},
    }
    if preset not in scales:
        raise ValueError(f"unknown preset: {preset}")
    return {
        "schema_version": 1,
        "stage": "finite_batch_sampled_two_state_mdp",
        "preset": preset,
        "dtype": "torch.float64",
        "device": "cpu",
        "alpha": 0.05,
        "record_interval": 10,
        "base_seed": 23,
        "methods": list(METHODS),
        "initializations": ["uniform", "adverse"],
        "stream_derivation": "one CPU torch.Generator seeded 23 per unit; seed rows share paired base uniforms across methods",
        "primary_returns": "raw reward-to-go, trajectory mean",
        "barrier_pooling": "valid-transition mean with random M=N+K1",
        "fisher": "undamped sampled-action S^T S / M",
        "scales": scales[preset],
        "torch_version": torch.__version__,
    }


def prepare_output(output: Path, config: dict, resume: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "config.json"
    contents = list(output.iterdir())
    if contents:
        if not config_path.exists() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("output directory contains an incompatible configuration")
        if not resume:
            raise ValueError("output directory is non-empty; use --resume")
    else:
        config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe(value) -> str:
    return str(value).replace(".", "p").replace("-", "m")


def _unit_path(output: Path, experiment: str, config: SampledTrainingConfig) -> Path:
    schedule = (
        ""
        if config.handoff_update is None
        else f"__handoff{config.handoff_update}__bafter{_safe(config.beta_after)}"
    )
    return output / "archives" / (
        f"{experiment}__{config.initialization}__n{config.n_trajectories}__{config.method}"
        f"__b{_safe(config.beta)}__noise{_safe(config.reward_noise_std)}"
        f"__c{int(config.center_returns)}n{int(config.normalize_returns)}{schedule}.npz"
    )


def _run_unit(output: Path, experiment: str, config: SampledTrainingConfig, resume: bool) -> tuple[SampledTrainingResult, bool]:
    path = _unit_path(output, experiment, config)
    if path.exists():
        if not resume:
            raise ValueError(f"result already exists: {path}")
        return SampledTrainingResult.load(path, config), True
    result = train_sampled(config)
    result.save(path)
    return result, False


def run_suite(output: Path, *, preset: str, resume: bool = False) -> list[tuple[str, SampledTrainingResult]]:
    suite = manifest(preset)
    prepare_output(output, suite, resume)
    verification = run_verification()
    (output / "verification.json").write_text(
        json.dumps({"schema_version": 1, "tolerances": TOLERANCES, **verification.to_dict()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not verification.passed:
        raise ValueError("sampled estimator verification failed")

    scale = suite["scales"]
    audit_config = EstimatorAuditConfig(
        tuple(scale["audit_policies"]), tuple(scale["audit_batches"]), scale["repetitions"], 23001,
        min(1_000, scale["repetitions"]),
    )
    audit_path = output / "audit.json"
    if audit_path.exists():
        if not resume:
            raise ValueError("audit already exists; use --resume")
        audit = EstimatorAuditResult.load_json(audit_path, audit_config)
    else:
        audit = run_estimator_audit(audit_config)
        audit.save_json(audit_path)
    write_audit_csv(output / "audit.csv", audit)
    make_audit_plots(output / "plots" / "audit", audit)

    results: list[tuple[str, SampledTrainingResult]] = []
    status: list[dict] = []
    groups: dict[tuple, list[SampledTrainingResult]] = {}
    common = dict(n_seeds=scale["n_seeds"], alpha=0.05, updates=scale["updates"], record_interval=10, base_seed=23)
    for initialization in ("uniform", "adverse"):
        for n in scale["main_batches"]:
            reward_config = SampledTrainingConfig("reward_only", initialization, n, beta=0.0, label="main", **common)
            reward, skipped = _run_unit(output, "main", reward_config, resume)
            results.append(("main", reward))
            status.append({"experiment": "main", "method": reward_config.method, "initialization": initialization, "n": n, "beta": 0.0, "skipped": int(skipped), "finite_fraction": float(reward.finite.mean())})
            for beta in scale["betas"]:
                group = [reward]
                for method in METHODS[1:]:
                    config = SampledTrainingConfig(method, initialization, n, beta=beta, label="main", **common)
                    result, skipped = _run_unit(output, "main", config, resume)
                    results.append(("main", result)); group.append(result)
                    status.append({"experiment": "main", "method": method, "initialization": initialization, "n": n, "beta": beta, "skipped": int(skipped), "finite_fraction": float(result.finite.mean())})
                groups[("main", initialization, n, beta)] = group

    secondary_n = 4 if preset == "smoke" else 32
    processed_n = 4 if preset == "smoke" else 128
    for experiment, n, noise, center, normalize in (
        ("gaussian", secondary_n, 1.0, False, False),
        ("processed", processed_n, 0.0, True, True),
    ):
        group = []
        for method in METHODS:
            beta = 0.0 if method == "reward_only" else 0.1
            config = SampledTrainingConfig(
                method, "adverse", n, beta=beta, reward_noise_std=noise,
                center_returns=center, normalize_returns=normalize, label=experiment, **common,
            )
            result, skipped = _run_unit(output, experiment, config, resume)
            results.append((experiment, result)); group.append(result)
            status.append({"experiment": experiment, "method": method, "initialization": "adverse", "n": n, "beta": beta, "skipped": int(skipped), "finite_fraction": float(result.finite.mean())})
        groups[(experiment, "adverse", n, 0.1)] = group

    for (experiment, initialization, n, beta), group in groups.items():
        destination = output / "plots" / "training" / f"{experiment}__{initialization}__n{n}__b{_safe(beta)}"
        make_training_group_plots(destination, group)
    make_endpoint_heatmaps(output / "plots" / "endpoints", results)
    write_rows(output / "summary.csv", training_summary_rows(results))
    write_rows(output / "paired_differences.csv", paired_difference_rows(results))
    write_rows(output / "run_status.csv", status)
    print(f"Completed {len(results)} sampled training units in {output}")
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("smoke", "pilot", "full"), required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_suite(args.output_dir, preset=args.preset, resume=args.resume)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
