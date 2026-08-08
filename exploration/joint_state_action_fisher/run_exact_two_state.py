"""Run Step 2 exact Euclidean-gradient joint-Fisher comparisons."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import subprocess
import sys

from .exact_two_state import (
    ALL_METHODS,
    build_run_configs,
    magnitude_matched_betas,
    train_exact,
    vector_field_rows,
)
from .reporting import (
    compact_summary,
    create_all_plots,
    endpoint_rows,
    gradient_rows,
    write_csv,
)
from .verify_identity import run_verification


SCHEMA_VERSION = 1
DEFAULT_OUTPUT = Path("exploration/results/joint_state_action_fisher/step2_two_state")


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False, timeout=10
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def experiment_config(*, smoke: bool) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": "exact_two_state_joint_fisher",
        "scientific_role": "exact_mechanism",
        "dtype": "torch.float64",
        "device": "cpu",
        "gradient_type": "exact_euclidean",
        "sampling": False,
        "alpha": 0.05,
        "updates": 40 if smoke else 2000,
        "same_beta_values": [0.1] if smoke else [0.01, 0.1, 0.2],
        "initializations": {
            "uniform": [0.0, 0.0, 0.0, 0.0],
            "adverse": [2.0, -2.0, -2.0, 2.0],
        },
        "methods": list(ALL_METHODS),
        "magnitude_matching": {"initialization": "adverse", "kappa": 1.0},
        "vector_field": {"beta": 0.1, "grid_size": 7 if smoke else 17},
        "state_weighting_convention": "transition_pooled_population",
        "normalization": "one_half_logdet",
        "smoke": smoke,
    }


def _prepare_output(output_dir: Path, config: dict[str, object], resume: bool) -> bool:
    if output_dir.exists() and any(output_dir.iterdir()):
        config_path = output_dir / "config.json"
        manifest_path = output_dir / "manifest.json"
        if resume and config_path.exists():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing != config:
                raise ValueError("incompatible configuration in existing output directory")
            if manifest_path.exists():
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("status") == "complete":
                    return True
            raise ValueError("compatible output is incomplete; exact runs are not silently overwritten")
        raise ValueError("output directory is nonempty; use --resume only for an identical completed run")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(
        json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return False


def _mechanism_summary(vector_rows, results) -> dict[str, object]:
    unique_geometry = [row for row in vector_rows if row["method"] == "reward_only"]
    cosines = [float(row["cosine_pooled_joint_regularizer"]) for row in unique_geometry]
    correction_dq = [float(row["joint_correction_dq_dt"]) for row in unique_geometry]
    pooled_dq = [float(row["pooled_regularizer_dq_dt"]) for row in unique_geometry]
    joint_dq = [float(row["joint_regularizer_dq_dt"]) for row in unique_geometry]
    contrasts = []
    lookup = {
        (result.config.protocol, result.config.initialization, result.config.beta, result.config.method): result
        for result in results
    }
    for (protocol, initialization, beta, method), joint in lookup.items():
        if method != "joint_state_action_logdet":
            continue
        pooled = lookup.get((protocol, initialization, beta, "pooled_policy_logdet"))
        if pooled is None:
            continue
        contrasts.append(
            {
                "protocol": protocol,
                "initialization": initialization,
                "beta": beta,
                "joint_minus_pooled_final_return": float(joint.endpoint["return"]) - float(pooled.endpoint["return"]),
                "joint_minus_pooled_final_q": float(joint.endpoint["q"]) - float(pooled.endpoint["q"]),
                "joint_minus_pooled_final_p_good": float(joint.endpoint["p_good"]) - float(pooled.endpoint["p_good"]),
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "h2_nonparallel_direction": {
            "minimum_grid_cosine": min(cosines),
            "maximum_grid_cosine": max(cosines),
            "fraction_cosine_below_0p999999": sum(value < 0.999999 for value in cosines) / len(cosines),
        },
        "h3_joint_correction": {
            "maximum_correction_dq": max(correction_dq),
            "minimum_correction_dq": min(correction_dq),
            "all_grid_corrections_reduce_q": all(value < 0.0 for value in correction_dq),
            "maximum_joint_minus_pooled_dq": max(joint - pooled for joint, pooled in zip(joint_dq, pooled_dq)),
        },
        "endpoint_contrasts": contrasts,
        "handoff_status": "deferred_until_fixed_objectives_are_interpreted",
    }


def run_experiment(output_dir: Path, *, smoke: bool, resume: bool) -> None:
    verification = run_verification()
    if not verification.passed:
        raise RuntimeError("Gate A failed: exact identity verification did not pass")
    config = experiment_config(smoke=smoke)
    if _prepare_output(output_dir, config, resume):
        print(f"Completed compatible run already exists: {output_dir}")
        return

    configs = build_run_configs(smoke=smoke)
    results = []
    for index, run_config in enumerate(configs, start=1):
        print(f"[{index:02d}/{len(configs):02d}] {run_config.run_id}")
        result = train_exact(run_config)
        if not result.finite:
            raise RuntimeError(f"non-finite exact run: {run_config.run_id}")
        results.append(result)
    results_tuple = tuple(results)
    checkpoints = tuple(row for result in results_tuple for row in result.checkpoints)
    vector_rows = vector_field_rows(grid_size=7 if smoke else 17)

    write_csv(output_dir / "checkpoints.csv", checkpoints)
    write_csv(output_dir / "endpoints.csv", endpoint_rows(results_tuple))
    write_csv(output_dir / "gradient_decomposition.csv", gradient_rows(results_tuple))
    write_csv(output_dir / "summary.csv", compact_summary(results_tuple))
    write_csv(output_dir / "vector_field.csv", vector_rows)
    coefficients = magnitude_matched_betas()
    (output_dir / "magnitude_matched_betas.json").write_text(
        json.dumps({key: value for key, value in coefficients.items()}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    mechanism = _mechanism_summary(vector_rows, results_tuple)
    (output_dir / "mechanism_summary.json").write_text(
        json.dumps(mechanism, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    create_all_plots(results_tuple, vector_rows, output_dir / "plots")

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scientific_role": "exact_mechanism",
        "status": "complete",
        "git_commit": _git_commit(),
        "python_version": platform.python_version(),
        "torch_version": __import__("torch").__version__,
        "dtype": "torch.float64",
        "device": "cpu",
        "exact_identity_gate_passed": True,
        "run_count": len(results_tuple),
        "checkpoint_count": len(checkpoints),
        "handoff_included": False,
        "artifacts": [
            "config.json",
            "checkpoints.csv",
            "endpoints.csv",
            "gradient_decomposition.csv",
            "summary.csv",
            "vector_field.csv",
            "magnitude_matched_betas.json",
            "mechanism_summary.json",
            "plots/",
            "manifest.json",
        ],
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Completed {len(results_tuple)} exact runs: {output_dir}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_experiment(args.output_dir, smoke=args.smoke, resume=args.resume)
    except (ValueError, RuntimeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
