"""Run the complete deterministic Step 3 experiment and create artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np
import torch

from .experiment import ALL_METHODS, ExactTrainingConfig, ExactTrainingResult, magnitude_matched_betas, train_exact
from .model import phi_from_q_and_good
from .reporting import make_basin_plot, make_main_plots, write_summary
from .verify import TOLERANCES, run_verification


DEFAULT_OUTPUT = Path("exploration/results/tabular_mdp/two_step_trap")
INITIALIZATIONS = {
    "uniform": [0.0, 0.0, 0.0, 0.0],
    "adverse": [2.0, -2.0, -2.0, 2.0],
}
BETAS = (0.01, 0.1, 0.2)
GRID = np.asarray((0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9), dtype=np.float64)


def manifest(smoke: bool) -> dict[str, object]:
    updates = 20 if smoke else 2000
    grid = GRID[:2] if smoke else GRID
    return {
        "schema_version": 1, "dtype": "torch.float64", "device": "cpu",
        "sampling": False, "mdp": "two_step_trap", "main_alpha": 0.05,
        "main_updates": updates, "main_betas": list(BETAS),
        "initializations": INITIALIZATIONS, "methods": list(ALL_METHODS),
        "transition_pool_weights": "mu0=1/(1+q), mu1=q/(1+q)",
        "magnitude_target": "norm of reward gradient at adverse initialization",
        "robustness_alphas": [0.025, 0.05, 0.1],
        "robustness_integration_horizon": 1.0 if smoke else 100.0,
        "basin_grid": grid.tolist(), "basin_beta": 0.1,
        "torch_version": torch.__version__, "smoke": smoke,
    }


def prepare_output(output: Path, config: dict[str, object], resume: bool) -> None:
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


def _run_unit(output: Path, experiment: str, config: ExactTrainingConfig, initial, resume: bool) -> tuple[ExactTrainingResult, bool]:
    safe_label = config.label.replace(".", "p")
    path = output / "archives" / f"{experiment}__{safe_label}__{config.method}__a{config.alpha:g}__b{config.beta:g}.npz"
    if path.exists():
        if not resume:
            raise ValueError(f"result already exists: {path}")
        return ExactTrainingResult.load(path, config), True
    result = train_exact(config, initial)
    result.save(path)
    return result, False


def run_suite(output: Path, *, resume: bool = False, smoke: bool = False) -> list[tuple[str, ExactTrainingResult]]:
    suite_manifest = manifest(smoke)
    prepare_output(output, suite_manifest, resume)
    verification = run_verification()
    verification_payload = {
        "schema_version": 1,
        "dtype": "torch.float64",
        "torch_version": torch.__version__,
        "tolerances": TOLERANCES,
        **verification.to_dict(),
    }
    (output / "verification.json").write_text(
        json.dumps(verification_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not verification.passed:
        raise ValueError("exact verification failed; refusing to run experiments")
    updates = 20 if smoke else 2000
    completed: list[tuple[str, ExactTrainingResult]] = []
    status: list[dict[str, object]] = []

    main_by_group: dict[tuple[str, float], list[ExactTrainingResult]] = {}
    for init_name, initial in INITIALIZATIONS.items():
        reward_config = ExactTrainingConfig("reward_only", 0.05, 0.0, updates, init_name)
        reward, skipped = _run_unit(output, "main", reward_config, initial, resume)
        completed.append(("main", reward)); status.append({"experiment": "main", "label": init_name, "method": "reward_only", "skipped": int(skipped), "finite": int(reward.finite.all())})
        for beta in BETAS:
            group = [reward]
            for method in ALL_METHODS[1:]:
                config = ExactTrainingConfig(method, 0.05, beta, updates, f"{init_name}_beta_{beta:g}")
                result, skipped = _run_unit(output, "main", config, initial, resume)
                completed.append(("main", result)); group.append(result)
                status.append({"experiment": "main", "label": config.label, "method": method, "skipped": int(skipped), "finite": int(result.finite.all())})
            main_by_group[(init_name, beta)] = group

    adverse = INITIALIZATIONS["adverse"]
    matched = magnitude_matched_betas(adverse)
    magnitude_results: list[ExactTrainingResult] = []
    reward_config = ExactTrainingConfig("reward_only", 0.05, 0.0, updates, "adverse_magnitude")
    reward, skipped = _run_unit(output, "magnitude", reward_config, adverse, resume)
    completed.append(("magnitude", reward)); magnitude_results.append(reward)
    status.append({"experiment": "magnitude", "label": reward.config.label, "method": reward.config.method, "skipped": int(skipped), "finite": int(reward.finite.all())})
    for method, beta in matched.items():
        config = ExactTrainingConfig(method, 0.05, beta, updates, "adverse_magnitude")
        result, skipped = _run_unit(output, "magnitude", config, adverse, resume)
        completed.append(("magnitude", result)); magnitude_results.append(result)
        status.append({"experiment": "magnitude", "label": config.label, "method": method, "skipped": int(skipped), "finite": int(result.finite.all())})
    (output / "magnitude_matched_betas.json").write_text(json.dumps(matched, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    integration_horizon = 1.0 if smoke else 100.0
    robustness_groups: dict[tuple[str, float], list[ExactTrainingResult]] = {}
    for init_name, initial in INITIALIZATIONS.items():
        for alpha in (0.025, 0.05, 0.1):
            count = round(integration_horizon / alpha)
            group = []
            for method in ALL_METHODS:
                beta = 0.0 if method == "reward_only" else 0.1
                config = ExactTrainingConfig(method, alpha, beta, count, f"{init_name}_alpha_{alpha:g}")
                result, skipped = _run_unit(output, "robustness", config, initial, resume)
                completed.append(("robustness", result)); group.append(result)
                status.append({"experiment": "robustness", "label": config.label, "method": method, "skipped": int(skipped), "finite": int(result.finite.all())})
            robustness_groups[(init_name, alpha)] = group

    grid = GRID[:2] if smoke else GRID
    q_mesh, good_mesh = np.meshgrid(grid, grid, indexing="ij")
    grid_phi = phi_from_q_and_good(q_mesh.reshape(-1), good_mesh.reshape(-1))
    basin_results = []
    for method in ALL_METHODS:
        beta = 0.0 if method == "reward_only" else 0.1
        config = ExactTrainingConfig(method, 0.05, beta, updates, "basin")
        result, skipped = _run_unit(output, "basin", config, grid_phi, resume)
        completed.append(("basin", result)); basin_results.append(result)
        status.append({"experiment": "basin", "label": "basin", "method": method, "skipped": int(skipped), "finite": int(result.finite.all())})

    plots = output / "plots"
    for (init_name, beta), group in main_by_group.items():
        make_main_plots(plots / "main" / f"{init_name}_beta_{str(beta).replace('.', 'p')}", group)
    make_main_plots(plots / "magnitude_controlled", magnitude_results)
    for (init_name, alpha), group in robustness_groups.items():
        make_main_plots(plots / "robustness" / f"{init_name}_alpha_{str(alpha).replace('.', 'p')}", group)
    (plots / "basin").mkdir(parents=True, exist_ok=True)
    for result in basin_results:
        make_basin_plot(plots / "basin" / f"{result.config.method}.png", result, grid)

    write_summary(output / "summary.csv", completed)
    with (output / "run_status.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("experiment", "label", "method", "skipped", "finite"))
        writer.writeheader(); writer.writerows(status)
    print(f"Completed {len(completed)} deterministic units in {output}")
    return completed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    try:
        run_suite(args.output_dir, resume=args.resume, smoke=args.smoke)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr); return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
