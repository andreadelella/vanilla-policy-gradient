"""Summarize a single-seed adaptive Fisher-gradient ratio sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RATIOS = (0.0, 0.05, 0.10, 0.20, 0.50)
COLORS = {
    0.0: "#303030",
    0.05: "#2878b5",
    0.10: "#2a9d5b",
    0.20: "#e1812c",
    0.50: "#c44e52",
}
INVARIANT_CONFIG_KEYS = (
    "seed",
    "env_id",
    "workers",
    "reward_trajectory_count",
    "training_fisher_trajectory_count",
    "horizon",
    "action_repeat",
    "hidden_sizes",
    "learning_rate",
    "gamma",
    "fisher_mu",
    "score_backend",
    "updates",
)


def _label(ratio: float) -> str:
    return f"{ratio * 100:g}%"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float64)
    if values.size >= window:
        result[window - 1 :] = np.convolve(
            values,
            np.ones(window, dtype=np.float64) / window,
            mode="valid",
        )
    return result


def _load(root: Path) -> tuple[dict[float, dict], list[dict], list[dict]]:
    runs: dict[float, dict] = {}
    diagnostics: list[dict] = []
    spectra: list[dict] = []
    reference_config = None
    for run_dir in sorted((root / "runs").glob("ratio_*pct")):
        config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        ratio = float(config["target_fisher_gradient_ratio"])
        if ratio not in RATIOS:
            continue
        result = json.loads((run_dir / "result.json").read_text(encoding="utf-8"))
        rows = _read_jsonl(run_dir / "diagnostics.jsonl")
        if reference_config is None:
            reference_config = config
        else:
            for key in INVARIANT_CONFIG_KEYS:
                if config[key] != reference_config[key]:
                    raise ValueError(
                        f"config mismatch for {key}: "
                        f"{config[key]!r} != {reference_config[key]!r}"
                    )
        runs[ratio] = {
            "run_dir": run_dir,
            "config": config,
            "result": result,
            "diagnostics": rows,
        }
        diagnostics.extend(
            {"target_gradient_ratio": ratio, **row}
            for row in rows
        )
        spectral_path = run_dir / "trajectory_fisher_analysis_4096" / "summary.csv"
        if spectral_path.exists():
            with spectral_path.open(encoding="utf-8") as stream:
                spectra.extend(
                    {
                        "target_gradient_ratio": ratio,
                        **{
                            key: float(value)
                            if key
                            not in {
                                "checkpoint",
                                "fisher_dtype",
                                "score_dtype",
                            }
                            else value
                            for key, value in row.items()
                        },
                    }
                    for row in csv.DictReader(stream)
                )
    missing = [ratio for ratio in RATIOS if ratio not in runs]
    if missing:
        raise ValueError(f"missing completed sweep ratios: {missing}")
    return runs, diagnostics, spectra


def _summary_rows(runs: dict[float, dict], spectra: list[dict]) -> list[dict]:
    rows = []
    for ratio in RATIOS:
        run = runs[ratio]
        diagnostics = run["diagnostics"]
        ratio_spectra = [
            row
            for row in spectra
            if math.isclose(row["target_gradient_ratio"], ratio)
        ]
        endpoint_spectrum = (
            max(ratio_spectra, key=lambda row: row["update"])
            if ratio_spectra
            else {}
        )
        achieved = np.asarray(
            [row["achieved_fisher_gradient_ratio"] for row in diagnostics],
            dtype=np.float64,
        )
        effective_betas = np.asarray(
            [row["effective_fisher_beta"] for row in diagnostics],
            dtype=np.float64,
        )
        result = run["result"]
        rows.append(
            {
                "target_gradient_ratio": ratio,
                "seed": run["config"]["seed"],
                "status": result["status"],
                "completed_updates": result["completed_updates"],
                "first_solved_update": result["first_solved_update"],
                "last_10_mean_return": result["last_10_mean_return"],
                "maximum_training_return": result["maximum_training_return"],
                "final_training_return": result["final_training_return"],
                "maximum_ratio_error": float(np.max(np.abs(achieved - ratio))),
                "median_effective_beta": float(np.median(effective_betas)),
                "minimum_effective_beta": float(np.min(effective_betas)),
                "maximum_effective_beta": float(np.max(effective_betas)),
                "endpoint_minimum_eigenvalue": endpoint_spectrum.get(
                    "minimum_eigenvalue"
                ),
                "endpoint_condition_number": endpoint_spectrum.get(
                    "positive_condition_number"
                ),
                "endpoint_effective_rank": endpoint_spectrum.get("effective_rank"),
            }
        )
    return rows


def _style_axis(axis) -> None:
    axis.grid(alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)


def _plot_returns(runs: dict[float, dict], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    for ratio in RATIOS:
        rows = runs[ratio]["diagnostics"]
        updates = np.asarray([row["update"] for row in rows])
        returns = np.asarray([row["return"] for row in rows])
        color = COLORS[ratio]
        axis.plot(updates, returns, color=color, alpha=0.15, linewidth=0.8)
        axis.plot(
            updates,
            _rolling_mean(returns, 10),
            color=color,
            linewidth=2.2,
            label=_label(ratio),
        )
    axis.axhline(
        90.0,
        color="#666666",
        linestyle=":",
        linewidth=1.8,
        label="solve threshold",
    )
    axis.set(
        title="MountainCarContinuous performance",
        xlabel="Policy update",
        ylabel="Native return",
    )
    axis.legend(
        title="Fisher/reward gradient target",
        frameon=False,
        ncol=2,
    )
    axis.text(
        0.01,
        0.02,
        "Faint: batch return   Solid: 10-update mean",
        transform=axis.transAxes,
        fontsize=9,
        color="#444444",
    )
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _plot_gradient_norms(runs: dict[float, dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), sharex=True)
    metrics = (
        ("reward_gradient_norm", r"Reward gradient $\|g_R\|$"),
        ("fisher_gradient_norm", r"Applied Fisher gradient $\|g_F\|$"),
    )
    for axis, (metric, title) in zip(axes, metrics):
        for ratio in RATIOS:
            rows = runs[ratio]["diagnostics"]
            updates = np.asarray([row["update"] for row in rows])
            values = np.asarray([row[metric] for row in rows])
            if metric == "fisher_gradient_norm" and ratio == 0.0:
                continue
            axis.plot(
                updates,
                _rolling_mean(values, 10),
                color=COLORS[ratio],
                linewidth=2.0,
                label=_label(ratio),
            )
        axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xlabel("Policy update")
        _style_axis(axis)
    axes[0].set_ylabel("Gradient norm, 10-update mean")
    axes[0].legend(
        title="Target ratio",
        frameon=False,
        ncol=2,
    )
    axes[1].text(
        0.02,
        0.04,
        "0% is identically zero and omitted on the log scale.",
        transform=axes[1].transAxes,
        fontsize=9,
        color="#444444",
    )
    figure.suptitle("Applied component-gradient sizes", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _plot_effective_beta(runs: dict[float, dict], output: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.2, 5.2))
    for ratio in RATIOS[1:]:
        rows = runs[ratio]["diagnostics"]
        updates = np.asarray([row["update"] for row in rows])
        beta = np.asarray([row["effective_fisher_beta"] for row in rows])
        axis.plot(
            updates,
            _rolling_mean(beta, 10),
            color=COLORS[ratio],
            linewidth=2.0,
            label=_label(ratio),
        )
    axis.set_yscale("log")
    axis.set(
        title="Adaptive barrier coefficient",
        xlabel="Policy update",
        ylabel=r"Effective $\beta$, 10-update mean",
    )
    axis.legend(title="Target ratio", frameon=False, ncol=2)
    _style_axis(axis)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _plot_training_geometry(runs: dict[float, dict], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.4, 5.2), sharex=True)
    for ratio in RATIOS:
        rows = runs[ratio]["diagnostics"]
        updates = np.asarray([row["update"] for row in rows])
        minimum = np.asarray([row["lambda_min"] for row in rows])
        condition = np.asarray(
            [row["lambda_max"] / row["lambda_min"] for row in rows]
        )
        axes[0].plot(
            updates,
            _rolling_mean(minimum, 10),
            color=COLORS[ratio],
            linewidth=1.8,
            label=_label(ratio),
        )
        axes[1].plot(
            updates,
            _rolling_mean(condition, 10),
            color=COLORS[ratio],
            linewidth=1.8,
        )
    axes[0].set_title("Minimum eigenvalue")
    axes[1].set_title("Condition number")
    axes[0].set_ylabel("10-update mean")
    for axis in axes:
        axis.set_yscale("log")
        axis.set_xlabel("Policy update")
        _style_axis(axis)
    axes[0].legend(title="Target ratio", frameon=False, ncol=2)
    figure.suptitle("Training-batch trajectory Fisher (256 trajectories)", fontsize=15)
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _plot_posthoc_geometry(spectra: list[dict], output: Path) -> None:
    if not spectra:
        return
    figure, axes = plt.subplots(1, 3, figsize=(15.4, 5.0), sharex=True)
    metrics = (
        ("minimum_eigenvalue", "Minimum eigenvalue", True),
        ("positive_condition_number", "Condition number", True),
        ("effective_rank", "Effective rank", False),
    )
    for ratio in RATIOS:
        rows = sorted(
            (
                row
                for row in spectra
                if math.isclose(row["target_gradient_ratio"], ratio)
            ),
            key=lambda row: row["update"],
        )
        updates = [row["update"] for row in rows]
        for axis, (metric, _, _) in zip(axes, metrics):
            axis.plot(
                updates,
                [row[metric] for row in rows],
                color=COLORS[ratio],
                marker="o",
                linewidth=1.8,
                markersize=4,
                label=_label(ratio),
            )
    for axis, (metric, title, logarithmic) in zip(axes, metrics):
        axis.set_title(title)
        axis.set_xlabel("Policy update")
        if metric == "minimum_eigenvalue":
            axis.set_yscale("symlog", linthresh=1e-13, linscale=0.5)
            axis.axhline(0.0, color="#777777", linewidth=0.8)
        elif logarithmic:
            axis.set_yscale("log")
        _style_axis(axis)
    axes[0].legend(title="Target ratio", frameon=False)
    figure.suptitle(
        "Checkpoint trajectory Fisher (4,096 common-random-number trajectories)",
        fontsize=15,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=200)
    plt.close(figure)


def summarize(root: Path) -> None:
    aggregate = root / "aggregate"
    figures = aggregate / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    runs, diagnostics, spectra = _load(root)
    _write_csv(aggregate / "summary.csv", _summary_rows(runs, spectra))
    _write_csv(aggregate / "diagnostics_long.csv", diagnostics)
    _write_csv(aggregate / "spectral_long.csv", spectra)
    _plot_returns(runs, figures / "01_return_vs_update.png")
    _plot_gradient_norms(runs, figures / "02_gradient_norms.png")
    _plot_effective_beta(runs, figures / "03_effective_beta.png")
    _plot_training_geometry(runs, figures / "04_training_fisher_geometry.png")
    _plot_posthoc_geometry(spectra, figures / "05_checkpoint_fisher_geometry.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    summarize(args.root)


if __name__ == "__main__":
    main()
