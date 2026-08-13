"""CSV and figure generation for Acrobot runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_training(rows: list[dict], destination: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        axes[0].plot([row["environment_steps"] for row in subset], [row["mean_batch_return"] for row in subset], label=method)
        barrier_rows = [row for row in subset if "mean_min_probability" in row]
        if barrier_rows:
            axes[1].plot([row["environment_steps"] for row in barrier_rows], [row["mean_min_probability"] for row in barrier_rows], label=method)
    axes[0].set(title="Training return", xlabel="environment steps", ylabel="batch mean return")
    axes[1].set(title="On-policy action geometry", xlabel="environment steps", ylabel="mean minimum probability")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_fisher(rows: list[dict], destination: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    for method in sorted({row["method"] for row in rows}):
        subset = [row for row in rows if row["method"] == method]
        x = [row.get("environment_steps", row["update"]) for row in subset]
        axes[0].plot(x, [row["largest_eigenvalue"] for row in subset], marker="o", label=f"{method}: max")
        axes[0].plot(x, [row["smallest_positive_eigenvalue"] for row in subset], marker=".", linestyle="--", label=f"{method}: min+")
        axes[1].plot(x, [row["trace"] for row in subset], marker="o", label=method)
    axes[0].set_yscale("log")
    axes[0].set(title="Empirical policy-Fisher extremes", xlabel="training progress", ylabel="eigenvalue")
    axes[1].set(title="Empirical policy-Fisher trace", xlabel="training progress", ylabel="trace")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def plot_explained_trace(rows: list[dict], output: Path) -> None:
    """Create the same three spectrum views used by ``fisher_analysis`` plus k90 over time."""

    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    methods = sorted({row["method"] for row in rows})
    final_update = max(int(row["update"]) for row in rows)
    colors = {method: color for method, color in zip(methods, ("#176B87", "#C2410C", "#3F6212"))}

    def eigenvalues(row):
        pairs = []
        for key, value in row.items():
            if key.startswith("eigenvalue_") and value not in (None, ""):
                pairs.append((int(key.split("_")[-1]), float(value)))
        return np.asarray([value for _, value in sorted(pairs)], dtype=np.float64)

    final_rows = [row for row in rows if int(row["update"]) == final_update]
    plot_specs = (
        ("raw_eigenspectrum.png", "Undamped empirical policy-Fisher eigenspectrum", "Eigenvalue", "raw"),
        ("trace_normalized_eigenspectrum.png", "Trace-normalized policy-Fisher eigenspectrum", "Eigenvalue / Fisher trace", "normalized"),
        ("cumulative_explained_trace.png", "Cumulative policy-Fisher trace", "Cumulative explained Fisher trace", "cumulative"),
    )
    for filename, title, ylabel, mode in plot_specs:
        figure, axis = plt.subplots(figsize=(7.2, 4.5))
        for method in methods:
            method_rows = [row for row in final_rows if row["method"] == method]
            curves = []
            for row in method_rows:
                values = eigenvalues(row)
                trace = values.sum()
                if mode == "normalized":
                    values = values / trace if trace > 0.0 else np.zeros_like(values)
                elif mode == "cumulative":
                    values = np.cumsum(values) / trace if trace > 0.0 else np.zeros_like(values)
                curves.append(values)
                axis.plot(np.arange(1, values.size + 1), values, color=colors[method], alpha=0.16, linewidth=0.8)
            common = min((curve.size for curve in curves), default=0)
            if common:
                median = np.median(np.stack([curve[:common] for curve in curves]), axis=0)
                axis.plot(np.arange(1, common + 1), median, color=colors[method], linewidth=2.2, label=f"{method} median (n={len(curves)})")
        if mode != "cumulative":
            axis.set_yscale("log")
        else:
            for threshold in (0.90, 0.95, 0.99):
                axis.axhline(threshold, color="#6B7280", linewidth=0.8, linestyle="--")
            axis.set_ylim(0.0, 1.01)
        axis.set(xlabel="Principal-component index", ylabel=ylabel, title=f"{title} at update {final_update}")
        axis.grid(True, which="both" if mode != "cumulative" else "major", alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(output / filename, dpi=180)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    summary_rows = []
    for method in methods:
        method_rows = [row for row in rows if row["method"] == method]
        updates = sorted({int(row["update"]) for row in method_rows})
        means, lows, highs = [], [], []
        for update in updates:
            values = np.asarray([float(row["k90"]) for row in method_rows if int(row["update"]) == update])
            means.append(float(values.mean()))
            lows.append(float(values.min()))
            highs.append(float(values.max()))
            summary_rows.append({"method": method, "update": update, "seed_count": values.size, "k90_mean": means[-1], "k90_min": lows[-1], "k90_max": highs[-1]})
        axis.plot(updates, means, marker="o", color=colors[method], label=method)
        axis.fill_between(updates, lows, highs, color=colors[method], alpha=0.16, label=f"{method} seed range")
    axis.set(xlabel="Optimizer update", ylabel="Components required for 90% trace", title="Effective Fisher dimension through training")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output / "components_90_through_training.png", dpi=180)
    plt.close(figure)

    import csv
    with (output / "components_90_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)


def plot_checkpoint_behavior(rows: list[dict], output: Path) -> None:
    """Plot evaluation return and categorical-policy diagnostics."""

    import matplotlib.pyplot as plt

    methods = sorted({row["method"] for row in rows})
    colors = {method: color for method, color in zip(methods, ("#176B87", "#C2410C", "#3F6212"))}
    figure, axes = plt.subplots(2, 2, figsize=(11, 7.5))
    fields = (
        ("stochastic_return", "Stochastic evaluation return"),
        ("deterministic_return", "Deterministic evaluation return"),
        ("entropy", "Mean action entropy"),
        ("mean_min_probability", "Mean minimum action probability"),
    )
    for axis, (field, title) in zip(axes.flat, fields):
        for method in methods:
            subset = [row for row in rows if row["method"] == method]
            updates = sorted({row["update"] for row in subset})
            values = [np.asarray([row[field] for row in subset if row["update"] == update]) for update in updates]
            means = np.asarray([value.mean() for value in values])
            lows = np.asarray([value.min() for value in values])
            highs = np.asarray([value.max() for value in values])
            axis.plot(updates, means, marker="o", color=colors[method], label=method)
            axis.fill_between(updates, lows, highs, color=colors[method], alpha=0.15)
        axis.set(title=title, xlabel="Optimizer update")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    figure.suptitle("Acrobot: reward only versus fixed categorical log barrier")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)
