"""Archive-only focused post-hoc analysis of the handoff experiment."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vpg.stats import mean_confidence_interval

from .posthoc import (
    FISHER_NAMES,
    StoredTrainingArchive,
    exact_policy_metrics,
    exact_reward_continuation,
    load_training_archive,
    policy_hash,
    sha256_file,
    tensor_metrics_to_numpy,
)


SOURCE_ROOT = Path(
    "exploration/results/tabular_mdp/two_step_trap_sampled/handoff/robustness/archives"
)
REFERENCE_ROOT = Path(
    "exploration/results/tabular_mdp/two_step_trap_sampled/handoff/full/archives"
)
DEFAULT_OUTPUT = Path(
    "exploration/results/tabular_mdp/two_step_trap_sampled/handoff_posthoc"
)
SWITCH_TIMES = (500, 1000, 1500, 2000, 2500)
FOCUS_SWITCH_TIMES = (500, 1000, 1500)
INITIALIZATIONS = ("uniform", "adverse")
CHECKPOINT_OFFSETS = (0, 50, 100, 250, 500)
FINAL_UPDATE = 4000


def _safe_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_rows(path: Path, rows: Iterable[dict]) -> int:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not materialized:
        path.write_text("", encoding="utf-8")
        return 0
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(materialized)
    return len(materialized)


def _json_write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def discover_switch_archives(source_root: Path = SOURCE_ROOT) -> dict[tuple[str, int], StoredTrainingArchive]:
    result: dict[tuple[str, int], StoredTrainingArchive] = {}
    for path in sorted(source_root.glob("*.npz")):
        archive = load_training_archive(path)
        config = archive.config
        if (
            config.get("method") != "detached_conditional_sampled"
            or config.get("n_trajectories") != 32
            or config.get("n_seeds") != 100
            or config.get("updates") != 4000
            or config.get("beta") != 0.2
            or config.get("beta_after") != 0.0
        ):
            continue
        key = (config["initialization"], int(config["handoff_update"]))
        if key in result:
            raise ValueError(f"duplicate switch archive for {key}")
        result[key] = archive
    expected = {(initialization, switch) for initialization in INITIALIZATIONS for switch in SWITCH_TIMES}
    if set(result) != expected:
        raise ValueError(f"switch archives differ from required grid: missing={sorted(expected-set(result))}")
    return result


def discover_reference_archives(
    reference_root: Path = REFERENCE_ROOT,
) -> dict[tuple[str, str], StoredTrainingArchive]:
    result: dict[tuple[str, str], StoredTrainingArchive] = {}
    labels = {
        ("reward_only", None, None): "reward_only",
        ("detached_conditional_sampled", None, None): "sampled_conditional_fixed",
    }
    for path in sorted(reference_root.glob("*.npz")):
        archive = load_training_archive(path)
        config = archive.config
        if config.get("initialization") not in INITIALIZATIONS or config.get("n_trajectories") != 32:
            continue
        if config.get("updates") != 4000 or config.get("n_seeds") != 100:
            continue
        method = config.get("method")
        if method == "reward_only" and config.get("beta") == 0.0:
            label = "reward_only"
        elif (
            method == "detached_conditional_sampled"
            and config.get("beta") == 0.2
            and config.get("handoff_update") is None
        ):
            label = "sampled_conditional_fixed"
        else:
            continue
        key = (config["initialization"], label)
        if key in result:
            raise ValueError(f"duplicate reference archive for {key}")
        result[key] = archive
    expected = {(initialization, label) for initialization in INITIALIZATIONS for label in ("reward_only", "sampled_conditional_fixed")}
    if set(result) != expected:
        raise ValueError(f"reference archives differ from required set: missing={sorted(expected-set(result))}")
    return result


def _source_snapshot(archives: Iterable[StoredTrainingArchive]) -> dict[str, dict]:
    return {
        _safe_relative(archive.path): {
            "sha256": archive.sha256,
            "size": archive.size,
            "mtime_ns": archive.mtime_ns,
        }
        for archive in archives
    }


def _verify_sources_unchanged(snapshot: dict[str, dict]) -> None:
    for source, before in snapshot.items():
        path = Path(source)
        stat = path.stat()
        if stat.st_size != before["size"] or stat.st_mtime_ns != before["mtime_ns"]:
            raise ValueError(f"source archive metadata changed during analysis: {source}")
        if sha256_file(path) != before["sha256"]:
            raise ValueError(f"source archive content changed during analysis: {source}")


def _scalar(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return float(value)


def _exact_rows_at_checkpoint(
    archive: StoredTrainingArchive,
    *,
    step: int,
    final_success: np.ndarray,
    checkpoint_name: str,
) -> list[dict]:
    index = archive.index_for_step(step)
    phi = archive.phi[index]
    exact = tensor_metrics_to_numpy(exact_policy_metrics(phi))
    rows: list[dict] = []
    source = _safe_relative(archive.path)
    run_id = archive.path.stem
    for seed in range(archive.config["n_seeds"]):
        row = {
            "run_id": run_id,
            "seed": seed,
            "seed_stream_id": f"{archive.config['base_seed']}:{seed}",
            "base_seed": archive.config["base_seed"],
            "initialization": archive.config["initialization"],
            "switch_time": archive.config["handoff_update"],
            "checkpoint": checkpoint_name,
            "update": step,
            "source_archive": source,
            "source_archive_sha256": archive.sha256,
            "policy_hash": policy_hash(phi[seed]),
            "final_success": bool(final_success[seed]),
            "coordinate_convention": "(s0_a0,s0_a1,s1_a0,s1_a1); a2 reference",
        }
        row.update({key: _scalar(value[seed]) for key, value in exact.items()})
        for key in (
            "k1",
            "m",
            "zero_s1",
            "empirical_fisher_rank",
            "empirical_fisher_min_eigenvalue",
            "empirical_fisher_logdet",
            "empirical_fisher_logdet_defined",
        ):
            row[key] = _scalar(archive.metrics[key][index, seed])
        row["sampled_s1_count"] = row["k1"]
        row["empirical_full_eigenvalues_available"] = False
        rows.append(row)
    return rows


def build_switch_and_trajectory_rows(
    archives: dict[tuple[str, int], StoredTrainingArchive],
) -> tuple[list[dict], list[dict]]:
    switch_rows: list[dict] = []
    trajectory_rows: list[dict] = []
    for initialization in INITIALIZATIONS:
        for switch in SWITCH_TIMES:
            archive = archives[(initialization, switch)]
            final_index = archive.index_for_step(FINAL_UPDATE)
            final_success = (
                (archive.metrics["q"][final_index] >= 0.9)
                & (archive.metrics["pi1_a0"][final_index] >= 0.9)
                & archive.finite
            )
            checkpoints = [("switch", switch)]
            checkpoints.extend((f"switch_plus_{offset}", switch + offset) for offset in CHECKPOINT_OFFSETS[1:])
            checkpoints.append(("final", FINAL_UPDATE))
            for name, step in checkpoints:
                rows = _exact_rows_at_checkpoint(
                    archive,
                    step=step,
                    final_success=final_success,
                    checkpoint_name=name,
                )
                trajectory_rows.extend(rows)
                if name == "switch":
                    switch_rows.extend(rows)
    return switch_rows, trajectory_rows


def _ci(values: np.ndarray) -> tuple[float, float, float]:
    mean, lower, upper = mean_confidence_interval(np.asarray(values, dtype=np.float64))
    return float(mean), float(lower), float(upper)


def _summary_metric_names() -> list[str]:
    metrics = [
        "population_return", "q", "p_good", "v1", "delta_safe",
        "d_explore_j", "explore_logit_reward_gradient", "reward_gradient_norm",
        "beta_conditional_gradient_norm", "reward_to_barrier_norm_ratio",
        "reward_barrier_cosine", "min_pi0", "min_pi1", "entropy0", "entropy1",
        "empirical_fisher_rank", "sampled_s1_count",
    ]
    dimensions = {"f0": 2, "f1": 2, "f_pool": 4, "f_ref": 4}
    for name in FISHER_NAMES:
        metrics.extend(
            [
                f"{name}_lambda_min", f"{name}_lambda_max", f"{name}_trace",
                f"{name}_logdet", f"{name}_condition",
            ]
        )
        metrics.extend(f"{name}_eigenvalue_{index}" for index in range(1, dimensions[name] + 1))
    return metrics


def summarize_switch_rows(rows: list[dict]) -> list[dict]:
    result: list[dict] = []
    for initialization in INITIALIZATIONS:
        for switch in SWITCH_TIMES:
            selected = [row for row in rows if row["initialization"] == initialization and row["switch_time"] == switch]
            for metric in _summary_metric_names():
                values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                finite = values[np.isfinite(values)]
                mean, lower, upper = _ci(finite)
                result.append(
                    {
                        "initialization": initialization,
                        "switch_time": switch,
                        "metric": metric,
                        "mean": mean,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "median": float(np.median(finite)),
                        "p05": float(np.quantile(finite, 0.05)),
                        "p95": float(np.quantile(finite, 0.95)),
                        "minimum": float(finite.min()),
                        "maximum": float(finite.max()),
                        "n": int(len(finite)),
                    }
                )
    return result


def summarize_endpoints(trajectory_rows: list[dict]) -> list[dict]:
    result: list[dict] = []
    for initialization in INITIALIZATIONS:
        for switch in SWITCH_TIMES:
            selected = [
                row for row in trajectory_rows
                if row["initialization"] == initialization
                and row["switch_time"] == switch
                and row["checkpoint"] == "final"
            ]
            for metric in ("population_return", "q", "p_good", "final_success"):
                values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                mean, lower, upper = _ci(values)
                result.append(
                    {
                        "initialization": initialization,
                        "switch_time": switch,
                        "metric": metric,
                        "mean": mean,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "median": float(np.median(values)),
                        "p05": float(np.quantile(values, 0.05)),
                        "p95": float(np.quantile(values, 0.95)),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                        "n": len(values),
                    }
                )
    return result


def range_overlap_rows(rows: list[dict]) -> list[dict]:
    result: list[dict] = []
    for initialization in INITIALIZATIONS:
        by_switch = {
            switch: [row for row in rows if row["initialization"] == initialization and row["switch_time"] == switch]
            for switch in FOCUS_SWITCH_TIMES
        }
        for right_switch in (1000, 1500):
            for metric in _summary_metric_names():
                left = np.asarray([row[metric] for row in by_switch[500]], dtype=np.float64)
                right = np.asarray([row[metric] for row in by_switch[right_switch]], dtype=np.float64)
                left, right = left[np.isfinite(left)], right[np.isfinite(right)]
                full_overlap = max(left.min(), right.min()) <= min(left.max(), right.max())
                left05, left95 = np.quantile(left, (0.05, 0.95))
                right05, right95 = np.quantile(right, (0.05, 0.95))
                central_overlap = max(left05, right05) <= min(left95, right95)
                result.append(
                    {
                        "initialization": initialization,
                        "metric": metric,
                        "left_switch": 500,
                        "right_switch": right_switch,
                        "full_ranges_overlap": bool(full_overlap),
                        "central_5_95_ranges_overlap": bool(central_overlap),
                        "left_min": float(left.min()),
                        "left_max": float(left.max()),
                        "right_min": float(right.min()),
                        "right_max": float(right.max()),
                        "left_p05": float(left05),
                        "left_p95": float(left95),
                        "right_p05": float(right05),
                        "right_p95": float(right95),
                    }
                )
    return result


def _curve_summary_rows(
    *,
    initialization: str,
    switch: int,
    kind: str,
    steps: np.ndarray,
    phi: np.ndarray,
) -> list[dict]:
    shape = phi.shape
    exact = tensor_metrics_to_numpy(exact_policy_metrics(phi.reshape(-1, 4)))
    rows: list[dict] = []
    for metric in ("population_return", "q", "p_good"):
        values = exact[metric].reshape(shape[0], shape[1])
        for index, step in enumerate(steps):
            mean, lower, upper = _ci(values[index])
            rows.append(
                {
                    "initialization": initialization,
                    "switch_time": switch,
                    "continuation": kind,
                    "update": int(step),
                    "metric": metric,
                    "mean": mean,
                    "ci_lower": lower,
                    "ci_upper": upper,
                }
            )
    return rows


def build_counterfactual_rows(
    archives: dict[tuple[str, int], StoredTrainingArchive],
) -> tuple[list[dict], list[dict], list[dict], dict[tuple[str, int], dict]]:
    final_rows: list[dict] = []
    curve_rows: list[dict] = []
    continuation_data: dict[tuple[str, int], dict] = {}
    for initialization in INITIALIZATIONS:
        for switch in FOCUS_SWITCH_TIMES:
            archive = archives[(initialization, switch)]
            switch_index = archive.index_for_step(switch)
            actual_steps = archive.steps[switch_index:]
            actual_phi = archive.phi[switch_index:]
            exact_steps, exact_phi = exact_reward_continuation(
                archive.phi[switch_index], start_update=switch, final_update=FINAL_UPDATE,
                alpha=0.05, record_interval=10,
            )
            if not np.array_equal(actual_steps, exact_steps):
                raise ValueError("sampled and exact continuation grids differ")
            actual_final = tensor_metrics_to_numpy(exact_policy_metrics(actual_phi[-1]))
            exact_final = tensor_metrics_to_numpy(exact_policy_metrics(exact_phi[-1]))
            source = _safe_relative(archive.path)
            for seed in range(archive.config["n_seeds"]):
                actual_success = bool(actual_final["q"][seed] >= 0.9 and actual_final["p_good"][seed] >= 0.9)
                exact_success = bool(exact_final["q"][seed] >= 0.9 and exact_final["p_good"][seed] >= 0.9)
                final_rows.append(
                    {
                        "run_id": archive.path.stem,
                        "seed": seed,
                        "initialization": initialization,
                        "switch_time": switch,
                        "source_archive": source,
                        "switch_policy_hash": policy_hash(archive.phi[switch_index, seed]),
                        "sampled_final_return": float(actual_final["population_return"][seed]),
                        "exact_final_return": float(exact_final["population_return"][seed]),
                        "sampled_final_q": float(actual_final["q"][seed]),
                        "exact_final_q": float(exact_final["q"][seed]),
                        "sampled_final_p_good": float(actual_final["p_good"][seed]),
                        "exact_final_p_good": float(exact_final["p_good"][seed]),
                        "sampled_success": actual_success,
                        "exact_success": exact_success,
                        "success_agreement": actual_success == exact_success,
                    }
                )
            curve_rows.extend(
                _curve_summary_rows(
                    initialization=initialization, switch=switch, kind="sampled",
                    steps=actual_steps, phi=actual_phi,
                )
            )
            curve_rows.extend(
                _curve_summary_rows(
                    initialization=initialization, switch=switch, kind="exact",
                    steps=exact_steps, phi=exact_phi,
                )
            )
            continuation_data[(initialization, switch)] = {
                "steps": actual_steps,
                "actual_phi": actual_phi,
                "exact_phi": exact_phi,
            }

    summary_rows: list[dict] = []
    numeric_metrics = (
        "sampled_final_return", "exact_final_return", "sampled_final_q", "exact_final_q",
        "sampled_final_p_good", "exact_final_p_good", "sampled_success", "exact_success",
        "success_agreement",
    )
    for initialization in INITIALIZATIONS:
        for switch in FOCUS_SWITCH_TIMES:
            selected = [row for row in final_rows if row["initialization"] == initialization and row["switch_time"] == switch]
            for metric in numeric_metrics:
                values = np.asarray([row[metric] for row in selected], dtype=np.float64)
                mean, lower, upper = _ci(values)
                summary_rows.append(
                    {
                        "initialization": initialization,
                        "switch_time": switch,
                        "metric": metric,
                        "mean": mean,
                        "ci_lower": lower,
                        "ci_upper": upper,
                        "median": float(np.median(values)),
                        "p05": float(np.quantile(values, 0.05)),
                        "p95": float(np.quantile(values, 0.95)),
                        "minimum": float(values.min()),
                        "maximum": float(values.max()),
                        "n": len(values),
                    }
                )
    return final_rows, summary_rows, curve_rows, continuation_data


def _switch_group(rows: list[dict], initialization: str, switch: int) -> list[dict]:
    return [row for row in rows if row["initialization"] == initialization and row["switch_time"] == switch]


def _plot_switch_lines(axes, rows: list[dict], metrics: list[tuple[str, str]]) -> None:
    axes = list(axes)
    for initialization in INITIALIZATIONS:
        for ax, (metric, title) in zip(axes, metrics):
            means, lows, highs = [], [], []
            for switch in SWITCH_TIMES:
                values = np.asarray([row[metric] for row in _switch_group(rows, initialization, switch)], dtype=np.float64)
                mean, low, high = _ci(values)
                means.append(mean); lows.append(low); highs.append(high)
            ax.plot(SWITCH_TIMES, means, marker="o", label=initialization)
            ax.fill_between(SWITCH_TIMES, lows, highs, alpha=0.12)
            ax.set_title(title)
            ax.set_xlabel("switch update")
            ax.grid(alpha=0.25)


def figure_a(output: Path, rows: list[dict]) -> None:
    metrics = [
        ("population_return", "$J$ at switch"), ("q", "$q$ at switch"),
        ("p_good", "$\\pi_1(a_0)$ at switch"), ("v1", "$V_1$"),
        ("delta_safe", "$\\Delta_{safe}=V_1-0.5$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7)); flat = axes.flat
    _plot_switch_lines(flat, rows, metrics)
    flat[4].axhline(0.0, color="black", linestyle="--", linewidth=1)
    flat[0].legend()
    flat[5].axis("off")
    fig.tight_layout(); fig.savefig(output / "figure_a_behavior_at_switch.png", dpi=180); plt.close(fig)


def figure_b(output: Path, rows: list[dict]) -> None:
    metrics = [
        ("explore_logit_reward_gradient", "Explore-logit reward component"),
        ("d_explore_j", "$D_{explore}J$"),
        ("reward_gradient_norm", "$||\\nabla J||_2$"),
        ("reward_barrier_cosine", "$\\cos(\\nabla J,g_{cond})$"),
        ("reward_to_barrier_norm_ratio", "$||\\nabla J||/||\\beta g_{cond}||$"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7)); flat = axes.flat
    _plot_switch_lines(flat, rows, metrics)
    flat[0].legend(); flat[5].axis("off")
    fig.tight_layout(); fig.savefig(output / "figure_b_reward_readiness.png", dpi=180); plt.close(fig)


def figure_c(output: Path, rows: list[dict]) -> None:
    dimensions = {"f0": 2, "f1": 2, "f_pool": 4, "f_ref": 4}
    fig, axes = plt.subplots(4, 3, figsize=(11, 11), sharey="row")
    for row_index, name in enumerate(FISHER_NAMES):
        for col_index, switch in enumerate(FOCUS_SWITCH_TIMES):
            selected = _switch_group(rows, "adverse", switch)
            means, lows, highs = [], [], []
            for eig_index in range(1, dimensions[name] + 1):
                values = np.asarray([row[f"{name}_eigenvalue_{eig_index}"] for row in selected])
                mean, low, high = _ci(values); means.append(mean); lows.append(low); highs.append(high)
            x = np.arange(1, dimensions[name] + 1)
            ax = axes[row_index, col_index]
            ax.plot(x, means, marker="o"); ax.fill_between(x, lows, highs, alpha=0.18)
            ax.set_yscale("log"); ax.grid(alpha=0.25); ax.set_xticks(x)
            if row_index == 0: ax.set_title(f"switch {switch}")
            if col_index == 0: ax.set_ylabel(f"{name} eigenvalue")
            if row_index == 3: ax.set_xlabel("descending index")
    fig.tight_layout(); fig.savefig(output / "figure_c_exact_fisher_spectra.png", dpi=180); plt.close(fig)


def _trajectory_arrays(archive: StoredTrainingArchive, start: int) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    index = archive.index_for_step(start)
    phi = archive.phi[index:]
    shape = phi.shape
    exact = tensor_metrics_to_numpy(exact_policy_metrics(phi.reshape(-1, 4)))
    reshaped = {key: value.reshape(shape[0], shape[1]) for key, value in exact.items()}
    reshaped["empirical_fisher_rank"] = archive.metrics["empirical_fisher_rank"][index:]
    return archive.steps[index:], reshaped


def figure_d(
    output: Path,
    archives: dict[tuple[str, int], StoredTrainingArchive],
    references: dict[tuple[str, str], StoredTrainingArchive],
) -> None:
    metrics = [
        ("population_return", "Return"), ("q", "$q$"), ("p_good", "$\\pi_1(a_0)$"),
        ("v1", "$V_1$"), ("delta_safe", "$\\Delta_{safe}$"),
        ("f1_lambda_min", "$\\lambda_{min}(F_1)$"),
        ("f_pool_lambda_min", "$\\lambda_{min}(F_{pool})$"),
        ("f_ref_lambda_min", "$\\lambda_{min}(F_{ref})$"),
        ("empirical_fisher_rank", "Empirical Fisher rank"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(14, 11))
    for switch in FOCUS_SWITCH_TIMES:
        steps, values = _trajectory_arrays(archives[("adverse", switch)], switch)
        for ax, (metric, title) in zip(axes.flat, metrics):
            mean = np.nanmean(values[metric], axis=1)
            ax.plot(steps, mean, label=f"switch {switch}")
            ax.axvline(switch, color="grey", linewidth=0.6, alpha=0.35)
            ax.set_title(title); ax.set_xlabel("update"); ax.grid(alpha=0.2)
    for label, linestyle in (("reward_only", ":"), ("sampled_conditional_fixed", "--")):
        archive = references[("adverse", label)]
        steps, values = _trajectory_arrays(archive, 500)
        for ax, (metric, _) in zip(axes.flat[:5], metrics[:5]):
            ax.plot(steps, np.nanmean(values[metric], axis=1), color="black", linestyle=linestyle, linewidth=1, label=label.replace("_", " "))
    axes.flat[0].legend(fontsize=7)
    axes.flat[4].axhline(0.0, color="black", linewidth=0.8)
    fig.tight_layout(); fig.savefig(output / "figure_d_post_handoff_trajectories.png", dpi=180); plt.close(fig)


def figure_e(output: Path, continuation_data: dict[tuple[str, int], dict]) -> None:
    metrics = (("population_return", "Return"), ("q", "$q$"), ("p_good", "$\\pi_1(a_0)$"))
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), sharex="col")
    for col, switch in enumerate(FOCUS_SWITCH_TIMES):
        data = continuation_data[("adverse", switch)]
        steps = data["steps"]
        for kind, phi, linestyle in (("sampled", data["actual_phi"], "-"), ("exact", data["exact_phi"], "--")):
            shape = phi.shape
            exact = tensor_metrics_to_numpy(exact_policy_metrics(phi.reshape(-1, 4)))
            for row, (metric, title) in enumerate(metrics):
                values = exact[metric].reshape(shape[0], shape[1])
                axes[row, col].plot(steps, np.mean(values, axis=1), linestyle=linestyle, label=kind)
                axes[row, col].grid(alpha=0.2)
                if col == 0: axes[row, col].set_ylabel(title)
                if row == 0: axes[row, col].set_title(f"switch {switch}")
                if row == 2: axes[row, col].set_xlabel("update")
    axes[0, 0].legend()
    fig.tight_layout(); fig.savefig(output / "figure_e_sampled_vs_exact.png", dpi=180); plt.close(fig)


def figure_f(output: Path, rows: list[dict]) -> None:
    selected = [row for row in rows if row["initialization"] == "adverse" and row["switch_time"] in FOCUS_SWITCH_TIMES]
    specs = (
        ("delta_safe", "f_pool_lambda_min", "$\\Delta_{safe}$", "$\\lambda_{min}(F_{pool})$"),
        ("d_explore_j", "f_pool_lambda_min", "$D_{explore}J$", "$\\lambda_{min}(F_{pool})$"),
        ("delta_safe", "f_pool_logdet", "$\\Delta_{safe}$", "$\\log\\det F_{pool}$"),
        ("delta_safe", "f_ref_lambda_min", "$\\Delta_{safe}$", "$\\lambda_{min}(F_{ref})$"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (x_key, y_key, x_label, y_label) in zip(axes.flat, specs):
        for success, color, label in ((False, "tab:red", "final failure"), (True, "tab:blue", "final success")):
            subset = [row for row in selected if row["final_success"] is success]
            ax.scatter([row[x_key] for row in subset], [row[y_key] for row in subset], s=16, alpha=0.55, color=color, label=label)
        ax.set_xlabel(x_label); ax.set_ylabel(y_label); ax.grid(alpha=0.2)
    axes.flat[0].legend()
    fig.tight_layout(); fig.savefig(output / "figure_f_behavior_vs_geometry.png", dpi=180); plt.close(fig)


def _prepare_output(output: Path, config: dict, resume: bool) -> None:
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "posthoc_config.json"
    contents = list(output.iterdir())
    if contents:
        if not resume:
            raise ValueError("post-hoc output is non-empty; use --resume")
        if not config_path.exists() or json.loads(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("post-hoc output contains an incompatible configuration")
    else:
        _json_write(config_path, config)


def run_posthoc(output: Path = DEFAULT_OUTPUT, *, resume: bool = False) -> dict:
    switch_archives = discover_switch_archives()
    reference_archives = discover_reference_archives()
    all_sources = list(switch_archives.values()) + list(reference_archives.values())
    source_snapshot = _source_snapshot(all_sources)
    config = {
        "schema_version": 1,
        "analysis": "focused_handoff_posthoc",
        "training_rerun": False,
        "switch_times": list(SWITCH_TIMES),
        "focus_switch_times": list(FOCUS_SWITCH_TIMES),
        "initializations": list(INITIALIZATIONS),
        "checkpoint_offsets": list(CHECKPOINT_OFFSETS),
        "final_update": FINAL_UPDATE,
        "alpha": 0.05,
        "beta_before": 0.2,
        "beta_after": 0.0,
        "n_trajectories": 32,
        "n_seeds": 100,
        "success_criterion": "q>=0.9 and pi1(a0)>=0.9",
        "empirical_full_eigenvalues_available": False,
        "coordinate_convention": "(s0_a0,s0_a1,s1_a0,s1_a1), a2 fixed reference",
        "explore_contrast": [-1.0, 1.0, 0.0, 0.0],
        "source_archives": source_snapshot,
    }
    _prepare_output(output, config, resume)
    note = (
        "# Post-hoc implementation note\n\n"
        "All requested policy checkpoints are stored every 10 updates, so no training rerun is used. "
        "Behavior, exact reward gradients, conditional barriers, and population Fishers are reconstructed "
        "from stored float64 logits. Stored empirical diagnostics include K1, zero-s1, rank, minimum "
        "eigenvalue, and log-determinant status. Full empirical eigenvalue vectors and sampled actions were "
        "not archived and are therefore reported as unavailable. Exact counterfactual continuations use "
        "the deterministic population reward vector field only.\n"
    )
    (output / "posthoc_implementation_note.md").write_text(note, encoding="utf-8")

    print("Reconstructing switch and checkpoint metrics from stored logits", flush=True)
    switch_rows, trajectory_rows = build_switch_and_trajectory_rows(switch_archives)
    summary_rows = summarize_switch_rows(switch_rows)
    endpoint_rows = summarize_endpoints(trajectory_rows)
    overlap_rows = range_overlap_rows(switch_rows)
    print("Running deterministic exact-gradient counterfactual continuations", flush=True)
    counter_rows, counter_summary, counter_curves, continuation_data = build_counterfactual_rows(switch_archives)

    row_counts = {
        "posthoc_switch_metrics.csv": write_rows(output / "posthoc_switch_metrics.csv", switch_rows),
        "posthoc_switch_summary.csv": write_rows(output / "posthoc_switch_summary.csv", summary_rows),
        "posthoc_endpoint_summary.csv": write_rows(output / "posthoc_endpoint_summary.csv", endpoint_rows),
        "posthoc_range_overlap.csv": write_rows(output / "posthoc_range_overlap.csv", overlap_rows),
        "posthoc_trajectory_metrics.csv": write_rows(output / "posthoc_trajectory_metrics.csv", trajectory_rows),
        "posthoc_counterfactual_continuation.csv": write_rows(output / "posthoc_counterfactual_continuation.csv", counter_rows),
        "posthoc_counterfactual_summary.csv": write_rows(output / "posthoc_counterfactual_summary.csv", counter_summary),
        "posthoc_counterfactual_curves.csv": write_rows(output / "posthoc_counterfactual_curves.csv", counter_curves),
    }
    plots = output / "plots"; plots.mkdir(parents=True, exist_ok=True)
    print("Rendering focused figures A-F", flush=True)
    figure_a(plots, switch_rows); figure_b(plots, switch_rows); figure_c(plots, switch_rows)
    figure_d(plots, switch_archives, reference_archives); figure_e(plots, continuation_data); figure_f(plots, switch_rows)
    _verify_sources_unchanged(source_snapshot)
    artifacts = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name != "posthoc_manifest.json":
            artifacts[_safe_relative(path)] = {"sha256": sha256_file(path), "size": path.stat().st_size}
    manifest = {
        "schema_version": 1,
        "training_rerun": False,
        "source_archives_unchanged": True,
        "source_archives": source_snapshot,
        "row_counts": row_counts,
        "artifacts": artifacts,
    }
    _json_write(output / "posthoc_manifest.json", manifest)
    print(f"Post-hoc analysis completed in {output}", flush=True)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    try:
        run_posthoc(args.output_dir, resume=args.resume)
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
