"""Aggregation, plots, and scientific report for the neural experiment."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from vpg.stats import mean_confidence_interval


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        if not fields:
            return
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _ci(values: list[float]) -> tuple[float, float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    if len(array) == 1:
        return float(array[0]), float("nan"), float("nan"), float(np.median(array))
    mean, lower, upper = mean_confidence_interval(array)
    return float(mean), float(lower), float(upper), float(np.median(array))


def _all_run_directories(stage: Path) -> list[Path]:
    return sorted(path.parent for path in (stage / "runs").glob("*/seed_*/summary.json"))


def aggregate_confirmatory(root: Path) -> dict[str, list[dict]]:
    stage = root / "acrobot_confirmatory"
    endpoints: list[dict] = []
    checkpoint_behavior: list[dict] = []
    checkpoint_gradients: list[dict] = []
    training: dict[tuple[str, int], list[dict]] = {}
    for run_directory in _all_run_directories(stage):
        summary = json.loads((run_directory / "summary.json").read_text(encoding="utf-8"))
        label = run_directory.parent.name
        seed = int(summary["seed"])
        rows = _read_csv(run_directory / "training.csv")
        run_config = json.loads((run_directory / "config.json").read_text(encoding="utf-8"))
        if (
            run_config.get("collector_mode") == "complete_episodes"
            and len(rows) >= 2
            and int(rows[-1]["environment_steps"]) - int(rows[-2]["environment_steps"])
            < int(run_config["parallel_environments"]) * int(run_config["horizon"])
        ):
            # The final fixed-step remainder exists only to make interaction
            # counts exact. Its short truncated segments are not comparable to
            # the complete-episode training-return batches.
            rows = rows[:-1]
        training[(label, seed)] = rows
        returns = np.asarray([float(row["training_return"]) for row in rows], dtype=np.float64)
        steps = np.asarray([float(row["environment_steps"]) for row in rows], dtype=np.float64)
        auc = float(np.trapezoid(returns, steps) / steps[-1]) if len(steps) > 1 else float(returns[-1])
        final = summary["final"]
        endpoints.append({
            "run_label": label,
            "method": summary["method"],
            "seed": seed,
            "initial_weight_identifier": summary["initial_weight_identifier"],
            "optimizer": summary["optimizer"],
            "learning_rate": summary["learning_rate"],
            "total_environment_steps": summary["total_environment_steps"],
            "final_deterministic_return": final["deterministic_return"],
            "final_stochastic_return": final["stochastic_return"],
            "final_episode_length": final["episode_length"],
            "training_auc": auc,
            "failure": float(final["deterministic_return"]) < -200.0,
            "finite": summary["finite"],
        })
        for row in _read_csv(run_directory / "checkpoint_behavior.csv"):
            checkpoint_behavior.append({"run_label": label, **row})
        for row in _read_csv(run_directory / "checkpoint_gradients.csv"):
            checkpoint_gradients.append({"run_label": label, "method": summary["method"], "seed": seed, **row})

    endpoint_lookup = {(row["run_label"], int(row["seed"])): row for row in endpoints}
    labels = sorted({row["run_label"] for row in endpoints})
    baseline = "gpomdp_reward_only"
    paired: list[dict] = []
    for label in labels:
        if label == baseline:
            continue
        common = sorted({seed for name, seed in endpoint_lookup if name == label} & {seed for name, seed in endpoint_lookup if name == baseline})
        for metric in ("final_deterministic_return", "final_stochastic_return", "training_auc"):
            differences = [float(endpoint_lookup[(label, seed)][metric]) - float(endpoint_lookup[(baseline, seed)][metric]) for seed in common]
            mean, lower, upper, median = _ci(differences)
            paired.append({
                "method": label,
                "reference": baseline,
                "metric": metric,
                "seed_count": len(common),
                "mean_paired_difference": mean,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "median_paired_difference": median,
            })
    failures: list[dict] = []
    for label in labels:
        subset = [row for row in endpoints if row["run_label"] == label]
        failures.append({
            "run_label": label,
            "seed_count": len(subset),
            "failure_definition": "final deterministic Acrobot return < -200",
            "failure_count": sum(bool(row["failure"]) for row in subset),
            "failure_rate": float(np.mean([bool(row["failure"]) for row in subset])),
        })

    _write_csv(root / "seed_endpoints.csv", endpoints)
    _write_csv(root / "paired_method_differences.csv", paired)
    _write_csv(root / "checkpoint_behavior.csv", checkpoint_behavior)
    _write_csv(root / "checkpoint_gradients.csv", checkpoint_gradients)
    _write_csv(root / "failure_rates.csv", failures)

    fisher = _read_csv(stage / "checkpoint_fisher.csv")
    on_policy = [row for row in fisher if row["fisher"] == "on_policy"]
    reference = [row for row in fisher if row["fisher"] == "fixed_reference"]
    alignment = _read_csv(stage / "checkpoint_alignment.csv")
    _write_csv(root / "checkpoint_fisher_on_policy.csv", on_policy)
    _write_csv(root / "checkpoint_fisher_reference.csv", reference)
    _write_csv(root / "checkpoint_alignment.csv", alignment)
    return {
        "endpoints": endpoints,
        "paired": paired,
        "failures": failures,
        "behavior": checkpoint_behavior,
        "gradients": checkpoint_gradients,
        "fisher_on": on_policy,
        "fisher_reference": reference,
        "alignment": alignment,
        "training": training,
    }


def _mean_curve(rows_by_seed: list[list[dict]], x: str, y: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    series = [
        (
            np.asarray([float(row[x]) for row in rows], dtype=np.float64),
            np.asarray([float(row[y]) for row in rows], dtype=np.float64),
        )
        for rows in rows_by_seed
    ]
    identical = all(
        len(item_x) == len(series[0][0]) and np.array_equal(item_x, series[0][0])
        for item_x, _ in series
    )
    if identical:
        x_values = series[0][0]
        matrix = np.asarray([item_y for _, item_y in series])
    else:
        lower = max(float(item_x.min()) for item_x, _ in series)
        upper = min(float(item_x.max()) for item_x, _ in series)
        x_values = np.linspace(lower, upper, 101)
        matrix = np.asarray([
            np.interp(x_values, item_x, item_y) for item_x, item_y in series
        ])
    matrix = np.where(np.isfinite(matrix), matrix, np.nan)
    finite_counts = np.isfinite(matrix).sum(axis=0)
    means = np.divide(
        np.nansum(matrix, axis=0),
        finite_counts,
        out=np.full(matrix.shape[1], np.nan, dtype=np.float64),
        where=finite_counts > 0,
    )
    half = np.zeros_like(means)
    if matrix.shape[0] > 1:
        for index in range(matrix.shape[1]):
            finite = matrix[:, index][np.isfinite(matrix[:, index])]
            if finite.size >= 2:
                _, lower, upper = mean_confidence_interval(finite)
                half[index] = (float(upper) - float(lower)) / 2.0
            else:
                half[index] = np.nan
    return x_values, means, half


def create_plots(root: Path, data: dict[str, list[dict]]) -> None:
    plots = root / "plots"
    plots.mkdir(exist_ok=True)
    labels = sorted({key[0] for key in data["training"]})
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(labels))))

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for color, label in zip(colors, labels):
        seed_rows = [rows for (name, _), rows in data["training"].items() if name == label]
        x, mean, half = _mean_curve(seed_rows, "environment_steps", "training_return")
        axes[0, 0].plot(x, mean, label=label, color=color)
        axes[0, 0].fill_between(x, mean - half, mean + half, alpha=0.15, color=color)
        behavior_by_seed: dict[int, list[dict]] = {}
        for row in data["behavior"]:
            if row["run_label"] == label:
                behavior_by_seed.setdefault(int(row["seed"]), []).append(row)
        ordered = [sorted(rows, key=lambda row: int(row["environment_steps"])) for rows in behavior_by_seed.values()]
        if ordered:
            for axis, field in zip((axes[0, 1], axes[1, 0], axes[1, 1]), ("episode_length", "mean_min_probability", "entropy")):
                bx, bm, bh = _mean_curve(ordered, "environment_steps", field)
                axis.plot(bx, bm, label=label, color=color)
                axis.fill_between(bx, bm - bh, bm + bh, alpha=0.15, color=color)
    axes[0, 0].set_title("Training return"); axes[0, 0].set_xlabel("environment steps")
    axes[0, 1].set_title("Deterministic episode length"); axes[0, 1].set_xlabel("environment steps")
    axes[1, 0].set_title("Mean minimum action probability"); axes[1, 0].set_xlabel("environment steps")
    axes[1, 1].set_title("Policy entropy"); axes[1, 1].set_xlabel("environment steps")
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout(); figure.savefig(plots / "training_behavior.png", dpi=180); plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for color, label in zip(colors, labels):
        grouped: dict[int, list[dict]] = {}
        for row in data["gradients"]:
            if row["run_label"] == label:
                grouped.setdefault(int(row["seed"]), []).append(row)
        ordered = [sorted(rows, key=lambda row: int(row["environment_steps"])) for rows in grouped.values()]
        if not ordered:
            continue
        for axis, field in zip(axes.reshape(-1), ("reward_gradient_norm", "barrier_gradient_norm", "regularizer_gradient_norm", "reward_regularizer_cosine")):
            x, mean, half = _mean_curve(ordered, "environment_steps", field)
            axis.plot(x, mean, label=label, color=color)
            axis.fill_between(x, mean - half, mean + half, alpha=0.15, color=color)
            axis.set_title(field.replace("_", " ")); axis.set_xlabel("environment steps")
    axes[0, 0].legend(fontsize=7)
    figure.tight_layout(); figure.savefig(plots / "gradient_diagnostics.png", dpi=180); plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for color, label in zip(colors, labels):
        grouped: dict[int, list[dict]] = {}
        for row in data["behavior"]:
            if row["run_label"] == label:
                grouped.setdefault(int(row["seed"]), []).append(row)
        ordered = [sorted(rows, key=lambda row: int(row["environment_steps"])) for rows in grouped.values()]
        if not ordered:
            continue
        for axis, field in zip(axes, ("barrier_value", "beta")):
            x, mean, half = _mean_curve(ordered, "environment_steps", field)
            axis.plot(x, mean, label=label, color=color)
            axis.fill_between(x, mean - half, mean + half, alpha=0.15, color=color)
            axis.set_title(field.replace("_", " ")); axis.set_xlabel("environment steps")
    axes[0].set_ylabel("state-batch mean")
    axes[1].axvline(0.25 * 122880, color="black", linestyle=":", linewidth=1, label="25% target")
    axes[0].legend(fontsize=7); axes[1].legend(fontsize=7)
    figure.tight_layout(); figure.savefig(plots / "barrier_and_beta_schedule.png", dpi=180); plt.close(figure)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for fisher_rows, suffix, linestyle in ((data["fisher_on"], "on", "-"), (data["fisher_reference"], "reference", "--")):
        for color, label in zip(colors, labels):
            grouped: dict[int, list[dict]] = {}
            for row in fisher_rows:
                if row["run_label"] == label:
                    grouped.setdefault(int(row["seed"]), []).append(row)
            ordered = [sorted(rows, key=lambda row: int(row["environment_steps"])) for rows in grouped.values()]
            if not ordered:
                continue
            for axis, field in zip(axes.reshape(-1), ("smallest_positive_eigenvalue", "numerical_rank", "log_pseudodeterminant", "entropy_effective_rank")):
                x, mean, _ = _mean_curve(ordered, "environment_steps", field)
                axis.plot(x, mean, label=f"{label}:{suffix}", color=color, linestyle=linestyle)
                axis.set_title(field.replace("_", " ")); axis.set_xlabel("environment steps")
                if field == "smallest_positive_eigenvalue": axis.set_yscale("log")
    axes[0, 0].legend(fontsize=6, ncol=2)
    figure.tight_layout(); figure.savefig(plots / "fisher_checkpoint_metrics.png", dpi=180); plt.close(figure)

    spectra = root / "acrobot_confirmatory" / "fisher_spectra"
    representative_methods = (
        "gpomdp_reward_only",
        "gpomdp_logbarrier_fixed",
        "gpomdp_logbarrier_handoff_h25",
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=False, sharey=True)
    for row_index, fisher_name in enumerate(("on_policy", "fixed_reference")):
        fisher_rows = data["fisher_on"] if fisher_name == "on_policy" else data["fisher_reference"]
        for column_index, label in enumerate(representative_methods):
            candidates = sorted(
                [row for row in fisher_rows if row["run_label"] == label and int(row["seed"]) == 301],
                key=lambda row: int(row["environment_steps"]),
            )
            chosen = [candidates[0], min(candidates, key=lambda row: abs(int(row["environment_steps"]) - 30720)), candidates[-1]]
            for checkpoint_label, row in zip(("initial", "handoff", "final"), chosen):
                path = spectra / f"{label}__seed301__u{int(row['update']):06d}__{fisher_name}.npz"
                with np.load(path) as archive:
                    eigenvalues = archive["eigenvalues"]
                axes[row_index, column_index].plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, label=checkpoint_label)
            axes[row_index, column_index].set_yscale("log")
            axes[row_index, column_index].set_title(f"{label}\n{fisher_name}", fontsize=9)
            axes[row_index, column_index].set_xlabel("positive eigenvalue index")
            if column_index == 0:
                axes[row_index, column_index].set_ylabel("eigenvalue")
            axes[row_index, column_index].legend(fontsize=7)
    figure.tight_layout(); figure.savefig(plots / "checkpoint_full_spectra.png", dpi=180); plt.close(figure)

    behavior_lookup = {(row["run_label"], int(row["seed"]), int(row["environment_steps"])): row for row in data["behavior"]}
    figure, axes = plt.subplots(1, 4, figsize=(19, 4.5))
    for fisher_rows, marker, label in ((data["fisher_on"], "o", "on-policy"), (data["fisher_reference"], "x", "fixed-reference")):
        xs_rank=[]; xs_eig=[]; ys=[]; min_probs=[]
        for row in fisher_rows:
            key=(row["run_label"], int(row["seed"]), int(row["environment_steps"]))
            if key not in behavior_lookup: continue
            behavior=behavior_lookup[key]
            ys.append(float(behavior["deterministic_return"])); min_probs.append(float(behavior["mean_min_probability"]))
            xs_rank.append(float(row["entropy_effective_rank"])); xs_eig.append(float(row["smallest_positive_eigenvalue"]))
        axes[0].scatter(xs_rank, ys, s=12, alpha=.45, marker=marker, label=label)
        axes[1].scatter(xs_eig, ys, s=12, alpha=.45, marker=marker, label=label)
        axes[2].scatter(min_probs, xs_eig, s=12, alpha=.45, marker=marker, label=label)
    axes[0].set_xlabel("effective rank"); axes[0].set_ylabel("deterministic return")
    axes[1].set_xlabel("smallest positive eigenvalue"); axes[1].set_ylabel("deterministic return"); axes[1].set_xscale("log")
    axes[2].set_xlabel("minimum action probability"); axes[2].set_ylabel("smallest positive eigenvalue"); axes[2].set_yscale("log")
    alignment_reference = [row for row in data["alignment"] if row["fisher"] == "fixed_reference"]
    for label in labels:
        for seed in sorted({int(row["seed"]) for row in alignment_reference if row["run_label"] == label}):
            aligned = sorted([row for row in alignment_reference if row["run_label"] == label and int(row["seed"]) == seed], key=lambda row: int(row["environment_steps"]))
            behavioral = sorted([row for row in data["behavior"] if row["run_label"] == label and int(row["seed"]) == seed], key=lambda row: int(row["environment_steps"]))
            return_by_step = {int(row["environment_steps"]): float(row["deterministic_return"]) for row in behavioral}
            for current, following in zip(aligned[:-1], aligned[1:]):
                current_step = int(current["environment_steps"]); following_step = int(following["environment_steps"])
                if current_step in return_by_step and following_step in return_by_step:
                    axes[3].scatter(float(current["leading_k90_natural_energy_fraction"]), return_by_step[following_step] - return_by_step[current_step], s=10, alpha=.35)
    axes[3].set_xlabel("leading-subspace natural-energy fraction")
    axes[3].set_ylabel("next-checkpoint return change")
    axes[0].legend(); figure.suptitle("Descriptive geometry--behavior associations (not causal)")
    figure.tight_layout(); figure.savefig(plots / "geometry_vs_behavior.png", dpi=180); plt.close(figure)

    handoff_label = "gpomdp_logbarrier_handoff_h25"
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    behavior_grouped: dict[int, list[dict]] = {}
    for row in data["behavior"]:
        if row["run_label"] == handoff_label:
            behavior_grouped.setdefault(int(row["seed"]), []).append(row)
    behavior_ordered = [sorted(rows, key=lambda row: int(row["environment_steps"])) for rows in behavior_grouped.values()]
    for axis, field in zip(axes[0], ("deterministic_return", "entropy", "mean_min_probability")):
        x, mean, half = _mean_curve(behavior_ordered, "environment_steps", field)
        axis.plot(x, mean); axis.fill_between(x, mean-half, mean+half, alpha=.2)
        axis.set_title(field.replace("_", " ")); axis.axvline(30720, color="black", linestyle=":"); axis.set_xlabel("environment steps")
    for axis, field in zip(axes[1], ("smallest_positive_eigenvalue", "entropy_effective_rank", "log_pseudodeterminant")):
        for fisher_rows, fisher_label, linestyle in ((data["fisher_on"], "on-policy", "-"), (data["fisher_reference"], "fixed-reference", "--")):
            grouped: dict[int, list[dict]] = {}
            for row in fisher_rows:
                if row["run_label"] == handoff_label:
                    grouped.setdefault(int(row["seed"]), []).append(row)
            ordered = [sorted(rows, key=lambda row: int(row["environment_steps"])) for rows in grouped.values()]
            x, mean, _ = _mean_curve(ordered, "environment_steps", field)
            axis.plot(x, mean, label=fisher_label, linestyle=linestyle)
        axis.set_title(field.replace("_", " ")); axis.axvline(30720, color="black", linestyle=":"); axis.set_xlabel("environment steps")
        if field == "smallest_positive_eigenvalue": axis.set_yscale("log")
    axes[1, 0].legend(); figure.suptitle("Temporary-barrier handoff diagnostics (vertical line: 25% target)")
    figure.tight_layout(); figure.savefig(plots / "handoff_diagnostics.png", dpi=180); plt.close(figure)


def write_report(root: Path, data: dict[str, list[dict]]) -> None:
    labels = sorted({row["run_label"] for row in data["endpoints"]})
    endpoint_lines = []
    for label in labels:
        subset = [row for row in data["endpoints"] if row["run_label"] == label]
        values = [float(row["final_deterministic_return"]) for row in subset]
        mean, lower, upper, median = _ci(values)
        endpoint_lines.append(f"| `{label}` | {mean:.2f} [{lower:.2f}, {upper:.2f}] | {median:.2f} | {sum(bool(row['failure']) for row in subset)}/{len(subset)} |")
    paired_lines=[]
    for row in data["paired"]:
        if row["metric"] == "final_deterministic_return":
            paired_lines.append(f"| `{row['method']}` | {float(row['mean_paired_difference']):.2f} [{float(row['ci95_lower']):.2f}, {float(row['ci95_upper']):.2f}] |")

    support_lines = []
    fisher_lines = []
    for label in labels:
        behavior_final = []
        reference_final = []
        for seed in sorted({int(row["seed"]) for row in data["behavior"] if row["run_label"] == label}):
            rows = sorted([row for row in data["behavior"] if row["run_label"] == label and int(row["seed"]) == seed], key=lambda row: int(row["environment_steps"]))
            behavior_final.append(rows[-1])
            spectra = sorted([row for row in data["fisher_reference"] if row["run_label"] == label and int(row["seed"]) == seed], key=lambda row: int(row["environment_steps"]))
            reference_final.append(spectra[-1])
        support_lines.append(
            f"| `{label}` | {np.mean([float(row['mean_min_probability']) for row in behavior_final]):.4f} | "
            f"{np.mean([float(row['entropy']) for row in behavior_final]):.4f} | "
            f"{np.mean([float(row['barrier_value']) for row in behavior_final]):.4f} |"
        )
        fisher_lines.append(
            f"| `{label}` | {np.mean([float(row['numerical_rank']) for row in reference_final]):.1f} | "
            f"{np.mean([float(row['entropy_effective_rank']) for row in reference_final]):.2f} | "
            f"{np.mean([float(row['k90']) for row in reference_final]):.1f} | "
            f"{np.mean([float(row['smallest_positive_eigenvalue']) for row in reference_final]):.3e} |"
        )
    pilot = json.loads((root / "acrobot_pilot" / "pilot_selection.json").read_text(encoding="utf-8"))

    report = f"""# Neural discrete categorical log-barrier experiment

## Scope and naming

The implemented intervention is the **on-policy sampled-state conditional
categorical log barrier**

`B_state = mean_states mean_actions log pi(a|s)`.

Rollout states are detached and every categorical action is enumerated. This
is not a global neural Fisher log determinant or an occupancy-Fisher barrier.
The full parameter Fisher is analyzed only as an undamped diagnostic.

The primary GPOMDP methods share the `(8,8)` categorical MLP, Adam, learning
rate `{float(pilot['selected_learning_rate']):g}`, centered but unnormalized reward-to-go, paired seeds and
initial weights, and exactly equal environment-interaction budgets. NPG uses
SGD after the repository's damped sampled-action empirical-Fisher solve and is
a separate geometric reference.

## Completed tabular gate

The existing archive-only handoff post-hoc analysis was validated and not
rerun. It supports a reward-vector-field/behavioral feedback threshold: the
barrier preserves support long enough for downstream learning to become
self-sustaining. Better Fisher geometry accompanies the transition but is not
uniquely causal.

## Acrobot confirmatory endpoints

Higher Acrobot return is better (less negative). Intervals are two-sided 95%
Student-t intervals across the ten complete training seeds. Failure was
predeclared as final deterministic return below -200.

| Method | Final deterministic return, mean [95% CI] | Median | Failures |
|---|---:|---:|---:|
{chr(10).join(endpoint_lines)}

Paired final differences relative to reward-only:

| Method | Mean paired difference [95% CI] |
|---|---:|
{chr(10).join(paired_lines)}

The coefficient was calibrated once on independent early-gradient pilots:
`beta={float(pilot['selected_beta']):.3f}` made the median
`||beta grad B|| / ||grad J||` approximately `0.30`. The 25% handoff was chosen
because the 25% and 35% three-seed pilot means tied at
`{float(pilot['handoff_pilot_means']['gpomdp_logbarrier_handoff_h25']):.2f}`.
Every confirmatory run used exactly 122,880 interactions.
The raw archive retains the final short remainder batch used to hit that exact
budget; learning-curve plots and AUC omit that non-comparable truncated batch.

Final conditional-action support remained very similar across the four GPOMDP
methods:

| Method | Mean min action probability | Entropy | Barrier value |
|---|---:|---:|---:|
{chr(10).join(support_lines)}

The fixed barrier preserved slightly more support than reward-only, while the
temporary method lay between them after handoff. These differences did not
produce measurable deterministic-return improvement.

## Fisher interpretation

At each checkpoint, `F_on` enumerates all actions on fresh states from the
current policy, while `F_ref` enumerates all actions on one frozen state bank
constructed only from independent pilot runs. The reference bank is shared
across every confirmatory method, seed, and checkpoint. Both Fishers are
undamped. Numerical rank uses `1e-10 * max(1, lambda_max)` and is reported with
the score-row and parameter limits. The NPG optimizer's damping remains a
separate training choice.

Reported determinant-like information is the log pseudodeterminant over the
declared positive spectrum. It is not treated as a condition number. Likewise,
rank preservation is not itself an objective: successful categorical policies
may appropriately concentrate and lose weak Fisher directions late in
training.

Reward-gradient projections report both Euclidean and natural-gradient energy.
These diagnose whether preserved directions are reward-relevant; they do not
turn checkpoint associations into causal evidence.

Final fixed-reference summaries were:

| Method | Numerical rank | Entropy effective rank | k90 | Smallest positive eigenvalue |
|---|---:|---:|---:|---:|
{chr(10).join(fisher_lines)}

The four GPOMDP variants have nearly indistinguishable fixed-reference spectra
and alignment summaries. NPG instead concentrates sharply, with much smaller
effective rank and numerically zero minimum action probabilities, yet also
fails behaviorally. Thus the spectra describe different policy geometry but do
not rescue or explain successful learning, because there is no successful
Acrobot trajectory in this confirmatory run.

## Decision gate

The gate is **not passed**. The implementation is finite, the temporary schedule
sets beta exactly to zero, and the barrier modestly preserves action support,
but it improves neither final deterministic return nor paired learning-curve
area. All methods fail the declared threshold on all ten seeds. The next step
should therefore validate a learning-capable Acrobot GPOMDP baseline and study
reward-estimator/state-coverage diagnostics under a longer or otherwise
standardized budget. It should not be a broad beta sweep, and the project should
not yet claim neural confirmation of the tabular handoff mechanism. Gaussian
continuous-control barriers remain outside this stage.

## Limitations

- Beta and the learning rate were frozen using independent, small pilot sets;
  the experiment is not a broad hyperparameter comparison.
- On-policy Fisher changes confound conditional policy geometry with state
  visitation; the fixed-reference bank removes the second difference only on
  its declared pilot-state mixture.
- Checkpoints are repeated measurements, not independent samples.
- Scatter plots are descriptive and cannot establish that Fisher geometry
  caused subsequent return changes.
- The first fixed-segment Acrobot pilot was quarantined after it was found to
  truncate episodes too early; it is preserved as an implementation diagnostic
  and excluded from all scientific summaries.
"""
    (root / "report.md").write_text(report, encoding="utf-8")


def write_manifest(root: Path) -> None:
    artifacts = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and "quarantine" not in path.parts:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            artifacts[str(path.relative_to(root)).replace("\\", "/")] = {"size": path.stat().st_size, "sha256": digest}
    manifest = {
        "schema_version": 1,
        "experiment": "on-policy sampled-state conditional categorical log barrier",
        "artifacts": artifacts,
        "tabular_outputs_modified": False,
        "excluded_preserved_directories": [
            "acrobot_pilot_fixed_segment_quarantine"
        ],
    }
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def finalize_report(root: Path) -> dict:
    configs = root / "configs"
    configs.mkdir(exist_ok=True)
    for source, destination in (
        (root / "acrobot_pilot" / "pilot_selection.json", configs / "frozen_pilot_selection.json"),
        (root / "acrobot_confirmatory" / "confirmatory_result.json", configs / "confirmatory_protocol.json"),
        (root / "state_banks" / "acrobot_reference_states.json", configs / "state_bank_configuration.json"),
    ):
        destination.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    data = aggregate_confirmatory(root)
    create_plots(root, data)
    write_report(root, data)
    write_manifest(root)
    return {
        "endpoint_rows": len(data["endpoints"]),
        "paired_rows": len(data["paired"]),
        "on_policy_fisher_rows": len(data["fisher_on"]),
        "reference_fisher_rows": len(data["fisher_reference"]),
    }
