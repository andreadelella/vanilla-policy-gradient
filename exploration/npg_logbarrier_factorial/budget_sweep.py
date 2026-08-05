"""Sample-budget sweep: log-barrier handoff versus entropy handoff.

Motivation
----------
The fixed-budget grid (``entropy_damped_u2000``, 100 paired seeds, ``N x T =
{4,32,128} x 2000``) found that on the adverse start under Euclidean PG the
barrier handoff reaches 0.26/0.34/0.40 while the entropy handoff reaches
**0.00** in every cell, with zero discordant pairs in entropy's favour. That is
one horizon, though, and one budget split. A separation at a single budget does
not establish a separation in *sample complexity*: entropy might simply be
slower, in which case both curves reach 1 and the gap is a constant factor, not
a different rate. It might also be that the barrier's advantage shrinks once the
budget is large enough for plain reward-following to work unaided.

So this sweeps the budget and reports success as a function of it. The intended
use is to decide whether a finite-sample statement of the form "the barrier
needs fewer trajectories than entropy to reach the good basin with probability
1-delta" is worth trying to prove -- and if so, in which regime.

Budget definition
-----------------
Budget is **total sampled trajectories, ``B = N * T``**: ``N`` trajectories per
update times ``T`` updates. That is the honest sample-complexity currency here,
since both arms pay it identically and it is the only cost that grows.

``B`` is deliberately reached by more than one allocation. Every ``(N, T)`` in
the outer product of ``BATCH_SIZES`` and ``HORIZONS`` is run, so budgets such as
``B = 128000`` appear as ``(32, 4000)`` and ``(128, 1000)``. Comparing those two
separates *how much data* from *how it is spent*: an arm that only works at large
``N`` is being helped by gradient-noise reduction, whereas one that only works at
large ``T`` is being helped by having longer to travel. The barrier's rising
success in ``N`` at fixed ``T`` in the earlier grid suggests the first mechanism
matters for it, and the allocation contrast is what tests that.

Wall-clock is flat in ``N`` -- the sampler is vectorized over trajectories -- and
linear in ``T``, so cost is driven entirely by ``T``. Long horizons are cheap,
which is what makes the sweep affordable.

Held fixed across the sweep
---------------------------
* ``handoff_update = T // 2``. The *fraction* is held at 0.5 rather than the
  absolute update, matching both grids already on disk. Longer horizons
  therefore give the regularizer proportionally longer, which is the fair
  reading of "more budget" for a handoff method -- and it favours entropy, whose
  complaint could otherwise be that it never got time.
* ``beta = 0.2`` and ``c = 0.4685``, the force-matched pair from
  :mod:`run_sampled_entropy`. Equal initial push, so the sweep varies budget and
  holds the shape contrast fixed.

Coefficient robustness
----------------------
Force-matching pins ``c`` at one value, so a barrier win at that ``c`` alone
would be a claim about a point, not about the functional form. The secondary
arms re-run entropy at ``4c`` and ``16c`` on the adverse start at ``N = 32``. If
entropy still fails to move at ``16c``, the failure is structural -- the force
carries a factor of ``p`` and dies at the boundary regardless of scale. If it
succeeds, the honest conclusion is that entropy needs a much larger and
start-dependent coefficient to match a barrier that needed no tuning, which is a
weaker but still real asymmetry.

Prediction, recorded before the run
-----------------------------------
On adverse + Euclidean: the barrier's success rate rises with budget while
entropy's stays pinned near 0 across all four horizons, so the two curves
diverge rather than converge -- a rate separation, not a constant factor. On
adverse + natural both saturate, so no separation is detectable. On uniform
neither regularizer is needed and reward-only already succeeds, so all arms
coincide except possibly at the smallest budget.

Usage::

    python -m exploration.npg_logbarrier_factorial.budget_sweep --workers 10
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .run_experiment import DEFAULT_ROOT, _require_validation
from .run_sampled_entropy import (
    COMPARATOR_BETA,
    _install_damped_step,
    calibrate_entropy_coefficient,
)
from .sampled_two_state import (
    INITIALIZATIONS,
    SampledFactorialConfig,
    _write_csv,
    run_one,
)


# Powers of four, so each rung is a 4x budget increase and the ladder spans 64x
# in horizon. 250 is short enough that even the barrier should struggle; 16000 is
# 8x the horizon of the grid already on disk.
HORIZONS = (250, 1000, 4000, 16000)
BATCH_SIZES = (4, 32, 128)

# (label, method, entropy coefficient multiple or None)
PRIMARY_ARMS = (
    ("reward_only_pg", "sampled_pg_reward_only", None),
    ("barrier_pg", "sampled_pg_logbarrier_handoff", None),
    ("entropy_pg", "sampled_pg_entropy_handoff", 1.0),
    ("reward_only_npg", "sampled_npg_reward_only", None),
    ("barrier_npg", "sampled_npg_logbarrier_handoff", None),
    ("entropy_npg", "sampled_npg_entropy_handoff", 1.0),
)

# Coefficient-robustness arms. Restricted to the cell where the primary result
# lives -- adverse start, mid batch -- because their only job is to rule out
# "entropy lost because c was too small".
SCALE_MULTIPLES = (4.0, 16.0)
SCALE_INITIALIZATION = "adverse"
SCALE_BATCH_SIZE = 32

N_SEEDS = 100
SEED_CHUNK = 25

# Success is the pre-existing basin criterion from the factorial (final q >= 0.9
# and final pi1_good >= 0.9), reused verbatim so the sweep is comparable with
# damped_u2000 and entropy_damped_u2000 rather than defining a new bar.
SUCCESS_FIELD = "near_optimal_basin"


def _arm_label(base: str, multiple: float | None) -> str:
    if multiple is None or multiple == 1.0:
        return base
    return f"{base}_c{multiple:g}x"


def build_cells(coefficient: float) -> list[dict]:
    """Every (initialization, N, T, arm) cell, each tagged with its arm label."""

    cells: list[dict] = []
    for initialization in INITIALIZATIONS:
        for batch_size in BATCH_SIZES:
            for horizon in HORIZONS:
                for label, method, multiple in PRIMARY_ARMS:
                    cells.append({
                        "arm": label,
                        "coefficient_multiple": multiple or 0.0,
                        "config": SampledFactorialConfig(
                            method,
                            initialization,
                            batch_size,
                            N_SEEDS,
                            updates=horizon,
                            handoff_update=horizon // 2,
                            # Only the four forced checkpoints (handoff-1,
                            # handoff, handoff+1, T) are recorded; a periodic
                            # interval over 76M updates would dominate both the
                            # runtime and the artifact size.
                            record_interval=10**9,
                            entropy_coefficient=(
                                coefficient * multiple if multiple else 0.0
                            ),
                        ),
                    })
    for horizon in HORIZONS:
        for multiple in SCALE_MULTIPLES:
            for base, method in (
                ("entropy_pg", "sampled_pg_entropy_handoff"),
                ("entropy_npg", "sampled_npg_entropy_handoff"),
            ):
                cells.append({
                    "arm": _arm_label(base, multiple),
                    "coefficient_multiple": multiple,
                    "config": SampledFactorialConfig(
                        method,
                        SCALE_INITIALIZATION,
                        SCALE_BATCH_SIZE,
                        N_SEEDS,
                        updates=horizon,
                        handoff_update=horizon // 2,
                        record_interval=10**9,
                        entropy_coefficient=coefficient * multiple,
                    ),
                })
    return cells


def _chunks(total: int, size: int) -> list[tuple[int, int]]:
    return [(start, min(size, total - start)) for start in range(0, total, size)]


def _tag(rows: list[dict], cell: dict) -> list[dict]:
    """Stamp rows with the fields the sweep groups on but ``run_one`` omits.

    ``run_one``'s rows carry method/initialization/n_trajectories/seed but not the
    horizon or the coefficient, and here two cells can share every field it does
    emit while differing in ``updates`` -- so without this the endpoint table
    would silently collapse distinct budgets onto the same key.
    """

    config = cell["config"]
    budget = config.n_trajectories * config.updates
    return [
        {
            "arm": cell["arm"],
            "updates": config.updates,
            "handoff_update": config.handoff_update,
            "budget_trajectories": budget,
            "entropy_coefficient": config.entropy_coefficient,
            "coefficient_multiple": cell["coefficient_multiple"],
            **row,
        }
        for row in rows
    ]


def _run_task(payload: tuple[dict, str, float, int, int, str, str]):
    cell_fields, arm, multiple, offset, count, directory, name = payload
    config = SampledFactorialConfig(**cell_fields)
    cell = {"arm": arm, "coefficient_multiple": multiple, "config": config}
    chunk = replace(config, base_seed=config.base_seed + offset, n_seeds=count)
    started = time.perf_counter()
    checkpoints, endpoints, _ = run_one(chunk)
    for kind, rows in (("checkpoints", checkpoints), ("endpoints", endpoints)):
        final = Path(directory) / f"{name}.{kind}.csv"
        staged = final.with_suffix(".partial")
        _write_csv(staged, _tag(rows, cell))
        # Rename last, so a shard is only visible once it is whole and an
        # interrupted sweep resumes without trusting a truncated file.
        staged.replace(final)
    return name, len(endpoints), time.perf_counter() - started


def _worker_initializer() -> None:
    import torch

    torch.set_num_threads(1)
    _install_damped_step()


def _wilson(count: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = count / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half = z * math.sqrt(
        proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total)
    ) / denominator
    return max(0.0, center - half), min(1.0, center + half)


def _exact_mcnemar_p(a: int, b: int) -> float:
    n = a + b
    if n == 0:
        return 1.0
    smaller = min(a, b)
    tail = sum(math.comb(n, k) for k in range(smaller + 1)) / (2 ** n)
    return min(1.0, 2.0 * tail)


def _truthy(value) -> bool:
    return str(value).strip().lower() in ("true", "1")


CONTRASTS = (
    ("barrier_pg", "entropy_pg", "barrier_vs_entropy_euclidean"),
    ("barrier_npg", "entropy_npg", "barrier_vs_entropy_natural"),
    ("barrier_pg", "reward_only_pg", "barrier_vs_reward_only_euclidean"),
    ("barrier_npg", "reward_only_npg", "barrier_vs_reward_only_natural"),
    ("entropy_pg", "reward_only_pg", "entropy_vs_reward_only_euclidean"),
    ("entropy_npg", "reward_only_npg", "entropy_vs_reward_only_natural"),
    ("barrier_pg", "entropy_pg_c4x", "barrier_vs_entropy_4x_euclidean"),
    ("barrier_pg", "entropy_pg_c16x", "barrier_vs_entropy_16x_euclidean"),
    ("barrier_npg", "entropy_npg_c4x", "barrier_vs_entropy_4x_natural"),
    ("barrier_npg", "entropy_npg_c16x", "barrier_vs_entropy_16x_natural"),
)


def summarize(endpoints: list[dict], checkpoints: list[dict]) -> tuple[list, list]:
    """Per-cell success rates and paired budget-matched contrasts.

    Grouped on ``(initialization, n_trajectories, updates)`` -- unlike
    :func:`sampled_two_state._summaries`, which groups on the first two only and
    would merge every horizon in this sweep into one bucket.
    """

    handoff_good: dict[tuple, list[float]] = {}
    for row in checkpoints:
        key = (
            row["arm"], row["initialization"], int(row["n_trajectories"]),
            int(row["updates"]), int(row["seed"]),
        )
        if int(row["update"]) == int(row["handoff_update"]):
            handoff_good[key] = float(row["pi1_good"])

    grouped: dict[tuple, list[dict]] = {}
    for row in endpoints:
        key = (row["initialization"], int(row["n_trajectories"]), int(row["updates"]))
        grouped.setdefault(key, []).append(row)

    summaries: list[dict] = []
    contrasts: list[dict] = []
    for (initialization, batch_size, horizon) in sorted(grouped):
        group = grouped[(initialization, batch_size, horizon)]
        by_arm: dict[str, dict[int, dict]] = {}
        for row in group:
            by_arm.setdefault(row["arm"], {})[int(row["seed"])] = row

        for arm, rows in sorted(by_arm.items()):
            values = list(rows.values())
            successes = sum(_truthy(row[SUCCESS_FIELD]) for row in values)
            low, high = _wilson(successes, len(values))
            at_handoff = [
                handoff_good[(arm, initialization, batch_size, horizon, seed)]
                for seed in rows
                if (arm, initialization, batch_size, horizon, seed) in handoff_good
            ]
            summaries.append({
                "initialization": initialization,
                "n_trajectories": batch_size,
                "updates": horizon,
                "budget_trajectories": batch_size * horizon,
                "arm": arm,
                "n": len(values),
                "successes": successes,
                "success_rate": successes / len(values),
                "wilson95_low": round(low, 4),
                "wilson95_high": round(high, 4),
                "finite_rate": sum(_truthy(r["finite"]) for r in values) / len(values),
                "final_pi1_good_median": float(
                    np.median([float(r["final_pi1_good"]) for r in values])
                ),
                "final_return_median": float(
                    np.median([float(r["final_return"]) for r in values])
                ),
                # The regularizer's own doing, before the handoff releases it:
                # separates "the regularizer failed to move the policy" from
                # "it moved the policy and the reward phase fell back".
                "pi1_good_at_handoff_median": (
                    float(np.median(at_handoff)) if at_handoff else ""
                ),
            })

        for left, right, family in CONTRASTS:
            if left not in by_arm or right not in by_arm:
                continue
            seeds = sorted(set(by_arm[left]) & set(by_arm[right]))
            if not seeds:
                continue
            left_wins = sum(
                _truthy(by_arm[left][s][SUCCESS_FIELD])
                and not _truthy(by_arm[right][s][SUCCESS_FIELD])
                for s in seeds
            )
            right_wins = sum(
                _truthy(by_arm[right][s][SUCCESS_FIELD])
                and not _truthy(by_arm[left][s][SUCCESS_FIELD])
                for s in seeds
            )
            left_rate = sum(_truthy(by_arm[left][s][SUCCESS_FIELD]) for s in seeds) / len(seeds)
            right_rate = sum(_truthy(by_arm[right][s][SUCCESS_FIELD]) for s in seeds) / len(seeds)
            contrasts.append({
                "comparison_family": family,
                "initialization": initialization,
                "n_trajectories": batch_size,
                "updates": horizon,
                "budget_trajectories": batch_size * horizon,
                "left_arm": left,
                "right_arm": right,
                "paired_seeds": len(seeds),
                "left_success_rate": round(left_rate, 3),
                "right_success_rate": round(right_rate, 3),
                "success_rate_difference": round(left_rate - right_rate, 3),
                "left_only_wins": left_wins,
                "right_only_wins": right_wins,
                "exact_two_sided_mcnemar_p": _exact_mcnemar_p(left_wins, right_wins),
            })
    return summaries, contrasts


def _plot(summaries: list[dict], path: Path) -> None:
    arms = ("reward_only", "barrier", "entropy")
    colours = dict(zip(arms, ("0.55", "tab:blue", "tab:red")))
    styles = {4: ":", 32: "--", 128: "-"}
    panels = [
        (initialization, optimizer)
        for initialization in ("adverse", "uniform")
        for optimizer in ("pg", "npg")
    ]
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for axis, (initialization, optimizer) in zip(axes.ravel(), panels):
        for arm in arms:
            label = f"{arm}_{optimizer}"
            for batch_size in BATCH_SIZES:
                points = sorted(
                    (row["budget_trajectories"], row["success_rate"])
                    for row in summaries
                    if row["arm"] == label
                    and row["initialization"] == initialization
                    and row["n_trajectories"] == batch_size
                )
                if not points:
                    continue
                x, y = zip(*points)
                axis.plot(
                    x, y, styles[batch_size], color=colours[arm], marker="o",
                    markersize=3.5,
                    label=f"{arm} N={batch_size}",
                )
        axis.set_xscale("log")
        axis.set_ylim(-0.03, 1.03)
        axis.grid(alpha=0.3)
        axis.set_title(f"{initialization} start, {'Euclidean' if optimizer=='pg' else 'natural'}")
        axis.set_xlabel("budget: total sampled trajectories (N x T)")
    axes[0, 0].set_ylabel("success rate (near-optimal basin)")
    axes[1, 0].set_ylabel("success rate (near-optimal basin)")
    axes[0, 0].legend(fontsize=7, ncol=3, loc="upper left")
    figure.suptitle(
        "Sampled two-state trap: barrier vs entropy handoff at equal sample budget "
        f"({N_SEEDS} paired seeds per point)"
    )
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _load(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _concatenate(paths: list[Path], destination: Path) -> int:
    fields: list[str] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            for key in next(csv.reader(handle), []):
                if key not in fields:
                    fields.append(key)
    written = 0
    with destination.open("w", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields, restval="")
        writer.writeheader()
        for path in paths:
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    writer.writerow(row)
                    written += 1
    return written


def run_budget_sweep(
    output_directory: str | Path,
    *,
    workers: int | None = None,
    seed_chunk: int = SEED_CHUNK,
) -> dict:
    output = Path(output_directory)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_directory = output / "shards"
    shard_directory.mkdir(parents=True, exist_ok=True)

    calibration = calibrate_entropy_coefficient(output)
    coefficient = float(calibration["selected_entropy_coefficient"])
    cells = build_cells(coefficient)

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)

    plan = []
    for index, cell in enumerate(cells):
        for offset, count in _chunks(N_SEEDS, seed_chunk):
            plan.append((index, offset, count, f"cell{index:04d}_seed{offset:04d}"))
    pending = [
        entry for entry in plan
        if not all(
            (shard_directory / f"{entry[3]}.{kind}.csv").exists()
            for kind in ("checkpoints", "endpoints")
        )
    ]
    total_updates = sum(cell["config"].updates for cell in cells) * N_SEEDS
    print(
        f"budget sweep: {len(cells)} cells x {N_SEEDS} seeds -> {len(plan)} tasks; "
        f"{len(plan)-len(pending)} on disk, {len(pending)} pending; workers={workers}\n"
        f"  beta={COMPARATOR_BETA} vs c={coefficient:.6f} (force-matched at the adverse start)\n"
        f"  horizons {HORIZONS}, batches {BATCH_SIZES}, "
        f"budgets {min(BATCH_SIZES)*min(HORIZONS)}..{max(BATCH_SIZES)*max(HORIZONS)} trajectories\n"
        f"  {total_updates/1e6:.1f}M policy updates total",
        flush=True,
    )

    started = time.perf_counter()
    if pending:
        payloads = [
            (
                asdict(cells[index]["config"]),
                cells[index]["arm"],
                cells[index]["coefficient_multiple"],
                offset,
                count,
                str(shard_directory),
                name,
            )
            for index, offset, count, name in pending
        ]
        if workers == 1:
            _install_damped_step()
            for done, payload in enumerate(payloads, 1):
                name, rows, seconds = _run_task(payload)
                print(f"  {done}/{len(payloads)} {name} ({seconds:.1f}s)", flush=True)
        else:
            with ProcessPoolExecutor(
                max_workers=workers, initializer=_worker_initializer
            ) as pool:
                futures = [pool.submit(_run_task, payload) for payload in payloads]
                for done, future in enumerate(as_completed(futures), 1):
                    name, rows, seconds = future.result()
                    elapsed = time.perf_counter() - started
                    remaining = elapsed / done * (len(payloads) - done)
                    print(
                        f"  {done}/{len(payloads)} {name} ({seconds:.1f}s) "
                        f"elapsed {elapsed/60:.1f}m eta {remaining/60:.1f}m",
                        flush=True,
                    )
    wall_clock = time.perf_counter() - started

    ordered = [entry[3] for entry in plan]
    endpoint_paths = [shard_directory / f"{n}.endpoints.csv" for n in ordered]
    checkpoint_paths = [shard_directory / f"{n}.checkpoints.csv" for n in ordered]
    endpoint_rows = _concatenate(endpoint_paths, output / "budget_endpoints.csv")
    checkpoint_rows = _concatenate(checkpoint_paths, output / "budget_checkpoints.csv")

    summaries, contrasts = summarize(
        _load(endpoint_paths), _load(checkpoint_paths)
    )
    _write_csv(output / "budget_summaries.csv", summaries)
    _write_csv(output / "budget_contrasts.csv", contrasts)
    _plot(summaries, output / "budget_sweep.png")

    manifest = {
        "schema_version": 1,
        "complete": True,
        "stage": "entropy_vs_barrier_budget_sweep",
        "budget_definition": "total sampled trajectories, N * T",
        "horizons": list(HORIZONS),
        "batch_sizes": list(BATCH_SIZES),
        "handoff_fraction": 0.5,
        "seed_count": N_SEEDS,
        "base_seed": SampledFactorialConfig.base_seed,
        "paired_with_stages": ["damped_u2000", "entropy_damped_u2000"],
        "success_criterion": "final_q >= 0.9 and final_pi1_good >= 0.9",
        "beta": COMPARATOR_BETA,
        "entropy_coefficient": coefficient,
        "entropy_coefficient_multiples_tested": [1.0, *SCALE_MULTIPLES],
        "natural_step": "damped_identity_regularized",
        "cells": len(cells),
        "endpoint_rows": endpoint_rows,
        "checkpoint_rows": checkpoint_rows,
        "summary_rows": len(summaries),
        "contrast_rows": len(contrasts),
        "total_policy_updates": total_updates,
        "outcomes_used_for_selection": False,
        "predicted_direction": (
            "on adverse+Euclidean the barrier's success rate rises with budget while "
            "entropy stays near zero at every horizon, so the curves diverge rather "
            "than converge; on adverse+natural both saturate; on uniform all arms "
            "coincide because reward-only already succeeds"
        ),
        "wall_clock_seconds": round(wall_clock, 1),
        "workers": workers,
        "task_count": len(plan),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--seed-chunk", type=int, default=SEED_CHUNK)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    _require_validation(args.root)
    output = args.output or (
        args.root / "sampled_two_state" / "entropy_vs_barrier_budget_sweep"
    )
    result = run_budget_sweep(
        output, workers=args.workers, seed_chunk=args.seed_chunk
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
