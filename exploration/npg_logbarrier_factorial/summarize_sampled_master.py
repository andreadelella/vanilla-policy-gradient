"""One master digest across every sampled-factorial run: outcome plus collapse.

``summarize_sampled_compact`` writes one file per run and reports outcomes only.
Reading the damped/undamped contrast across four horizons therefore means opening
eight files and remembering which numbers mean "it worked" versus "it survived".
This writes a single CSV keyed by (scaling, horizon, initialization, batch,
method), so a run is one row and a comparison is a sort.

Beyond the outcome columns it carries the *collapse* columns, which is where the
damped/undamped difference actually lives. Success rate alone hides it: an
undamped cell can post a respectable success rate while every run in it has
already overflowed, because ``run_one`` keeps the last finite endpoint when it
breaks out of the update loop. The columns that expose that:

* ``completed_horizon_rate`` -- fraction of runs that never broke early.
* ``last_update_median`` / ``_min`` -- how far runs actually got. A handoff arm
  whose median sits one checkpoint past ``handoff_update`` collapsed *at* the
  handoff, which is the mechanism, not a coincidence.
* ``collapse_update_median`` -- same thing restricted to runs that did stop
  early, so it is not diluted by survivors. Blank when none stopped.
* ``step_norm_p50/p99/max`` -- reconstructed ``|step| = scale * |direction|``.
  The damped bound is ``sqrt(2*target_kl/damping)``; ``step_over_bound_rate``
  counts violations of it, which is 0 by construction when damped and nonzero
  exactly when the undamped scaling inflates.
* ``realized_kl_p99`` / ``max`` against ``target_kl`` -- how far the quadratic
  model was trusted past its validity.
* ``fisher_min_eigenvalue_p50`` and ``fisher_rank_deficient_rate`` -- the
  geometry itself degenerating, which is the cause the step statistics are the
  effect of.

Percentiles rather than means throughout: these distributions have 1e74 tails, so
a mean is not a summary of anything. Streams the checkpoint files, so peak memory
is one row regardless of the 320 MB inputs.

Usage::

    python -m exploration.npg_logbarrier_factorial.summarize_sampled_master
    python -m exploration.npg_logbarrier_factorial.summarize_sampled_master \
        --run damped_u2000 --run undamped_u2000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

from .run_experiment import DEFAULT_ROOT

# Checkpoints are written every RECORD_INTERVAL updates, so a collapse can only
# ever be localized to within that many updates.
RECORD_INTERVAL = 10

METHOD_ORDER = (
    "sampled_pg_reward_only",
    "sampled_pg_logbarrier_handoff",
    "sampled_pg_logbarrier_fixed",
    "sampled_npg_reward_only",
    "sampled_npg_logbarrier_handoff",
    "sampled_npg_logbarrier_fixed",
)


def _percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile on a pre-sorted list; inf-safe."""
    if not values:
        return float("nan")
    return values[min(len(values) - 1, int(fraction * (len(values) - 1)))]


def _float(text: str) -> float:
    """CSV field to float, mapping unparseable/blank to nan rather than raising."""
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


class _CellTrace:
    """Streaming accumulator for one (init, batch, method) cell's checkpoints."""

    def __init__(self) -> None:
        self.step_norms: list[float] = []
        self.realized_kl: list[float] = []
        self.min_eigenvalues: list[float] = []
        self.last_update: dict[int, int] = {}
        self.rank_deficient = 0
        self.rows = 0
        self.nonfinite_steps = 0

    def add(self, row: dict) -> None:
        self.rows += 1
        seed = int(row["seed"])
        update = int(row["update"])
        if update > self.last_update.get(seed, -1):
            self.last_update[seed] = update
        if int(row["undamped_fisher_rank"]) < 4:
            self.rank_deficient += 1
        eigenvalue = _float(row["undamped_fisher_minimum_eigenvalue"])
        if not math.isnan(eigenvalue):
            self.min_eigenvalues.append(eigenvalue)
        kl = _float(row["realized_kl"])
        if not math.isnan(kl):
            self.realized_kl.append(kl)
        # |step| is not stored directly; scale_factor * |direction| reconstructs
        # it exactly, and both are recorded for every natural update.
        scale = _float(row["scale_factor"])
        direction = _float(row["natural_direction_norm"])
        norm = scale * direction
        if math.isnan(norm) or math.isinf(norm):
            self.nonfinite_steps += 1
        else:
            self.step_norms.append(norm)


def _trace_by_cell(checkpoints: Path) -> dict[tuple, _CellTrace]:
    traces: dict[tuple, _CellTrace] = defaultdict(_CellTrace)
    with checkpoints.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["initialization"], int(row["n_trajectories"]), row["method"])
            traces[key].add(row)
    return traces


def _run_rows(directory: Path) -> list[dict]:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    horizon = int(manifest["preset"].rsplit("_u", 1)[1])
    handoff = int(manifest.get("handoff_update", horizon // 2))
    scaling = "damped" if manifest.get("natural_step", "").startswith("damped") else "undamped"

    summaries = list(csv.DictReader(
        (directory / "sampled_method_summaries.csv").open(newline="", encoding="utf-8")
    ))
    checkpoints = directory / "sampled_checkpoints.csv"
    traces = _trace_by_cell(checkpoints) if checkpoints.exists() else {}
    if not traces:
        print(
            f"  note: {directory.name} has no sampled_checkpoints.csv (gitignored); "
            "collapse and step columns will be blank"
        )

    # Read damping and target_kl from the configs the run actually used rather
    # than re-deriving them, so the bound below matches the executed arithmetic.
    configs = {
        (row["initialization"], int(row["n_trajectories"]), row["method"]): row
        for row in csv.DictReader(
            (directory / "method_configs.csv").open(newline="", encoding="utf-8")
        )
    }

    rows = []
    for summary in summaries:
        key = (summary["initialization"], int(summary["n_trajectories"]), summary["method"])
        config = configs[key]
        damping = float(config["damping"])
        target_kl = float(config["target_kl"])
        bound = math.sqrt(2.0 * target_kl / damping) if damping > 0 else float("inf")
        natural = "_npg_" in summary["method"]

        trace = traces.get(key)
        row = {
            "scaling": scaling,
            "horizon": horizon,
            "handoff_update": handoff,
            "initialization": summary["initialization"],
            "n_trajectories": int(summary["n_trajectories"]),
            "method": summary["method"],
            "optimizer": "npg" if natural else "pg",
            "seeds": int(summary["n"]),
            # ---- outcome ----
            "success_rate": round(float(summary["success_rate"]), 3),
            "success_wilson95_low": round(float(summary["success_rate_wilson95_low"]), 3),
            "success_wilson95_high": round(float(summary["success_rate_wilson95_high"]), 3),
            "final_return_mean": round(float(summary["final_return_mean"]), 4),
            "final_return_median": round(float(summary["final_return_median"]), 4),
            "final_q_median": round(float(summary["final_q_median"]), 4),
            "final_pi1_good_median": round(float(summary["final_pi1_good_median"]), 4),
            # ---- survival / collapse ----
            "completed_horizon_rate": round(float(summary["finite_rate"]), 3),
        }

        if trace:
            updates = sorted(trace.last_update.values())
            # Runs that stopped early. Restricting the median to these keeps the
            # collapse timing from being diluted by runs that finished.
            collapsed = [u for u in updates if u < horizon]
            steps = sorted(trace.step_norms)
            kls = sorted(trace.realized_kl)
            eigenvalues = sorted(trace.min_eigenvalues)
            over = sum(1 for value in steps if value > bound * (1.0 + 1e-9))
            row.update({
                "last_update_median": int(statistics.median(updates)) if updates else 0,
                "last_update_min": updates[0] if updates else 0,
                "collapsed_run_count": len(collapsed),
                "collapse_update_median": (
                    int(statistics.median(collapsed)) if collapsed else ""
                ),
                "collapse_at_handoff": (
                    # Within two record intervals after the handoff: the barrier
                    # held the metric up and its release is what broke the run.
                    # Overflow needs a step or two once beta drops, so an exact
                    # match on handoff_update would miss the mechanism.
                    bool(collapsed)
                    and 0 <= statistics.median(collapsed) - handoff <= 2 * RECORD_INTERVAL
                ),
                "fisher_rank_deficient_rate": round(trace.rank_deficient / trace.rows, 3),
                "fisher_min_eigenvalue_p50": f"{_percentile(eigenvalues, 0.5):.3e}",
            })
            if natural:
                row.update({
                    "step_norm_bound": round(bound, 4),
                    "step_norm_p50": f"{_percentile(steps, 0.5):.4g}",
                    "step_norm_p99": f"{_percentile(steps, 0.99):.4g}",
                    "step_norm_max": f"{steps[-1]:.4g}" if steps else "",
                    "step_over_bound_rate": round(over / len(steps), 4) if steps else "",
                    "nonfinite_step_count": trace.nonfinite_steps,
                    "target_kl": target_kl,
                    "realized_kl_p50": f"{_percentile(kls, 0.5):.3e}",
                    "realized_kl_p99": f"{_percentile(kls, 0.99):.3e}",
                    "realized_kl_max": f"{kls[-1]:.3e}" if kls else "",
                })
        rows.append(row)

    # Euclidean rows have no natural step, so the step/KL columns are absent
    # rather than zero. Fill them blank so every row has the same schema.
    fields = {key for row in rows for key in row}
    for row in rows:
        for field in fields:
            row.setdefault(field, "")
    return rows


def build(root: Path, runs: list[str] | None) -> list[dict]:
    directory = root / "sampled_two_state"
    names = runs or sorted(
        entry.name for entry in directory.iterdir()
        if entry.is_dir() and (entry / "manifest.json").exists()
        and ("damped_u" in entry.name or "undamped_u" in entry.name)
    )
    rows = []
    for name in names:
        print(f"reading {name} ...", flush=True)
        rows.extend(_run_rows(directory / name))

    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    rows.sort(key=lambda row: (
        row["scaling"],
        row["horizon"],
        row["initialization"],
        row["n_trajectories"],
        order.get(row["method"], len(order)),
    ))
    return rows


COLUMN_ORDER = (
    "scaling", "horizon", "handoff_update", "initialization", "n_trajectories",
    "method", "optimizer", "seeds",
    "success_rate", "success_wilson95_low", "success_wilson95_high",
    "final_return_mean", "final_return_median", "final_q_median",
    "final_pi1_good_median",
    "completed_horizon_rate", "last_update_median", "last_update_min",
    "collapsed_run_count", "collapse_update_median", "collapse_at_handoff",
    "fisher_rank_deficient_rate", "fisher_min_eigenvalue_p50",
    "step_norm_bound", "step_norm_p50", "step_norm_p99", "step_norm_max",
    "step_over_bound_rate", "nonfinite_step_count",
    "target_kl", "realized_kl_p50", "realized_kl_p99", "realized_kl_max",
)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--run",
        action="append",
        default=None,
        help="Run directory name, repeatable. Default: every damped_u*/undamped_u*.",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    rows = build(args.root, args.run)
    if not rows:
        raise SystemExit("no runs found")
    destination = args.output or (
        args.root / "sampled_two_state" / "master_summary.csv"
    )
    present = [column for column in COLUMN_ORDER if any(column in row for row in rows)]
    extra = sorted({key for row in rows for key in row} - set(present))
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=present + extra, restval="")
        writer.writeheader()
        writer.writerows(rows)
    size = destination.stat().st_size / 1024
    print(f"\nwrote {destination} ({len(rows)} rows, {len(present)+len(extra)} cols, {size:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
