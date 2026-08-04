"""Compact per-cell digest of a sampled two-state factorial run.

``sampled_method_summaries.csv`` already carries every statistic, but at 28
columns it is wide enough that the headline pattern is hard to see, and it omits
one thing that turns out to matter for reading the NPG rows: *how far each run
actually got*. ``finite=False`` in this experiment does not mean the policy blew
up to NaN -- it means the natural-gradient solve was rejected as invalid and
``run_one`` broke out of the update loop. A run can therefore be recorded as
non-finite and still have reached the near-optimal basin first.

This writes one small CSV, one row per (initialization, batch size, method),
carrying only the columns needed to read the result, plus ``last_update_median``
recovered from the checkpoint trace so early stopping is visible rather than
inferred. Reads the combined CSVs, writes nothing else, and streams the large
checkpoint file so memory stays flat.

Usage::

    python -m exploration.npg_logbarrier_factorial.summarize_sampled_compact \
        --input exploration/results/npg_logbarrier_factorial/sampled_two_state/full_parallel
"""

from __future__ import annotations

import argparse
import csv
import statistics
from collections import defaultdict
from pathlib import Path

CELL_KEY = ("initialization", "n_trajectories", "method")

# Order rows so the two arms of each comparison sit next to each other.
METHOD_ORDER = (
    "sampled_pg_reward_only",
    "sampled_pg_logbarrier_handoff",
    "sampled_pg_logbarrier_fixed",
    "sampled_npg_reward_only",
    "sampled_npg_logbarrier_handoff",
    "sampled_npg_logbarrier_fixed",
)


def _last_update_by_run(checkpoints: Path) -> dict[tuple, int]:
    """Highest recorded update per run, streamed so the 500 MB file is never held."""
    last: dict[tuple, int] = {}
    with checkpoints.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                row["initialization"],
                int(row["n_trajectories"]),
                row["method"],
                int(row["seed"]),
            )
            update = int(row["update"])
            if update > last.get(key, 0):
                last[key] = update
    return last


def build_digest(input_directory: Path) -> list[dict]:
    summaries = list(
        csv.DictReader((input_directory / "sampled_method_summaries.csv").open(
            newline="", encoding="utf-8"
        ))
    )
    checkpoints = input_directory / "sampled_checkpoints.csv"
    per_cell_updates: dict[tuple, list[int]] = defaultdict(list)
    if checkpoints.exists():
        for (init, n, method, _seed), update in _last_update_by_run(checkpoints).items():
            per_cell_updates[(init, n, method)].append(update)

    rows = []
    for summary in summaries:
        cell = (
            summary["initialization"],
            int(summary["n_trajectories"]),
            summary["method"],
        )
        updates = per_cell_updates.get(cell, [])
        rows.append({
            "initialization": summary["initialization"],
            "n_trajectories": int(summary["n_trajectories"]),
            "method": summary["method"],
            "seeds": int(summary["n"]),
            "success_rate": round(float(summary["success_rate"]), 3),
            "success_wilson95_low": round(float(summary["success_rate_wilson95_low"]), 3),
            "success_wilson95_high": round(float(summary["success_rate_wilson95_high"]), 3),
            "final_return_mean": round(float(summary["final_return_mean"]), 4),
            "final_return_median": round(float(summary["final_return_median"]), 4),
            # Fraction of runs that never hit an invalid natural solve. For the
            # NPG arms this is near zero while success is high, which is why the
            # next column is here.
            "completed_all_updates_rate": round(float(summary["finite_rate"]), 3),
            # A cell with no checkpoint rows at all stopped before the first
            # recorded update, so 0 is the honest value: it means "did not
            # survive to update 10", not "missing data".
            "last_update_median": (
                int(statistics.median(updates)) if updates else 0
            ),
            "last_update_min": min(updates) if updates else 0,
            "zero_s1_batch_fraction_mean": round(
                float(summary["zero_s1_batch_fraction_mean"]), 4
            ),
            "reached_safe_positive_rate": round(
                float(summary["delta_safe_positive_rate"]), 3
            ),
        })

    order = {method: index for index, method in enumerate(METHOD_ORDER)}
    rows.sort(key=lambda row: (
        row["initialization"],
        row["n_trajectories"],
        order.get(row["method"], len(order)),
    ))
    return rows


def write_digest(input_directory: Path, output: Path | None = None) -> Path:
    rows = build_digest(input_directory)
    destination = output or input_directory / "compact_summary.csv"
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Directory holding sampled_method_summaries.csv and sampled_checkpoints.csv.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination CSV. Default: <input>/compact_summary.csv",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    path = write_digest(args.input, args.output)
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    print(f"wrote {path} ({len(rows)} rows, {path.stat().st_size/1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
