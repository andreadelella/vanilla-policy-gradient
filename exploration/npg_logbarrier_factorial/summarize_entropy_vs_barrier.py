"""Paired entropy-versus-barrier contrast on the sampled two-state trap.

The entropy grid runs in its own directory, so its arms cannot be paired against
the barrier arms by :func:`sampled_two_state._summaries`, which only ever sees one
directory's endpoints. Both grids use ``base_seed = 91_000`` and the same
per-seed generator, so seed ``91_000 + i`` is the *same* stream of sampled
batches in both -- which is what makes a paired comparison across directories
legitimate rather than merely an aggregate one.

What this is testing
--------------------
On Acrobot the entropy handoff matched the log-barrier handoff, because both
restoring forces are bounded in logit space and agree within a factor of 1.6 over
the minimum-probability band Acrobot failures occupy (0.10-0.26). The adverse
two-state initialization starts the good action at ``p = 0.0159``, well past the
entropy gradient's non-monotone peak near ``p = 0.119``, where the barrier's
monotone ``1 - 3p`` is several times stronger. So the recorded prediction is that
entropy loses here, most visibly on the adverse initialization.

Reports per (initialization, batch size, optimizer):

* ``success_rate`` for both arms, with Wilson intervals, plus the paired
  difference and its exact-McNemar discordance counts.
* ``final_pi1_good_median`` -- the probability the good action ends at, which is
  the quantity the force-shape argument is actually about.
* ``completed_horizon_rate`` -- whether an arm survived at all, since on this MDP
  an NPG arm can collapse rather than merely underperform.

Usage::

    python -m exploration.npg_logbarrier_factorial.summarize_entropy_vs_barrier \
        --entropy entropy_damped_u2000 --barrier damped_u2000
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from .run_experiment import DEFAULT_ROOT


PAIRS = (
    ("sampled_pg_entropy_handoff", "sampled_pg_logbarrier_handoff", "euclidean"),
    ("sampled_npg_entropy_handoff", "sampled_npg_logbarrier_handoff", "natural"),
)


def _wilson(count: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    proportion = count / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total))
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)


def _exact_mcnemar_p(a: int, b: int) -> float:
    """Two-sided exact McNemar: binomial tail on the discordant pairs."""
    n = a + b
    if n == 0:
        return 1.0
    coefficients = [math.comb(n, k) for k in range(n + 1)]
    total = float(sum(coefficients))
    observed = min(a, b)
    tail = sum(coefficients[k] for k in range(observed + 1))
    return min(1.0, 2.0 * tail / total)


def _load(directory: Path) -> dict[tuple, dict]:
    path = directory / "sampled_endpoints.csv"
    if not path.exists():
        raise FileNotFoundError(f"no endpoints at {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return {
            (
                row["method"],
                row["initialization"],
                int(row["n_trajectories"]),
                int(row["seed"]),
            ): row
            for row in csv.DictReader(handle)
        }


def _truthy(value: str) -> bool:
    return str(value).strip().lower() in ("true", "1")


def build(entropy_directory: Path, barrier_directory: Path) -> list[dict]:
    entropy = _load(entropy_directory)
    barrier = _load(barrier_directory)

    cells = sorted({(key[1], key[2]) for key in entropy})
    rows: list[dict] = []
    for initialization, batch in cells:
        for entropy_method, barrier_method, family in PAIRS:
            seeds = sorted(
                seed
                for (method, init, n, seed) in entropy
                if method == entropy_method and init == initialization and n == batch
                if (barrier_method, initialization, batch, seed) in barrier
            )
            if not seeds:
                continue
            left = [entropy[(entropy_method, initialization, batch, s)] for s in seeds]
            right = [barrier[(barrier_method, initialization, batch, s)] for s in seeds]

            entropy_wins = sum(
                _truthy(a["near_optimal_basin"]) and not _truthy(b["near_optimal_basin"])
                for a, b in zip(left, right)
            )
            barrier_wins = sum(
                _truthy(b["near_optimal_basin"]) and not _truthy(a["near_optimal_basin"])
                for a, b in zip(left, right)
            )
            entropy_successes = sum(_truthy(a["near_optimal_basin"]) for a in left)
            barrier_successes = sum(_truthy(b["near_optimal_basin"]) for b in right)
            entropy_low, entropy_high = _wilson(entropy_successes, len(seeds))
            barrier_low, barrier_high = _wilson(barrier_successes, len(seeds))

            def median(rows_, key):
                return statistics.median(float(row[key]) for row in rows_)

            rows.append({
                "initialization": initialization,
                "n_trajectories": batch,
                "optimizer": family,
                "paired_seeds": len(seeds),
                "entropy_success_rate": round(entropy_successes / len(seeds), 3),
                "barrier_success_rate": round(barrier_successes / len(seeds), 3),
                "success_rate_difference": round(
                    (entropy_successes - barrier_successes) / len(seeds), 3
                ),
                "entropy_wilson95_low": round(entropy_low, 3),
                "entropy_wilson95_high": round(entropy_high, 3),
                "barrier_wilson95_low": round(barrier_low, 3),
                "barrier_wilson95_high": round(barrier_high, 3),
                "entropy_wins": entropy_wins,
                "barrier_wins": barrier_wins,
                "exact_two_sided_mcnemar_p": round(
                    _exact_mcnemar_p(entropy_wins, barrier_wins), 6
                ),
                "entropy_final_pi1_good_median": round(median(left, "final_pi1_good"), 4),
                "barrier_final_pi1_good_median": round(median(right, "final_pi1_good"), 4),
                "entropy_final_return_median": round(median(left, "final_return"), 4),
                "barrier_final_return_median": round(median(right, "final_return"), 4),
                "entropy_completed_horizon_rate": round(
                    sum(_truthy(a["finite"]) for a in left) / len(seeds), 3
                ),
                "barrier_completed_horizon_rate": round(
                    sum(_truthy(b["finite"]) for b in right) / len(seeds), 3
                ),
            })
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--entropy", default="entropy_damped_u2000")
    parser.add_argument("--barrier", default="damped_u2000")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    base = args.root / "sampled_two_state"
    rows = build(base / args.entropy, base / args.barrier)
    if not rows:
        raise SystemExit("no paired cells found")

    destination = args.output or (base / args.entropy / "entropy_vs_barrier.csv")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"paired entropy handoff vs log-barrier handoff, {rows[0]['paired_seeds']} seeds per cell")
    print(
        f"{'init':9} {'batch':>5} {'opt':>9} "
        f"{'entropy':>8} {'barrier':>8} {'diff':>7} "
        f"{'e_win':>5} {'b_win':>5} {'McNemar':>9} "
        f"{'e_pi_good':>9} {'b_pi_good':>9}"
    )
    for row in rows:
        print(
            f"{row['initialization']:9} {row['n_trajectories']:5} {row['optimizer']:>9} "
            f"{row['entropy_success_rate']:8.3f} {row['barrier_success_rate']:8.3f} "
            f"{row['success_rate_difference']:+7.3f} "
            f"{row['entropy_wins']:5} {row['barrier_wins']:5} "
            f"{row['exact_two_sided_mcnemar_p']:9.4g} "
            f"{row['entropy_final_pi1_good_median']:9.4f} "
            f"{row['barrier_final_pi1_good_median']:9.4f}"
        )
    print(f"\nwrote {destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
