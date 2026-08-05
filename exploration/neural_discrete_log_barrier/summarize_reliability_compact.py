"""Compact digest of the pooled reliability comparison.

The 200 paired seeds live in two stages on disk -- ``conf_20`` holds 501..520 and
``reliability_extension_60_total`` holds 521..700 -- so the headline numbers
currently only exist in a chat transcript or have to be recomputed by hand. This
writes them to two small CSVs next to the run directories.

Nothing is recomputed independently: the per-seed endpoint, the failure rule, and
the Wilson interval all come from :mod:`reliability` itself
(``_environment_step_endpoint``, ``FAILURE_*``, ``_wilson_interval``), so this
digest cannot disagree with the confirmatory analysis. The exact McNemar test and
the paired t interval are computed here because the pooled pair set spans two
stages and no existing function takes that union.

Outputs, both in the stage directory given by ``--output``:

* ``reliability_compact_summary.csv`` -- one row per method: failures, rate,
  Wilson 95%, and paired secondary endpoints.
* ``reliability_compact_pairs.csv`` -- one row per seed, both arms side by side,
  so the discordant pairs behind the McNemar count are inspectable.

Usage::

    python -m exploration.neural_discrete_log_barrier.summarize_reliability_compact
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from vpg.stats import mean_confidence_interval

from .reliability import (
    FAILURE_RETURN_THRESHOLD,
    FAILURE_TERMINATION_THRESHOLD,
    _environment_step_endpoint,
    _wilson_interval,
)

DEFAULT_ROOT = Path(
    "exploration/results/neural_discrete_log_barrier/acrobot_gpomdp_regularizer_ablation"
)
METHODS = ("reward_only", "logbarrier_handoff_h25")
# The pair set is split across stages by seed, not by method. Both stages ran the
# same two arms, so a seed is usable only if both arms are present in one stage.
STAGES = ("conf_20", "reliability_extension_60_total")

SECONDARY = (
    ("final_stochastic_return", "final return"),
    ("final_stochastic_termination_rate", "termination rate"),
    ("environment_step_return_auc", "environment-step AUC"),
)


def _collect(root: Path) -> dict[str, dict[int, dict]]:
    """Per method, per seed endpoint, pooled over stages."""
    collected: dict[str, dict[int, dict]] = {method: {} for method in METHODS}
    for stage in STAGES:
        for method in METHODS:
            directory = root / stage / "runs" / method
            if not directory.is_dir():
                continue
            for run in sorted(directory.iterdir()):
                if not run.is_dir() or not run.name.startswith("seed_"):
                    continue
                endpoint = _environment_step_endpoint(run, method)
                seed = int(run.name.removeprefix("seed_"))
                if seed in collected[method]:
                    raise ValueError(f"seed {seed} appears twice for {method}")
                endpoint["seed"] = seed
                endpoint["stage"] = stage
                collected[method][seed] = endpoint
    return collected


def _exact_mcnemar(a: int, b: int) -> float:
    """Two-sided exact binomial test on the discordant pairs.

    Under the null each discordant pair is a fair coin, so the p-value is the
    two-sided binomial tail at n = a + b. Exact rather than chi-square because
    the discordant count here is small.
    """
    n = a + b
    if n == 0:
        return 1.0
    observed = min(a, b)
    tail = sum(math.comb(n, k) for k in range(observed + 1))
    return min(1.0, 2.0 * tail / (2.0**n))


def _paired_interval(differences: np.ndarray) -> tuple[float, float]:
    """Student-t 95% half-width on the paired mean, returned as (mean, half).

    Uses the same helper ``reliability`` uses for its own paired intervals, so the
    two agree by construction rather than by coincidence.
    """
    mean, low, _ = mean_confidence_interval(differences)
    return float(mean), float(mean - low)


def build(root: Path) -> tuple[list[dict], list[dict], dict]:
    collected = _collect(root)
    shared = sorted(set(collected[METHODS[0]]) & set(collected[METHODS[1]]))
    if not shared:
        raise SystemExit(f"no seeds present for both arms under {root}")

    pairs = []
    for seed in shared:
        reward, handoff = (collected[method][seed] for method in METHODS)
        row = {"seed": seed, "stage": reward["stage"]}
        for method, endpoint in zip(METHODS, (reward, handoff)):
            row[f"{method}_failure"] = endpoint["failure"]
            row[f"{method}_final_return"] = round(endpoint["final_stochastic_return"], 3)
            row[f"{method}_termination_rate"] = round(
                endpoint["final_stochastic_termination_rate"], 4
            )
            row[f"{method}_environment_step_auc"] = round(
                endpoint["environment_step_return_auc"], 3
            )
        row["discordant"] = row[f"{METHODS[0]}_failure"] != row[f"{METHODS[1]}_failure"]
        pairs.append(row)

    summaries = []
    for method in METHODS:
        endpoints = [collected[method][seed] for seed in shared]
        failures = sum(endpoint["failure"] for endpoint in endpoints)
        low, high = _wilson_interval(failures, len(endpoints))
        row = {
            "method": method,
            "paired_seeds": len(endpoints),
            "failures": failures,
            "failure_rate": round(failures / len(endpoints), 4),
            "failure_wilson95_low": round(low, 4),
            "failure_wilson95_high": round(high, 4),
        }
        for key, _ in SECONDARY:
            values = np.asarray([endpoint[key] for endpoint in endpoints])
            row[f"{key}_mean"] = round(float(values.mean()), 4)
            row[f"{key}_median"] = round(float(np.median(values)), 4)
            row[f"{key}_sd"] = round(float(values.std(ddof=1)), 4)
        summaries.append(row)

    # a: reward_only fails while the handoff survives -- the direction the
    # preregistration predicted. b: the reverse.
    a = sum(1 for row in pairs if row["reward_only_failure"] and not row["logbarrier_handoff_h25_failure"])
    b = sum(1 for row in pairs if row["logbarrier_handoff_h25_failure"] and not row["reward_only_failure"])
    both = sum(1 for row in pairs if row["reward_only_failure"] and row["logbarrier_handoff_h25_failure"])
    neither = len(pairs) - a - b - both

    paired = {}
    for key, label in SECONDARY:
        differences = np.asarray([
            collected[METHODS[1]][seed][key] - collected[METHODS[0]][seed][key]
            for seed in shared
        ])
        mean, half = _paired_interval(differences)
        paired[key] = {
            "label": label,
            "mean": round(mean, 4),
            "half_width": round(half, 4),
            "significant": bool(abs(mean) > half),
        }

    result = {
        "paired_seeds": len(pairs),
        "seed_min": shared[0],
        "seed_max": shared[-1],
        "contiguous": shared == list(range(shared[0], shared[-1] + 1)),
        "discordant_reward_fails_handoff_ok": a,
        "discordant_handoff_fails_reward_ok": b,
        "both_fail": both,
        "neither_fails": neither,
        "exact_mcnemar_p": _exact_mcnemar(a, b),
        "failure_rule": (
            f"final stochastic return < {FAILURE_RETURN_THRESHOLD} OR final "
            f"stochastic termination rate < {FAILURE_TERMINATION_THRESHOLD}"
        ),
        "paired_secondary": paired,
    }
    return summaries, pairs, result


def _write(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination directory. Default: <root>/reliability_extension_60_total",
    )
    args = parser.parse_args(argv)

    summaries, pairs, result = build(args.root)
    output = args.output or args.root / "reliability_extension_60_total"
    _write(output / "reliability_compact_summary.csv", summaries)
    _write(output / "reliability_compact_pairs.csv", pairs)

    print(
        f"{result['paired_seeds']} paired seeds "
        f"{result['seed_min']}..{result['seed_max']} "
        f"(contiguous={result['contiguous']})"
    )
    for row in summaries:
        print(
            f"  {row['method']:24} {row['failures']:3}/{row['paired_seeds']} "
            f"= {row['failure_rate']*100:5.2f}%  Wilson95 "
            f"[{row['failure_wilson95_low']*100:.2f}%, {row['failure_wilson95_high']*100:.2f}%]"
        )
    print(
        f"  discordant a={result['discordant_reward_fails_handoff_ok']} "
        f"b={result['discordant_handoff_fails_reward_ok']} "
        f"both={result['both_fail']} neither={result['neither_fails']}  "
        f"exact McNemar p={result['exact_mcnemar_p']:.6f}"
    )
    for key, entry in result["paired_secondary"].items():
        verdict = "SIG" if entry["significant"] else "ns"
        print(
            f"  paired {entry['label']:22} {entry['mean']:+9.4f} "
            f"+/-{entry['half_width']:.4f}  {verdict}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
