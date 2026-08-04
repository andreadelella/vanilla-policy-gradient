"""Parallel driver for the sampled two-state factorial.

The serial :func:`sampled_two_state.run_sampled_factorial` runs all 36 cells
(2 initializations x 3 batch sizes x 6 methods) in one process and accumulates
every checkpoint row in memory until the loop ends, so it writes nothing until
the whole grid is done: a crash at 95% loses everything, and the heap grows to
several GB.

This module changes only the *scheduling*, never the simulation. Per-seed work
still comes from :func:`sampled_two_state.run_one`, and the CSV/summary/plot
writers are the originals, so the outputs are equivalent to the serial run --
see ``--self-check``, which diffs both paths on the smoke preset.

Three differences, all in the driver:

* work is split into (cell, seed-chunk) tasks and spread over processes;
* each task writes its own shard under ``shards/`` as soon as it finishes, so
  the run is resumable and the parent never holds the checkpoint rows;
* the combined CSVs are assembled by streaming the shards, keeping peak memory
  proportional to one shard rather than the whole grid.

Seed identity is preserved by construction. ``run_one`` derives both the RNG and
the recorded seed from ``base_seed + seed_index``, so a chunk covering
``seed_index`` 25..49 is reproduced exactly by a config with
``base_seed += 25`` and ``n_seeds = 25``.

Usage::

    python -m exploration.npg_logbarrier_factorial.sampled_two_state_parallel \
        --preset full --workers 10
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, replace
from pathlib import Path

from .run_experiment import DEFAULT_ROOT, _require_validation
from .sampled_two_state import (
    INITIALIZATIONS,
    METHODS,
    SampledFactorialConfig,
    _plot,
    _summaries,
    _write_csv,
    run_one,
)


# Mirrors the preset table inside run_sampled_factorial. Kept as data here so the
# serial module is untouched; --self-check fails loudly if the two ever diverge.
PRESETS = {
    "smoke": {"n_seeds": 4, "updates": 50, "handoff": 25, "batch_sizes": (4, 32)},
    "pilot": {"n_seeds": 20, "updates": 4000, "handoff": 2000, "batch_sizes": (4, 32, 128)},
    "full": {"n_seeds": 100, "updates": 4000, "handoff": 2000, "batch_sizes": (4, 32, 128)},
}

SHARD_KINDS = ("checkpoints", "endpoints", "missing")


def _cell_configs(preset: str) -> list[SampledFactorialConfig]:
    """The 36 canonical cells, in the serial loop's order.

    Row order in the combined CSVs follows this order, so a parallel run is
    byte-comparable with a serial one even though tasks finish out of order.
    """
    spec = PRESETS[preset]
    return [
        SampledFactorialConfig(
            method,
            initialization,
            n,
            spec["n_seeds"],
            updates=spec["updates"],
            handoff_update=spec["handoff"],
            record_interval=10,
        )
        for initialization in INITIALIZATIONS
        for n in spec["batch_sizes"]
        for method in METHODS
    ]


def _chunks(total: int, size: int) -> list[tuple[int, int]]:
    return [(start, min(size, total - start)) for start in range(0, total, size)]


def _task_name(cell_index: int, offset: int) -> str:
    return f"cell{cell_index:02d}_seed{offset:04d}"


def _shard_path(directory: Path, name: str, kind: str) -> Path:
    return directory / f"{name}.{kind}.csv"


def _initializer() -> None:
    # One thread per worker: these are 4-parameter policies, so intra-op
    # threading only adds contention once several workers run at once.
    import torch

    torch.set_num_threads(1)


def _run_task(payload: tuple[dict, int, int, str, str]) -> tuple[str, int, float]:
    """Run one seed-chunk of one cell and write its shards. Runs in a worker."""
    cell_fields, offset, count, directory, name = payload
    shard_directory = Path(directory)
    config = SampledFactorialConfig(**cell_fields)
    chunk = replace(
        config,
        base_seed=config.base_seed + offset,
        n_seeds=count,
    )
    started = time.perf_counter()
    checkpoints, endpoints, missing = run_one(chunk)
    for kind, rows in (
        ("checkpoints", checkpoints),
        ("endpoints", endpoints),
        ("missing", missing),
    ):
        final = _shard_path(shard_directory, name, kind)
        staged = final.with_suffix(".partial")
        _write_csv(staged, rows)
        # Rename last: a shard is only visible once it is complete, so an
        # interrupted run resumes without trusting a half-written file.
        staged.replace(final)
    return name, len(endpoints), time.perf_counter() - started


def _shard_complete(directory: Path, name: str) -> bool:
    return all(_shard_path(directory, name, kind).exists() for kind in SHARD_KINDS)


def _union_fields(paths: list[Path]) -> list[str]:
    """Field union in first-appearance order, matching _write_csv on one big list.

    Every row a cell emits has the same keys, so taking the union over shard
    headers in cell order gives the same column order the serial writer would
    have derived row by row.
    """
    fields: list[str] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            header = next(csv.reader(handle), [])
        for key in header:
            if key not in fields:
                fields.append(key)
    return fields


def _concatenate(paths: list[Path], destination: Path) -> int:
    """Stream shards into one CSV, so peak memory stays at one shard."""
    fields = _union_fields(paths)
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


def _load_rows(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def _coerce_endpoints(rows: list[dict]) -> list[dict]:
    """Restore the types _summaries and _plot expect after the CSV round-trip."""
    coerced = []
    for row in rows:
        entry = dict(row)
        entry["n_trajectories"] = int(row["n_trajectories"])
        entry["seed"] = int(row["seed"])
        entry["finite"] = row["finite"] == "True"
        entry["near_optimal_basin"] = row["near_optimal_basin"] == "True"
        entry["invalid_solve_count"] = int(row["invalid_solve_count"])
        for key in ("final_return", "final_q", "final_pi1_good", "zero_s1_batch_fraction"):
            entry[key] = float(row[key])
        first = row["first_delta_safe_positive_update"]
        entry["first_delta_safe_positive_update"] = int(first) if first else None
        coerced.append(entry)
    return coerced


def run_sampled_factorial_parallel(
    output_directory: str | Path,
    *,
    preset: str = "smoke",
    workers: int | None = None,
    seed_chunk: int = 25,
) -> dict:
    """Run the sampled factorial across processes, writing shards as it goes."""
    if preset not in PRESETS:
        raise ValueError("preset must be smoke, pilot, or full")
    if seed_chunk < 1:
        raise ValueError("seed_chunk must be positive")
    output = Path(output_directory)
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    shard_directory = output / "shards"
    shard_directory.mkdir(parents=True, exist_ok=True)

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 2)

    cells = _cell_configs(preset)
    n_seeds = PRESETS[preset]["n_seeds"]
    plan: list[tuple[int, int, int, str]] = []
    for cell_index, _ in enumerate(cells):
        for offset, count in _chunks(n_seeds, seed_chunk):
            plan.append((cell_index, offset, count, _task_name(cell_index, offset)))

    pending = [entry for entry in plan if not _shard_complete(shard_directory, entry[3])]
    print(
        f"{len(cells)} cells x {n_seeds} seeds -> {len(plan)} tasks; "
        f"{len(plan)-len(pending)} already on disk, {len(pending)} pending; "
        f"workers={workers}, seed_chunk={seed_chunk}",
        flush=True,
    )

    started = time.perf_counter()
    if pending:
        payloads = [
            (asdict(cells[index]), offset, count, str(shard_directory), name)
            for index, offset, count, name in pending
        ]
        if workers == 1:
            for done, payload in enumerate(payloads, 1):
                name, rows, seconds = _run_task(payload)
                print(f"  {done}/{len(payloads)} {name} ({rows} seeds, {seconds:.1f}s)", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=workers, initializer=_initializer) as pool:
                futures = [pool.submit(_run_task, payload) for payload in payloads]
                for done, future in enumerate(as_completed(futures), 1):
                    name, rows, seconds = future.result()
                    elapsed = time.perf_counter() - started
                    remaining = elapsed / done * (len(payloads) - done)
                    print(
                        f"  {done}/{len(payloads)} {name} ({rows} seeds, {seconds:.1f}s) "
                        f"elapsed {elapsed/60:.1f}m, eta {remaining/60:.1f}m",
                        flush=True,
                    )
    wall_clock = time.perf_counter() - started

    ordered = [entry[3] for entry in plan]
    counts = {}
    for kind, filename in (
        ("checkpoints", "sampled_checkpoints.csv"),
        ("missing", "sampled_missing_state_audit.csv"),
    ):
        paths = [_shard_path(shard_directory, name, kind) for name in ordered]
        counts[kind] = _concatenate(paths, output / filename)

    endpoint_paths = [_shard_path(shard_directory, name, "endpoints") for name in ordered]
    counts["endpoints"] = _concatenate(endpoint_paths, output / "sampled_endpoints.csv")
    endpoints = _coerce_endpoints(_load_rows(endpoint_paths))

    _write_csv(output / "method_configs.csv", [asdict(cell) for cell in cells])
    summaries, paired = _summaries(endpoints)
    _write_csv(output / "sampled_method_summaries.csv", summaries)
    _write_csv(output / "sampled_paired_differences.csv", paired)
    _plot(endpoints, output / "sampled_endpoints.png")

    manifest = {
        "schema_version": 1,
        "complete": True,
        "preset": preset,
        "seed_count": n_seeds,
        "endpoint_rows": counts["endpoints"],
        "checkpoint_rows": counts["checkpoints"],
        "missing_state_rows": counts["missing"],
        "summary_rows": len(summaries),
        "paired_difference_rows": len(paired),
        "raw_uncentered_unnormalized_returns": True,
        "oracle_information_inserted_for_missing_s1": False,
        # Driver-only provenance; the simulation is the serial code path.
        "driver": "parallel",
        "workers": workers,
        "seed_chunk": seed_chunk,
        "task_count": len(plan),
        "wall_clock_seconds": round(wall_clock, 1),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def _self_check(root: Path) -> int:
    """Prove the parallel driver reproduces the serial run on the smoke preset."""
    import shutil

    from .sampled_two_state import run_sampled_factorial

    scratch = root / "_self_check"
    if scratch.exists():
        shutil.rmtree(scratch)
    serial_output = scratch / "serial"
    parallel_output = scratch / "parallel"

    print("running serial smoke ...", flush=True)
    run_sampled_factorial(serial_output, preset="smoke")
    print("running parallel smoke (chunked, 2 seeds per task) ...", flush=True)
    run_sampled_factorial_parallel(
        parallel_output, preset="smoke", workers=4, seed_chunk=2
    )

    failures = []
    for name in (
        "sampled_checkpoints.csv",
        "sampled_endpoints.csv",
        "sampled_missing_state_audit.csv",
        "method_configs.csv",
        "sampled_method_summaries.csv",
        "sampled_paired_differences.csv",
    ):
        left = (serial_output / name).read_text(encoding="utf-8")
        right = (parallel_output / name).read_text(encoding="utf-8")
        status = "identical" if left == right else "DIFFERS"
        if left != right:
            failures.append(name)
            left_lines, right_lines = left.splitlines(), right.splitlines()
            status += f" (serial {len(left_lines)} lines, parallel {len(right_lines)})"
            for index, (a, b) in enumerate(zip(left_lines, right_lines)):
                if a != b:
                    status += f"; first diff line {index+1}"
                    break
        print(f"  {name:38} {status}")
    if failures:
        print(f"\nSELF-CHECK FAILED: {failures}")
        return 1
    print("\nSELF-CHECK PASSED: parallel output is byte-identical to serial")
    shutil.rmtree(scratch)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--preset", choices=tuple(PRESETS), default="smoke")
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Worker processes. Default: CPU count minus two.",
    )
    parser.add_argument(
        "--seed-chunk",
        type=int,
        default=25,
        help="Seeds per task. Smaller means finer progress and better balance.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory. Default: <root>/sampled_two_state/<preset>_parallel, "
             "kept apart from the serial preset directory.",
    )
    parser.add_argument(
        "--self-check",
        action="store_true",
        help="Diff a serial and a parallel smoke run, then exit.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.self_check:
        return _self_check(args.root)
    _require_validation(args.root)
    output = args.output or args.root / "sampled_two_state" / f"{args.preset}_parallel"
    result = run_sampled_factorial_parallel(
        output,
        preset=args.preset,
        workers=args.workers,
        seed_chunk=args.seed_chunk,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
