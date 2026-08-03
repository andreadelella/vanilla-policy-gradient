"""Command-line analysis of saved training reward curves.

Post-hoc plotting of saved reward files. See plotting.py for live training-time plots.
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from vpg.stats import mean_confidence_interval


@dataclass(frozen=True)
class RewardData:
    curves: np.ndarray
    labels: list[str]
    seeds: np.ndarray | None
    sources: tuple[Path, ...]


_AGGREGATE_NAMES = ("training_rewards.npz", "training_rewards.npy")


def _seed_from_name(path: Path) -> int | None:
    """Extract the seed id from a per-seed reward path, or None if absent."""
    match = re.search(r"seed(\d+)", path.name) or re.search(
        r"seed(\d+)", path.parent.name
    )
    return int(match.group(1)) if match else None


def discover_seed_files(directory: Path) -> list[Path]:
    """Return the per-seed reward files in a run directory, ordered by seed.

    train.py's multiseed path writes each seed's curve twice, and both copies
    outlive an interrupted or resumed run:

        <run>/training_rewards_seed<N>.npz   once seed N finishes
        <run>/seed<N>/training_rewards.npz   by seed N itself, also mid-run

    Either is preferred over <run>/training_rewards.npz, which covers only the
    seeds of the most recent invocation: resuming a run with a different --seeds
    list overwrites it, silently dropping the earlier seeds from every plot.

    The top-level file wins when both exist, because it is written only after a
    seed completes. The nested copy is also written periodically during training,
    so for an interrupted seed it is the truncated curve -- still worth loading,
    but only when no completed copy exists.
    """
    by_seed: dict[int, Path] = {}
    # Nested copies first so the completed top-level files overwrite them.
    for pattern in ("seed*/training_rewards.npz", "training_rewards_seed*.npz"):
        for candidate in sorted(directory.glob(pattern)):
            seed = _seed_from_name(candidate)
            if seed is not None:
                by_seed[seed] = candidate
    return [by_seed[seed] for seed in sorted(by_seed)]


def resolve_reward_inputs(path: str | Path) -> tuple[Path, ...]:
    """Expand one input into the reward files it refers to.

    A file resolves to itself. A run directory resolves to every per-seed file it
    holds, falling back to the aggregate matrix when no per-seed file exists.
    """
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        seed_files = discover_seed_files(path)
        if seed_files:
            return tuple(seed_files)
        for name in _AGGREGATE_NAMES:
            candidate = path / name
            if candidate.is_file():
                return (candidate,)
        raise FileNotFoundError(
            f"No per-seed or aggregate training_rewards file found in {path}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"Reward input does not exist: {path}")
    if path.suffix.lower() not in {".npy", ".npz"}:
        raise ValueError(f"Unsupported reward file {path}; use .npy or .npz")
    return (path,)


def resolve_reward_path(path: str | Path) -> Path:
    """Resolve a single NPY/NPZ input or a run directory's aggregate rewards.

    Kept for callers that need exactly one path; prefer resolve_reward_inputs,
    which picks up every seed of a multiseed run.
    """
    path = Path(path).expanduser().resolve()
    if path.is_dir():
        for name in _AGGREGATE_NAMES:
            candidate = path / name
            if candidate.is_file():
                return candidate
        raise FileNotFoundError(
            f"No training_rewards.npz or training_rewards.npy found in {path}"
        )
    return resolve_reward_inputs(path)[0]


def load_reward_file(path: str | Path) -> RewardData:
    """Load one reward file and normalize it to [curves, iterations]."""
    path = resolve_reward_path(path)
    seeds = None

    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as archive:
            if "rewards" not in archive.files:
                raise ValueError(
                    f"{path} has no 'rewards' array (found {archive.files})"
                )
            curves = np.array(archive["rewards"], dtype=np.float64, copy=True)
            if "seeds" in archive.files:
                seeds = np.array(archive["seeds"], copy=True)
    else:
        curves = np.array(
            np.load(path, allow_pickle=False), dtype=np.float64, copy=True
        )

    if curves.ndim == 1:
        curves = curves[None, :]
    if curves.ndim != 2 or curves.shape[1] == 0:
        raise ValueError(
            f"Rewards in {path} must have shape [iterations] or "
            f"[curves, iterations], got {curves.shape}"
        )
    if not np.all(np.isfinite(curves)):
        raise ValueError(f"Rewards contain NaN or infinite values: {path}")

    if seeds is not None:
        if seeds.ndim != 1 or len(seeds) != len(curves):
            raise ValueError(
                f"Expected one seed per curve in {path}; "
                f"seeds={seeds.shape}, curves={curves.shape}"
            )
        labels = [f"seed {seed}" for seed in seeds]
    elif len(curves) == 1:
        labels = [path.stem]
    else:
        labels = [f"{path.stem} #{index + 1}" for index in range(len(curves))]

    return RewardData(curves, labels, seeds, (path,))


def load_reward_files(
    paths: Sequence[str | Path], truncate: bool = False
) -> RewardData:
    """Combine selected reward files, retaining every curve in input order.

    Directories expand to all of their per-seed files, so passing a multiseed run
    directory loads every seed rather than whatever the aggregate matrix happens
    to hold. Duplicate inputs are loaded once.

    truncate=True clips every curve to the shortest length instead of raising, so
    a run whose seeds stopped at different iterations can still be plotted.
    """
    if not paths:
        raise ValueError("Select at least one reward file or run directory")

    resolved: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        for candidate in resolve_reward_inputs(path):
            if candidate not in seen:
                seen.add(candidate)
                resolved.append(candidate)

    datasets = [load_reward_file(path) for path in resolved]
    iteration_counts = {dataset.curves.shape[1] for dataset in datasets}
    if len(iteration_counts) != 1:
        details = ", ".join(
            f"{dataset.sources[0].name}={dataset.curves.shape[1]}"
            for dataset in datasets
        )
        if not truncate:
            raise ValueError(
                f"All selected curves must have equal length: {details}. "
                f"Pass truncate=True to clip them to the shortest."
            )
        shortest = min(iteration_counts)
        print(f"truncating to {shortest} iterations ({details})")
        datasets = [
            RewardData(
                dataset.curves[:, :shortest],
                dataset.labels,
                dataset.seeds,
                dataset.sources,
            )
            for dataset in datasets
        ]

    curves = np.concatenate([dataset.curves for dataset in datasets], axis=0)
    labels = [
        label for dataset in datasets for label in dataset.labels
    ]
    sources = tuple(
        source for dataset in datasets for source in dataset.sources
    )
    if all(dataset.seeds is not None for dataset in datasets):
        seeds = np.concatenate([dataset.seeds for dataset in datasets])
    else:
        seeds = None
    return RewardData(curves, labels, seeds, sources)


def _finish_plot(
    output: str | Path,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Path:
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()
    return output


def plot_single(
    data: RewardData,
    output: str | Path,
    title: str = "Training rewards",
    xlabel: str = "Iteration",
    ylabel: str = "Average training return",
) -> Path:
    """Plot each selected reward curve without aggregation."""
    plt.figure(figsize=(8, 5))
    x_values = np.arange(data.curves.shape[1])
    for curve, label in zip(data.curves, data.labels):
        plt.plot(x_values, curve, label=label)
    return _finish_plot(output, title, xlabel, ylabel)


def plot_ci(
    data: RewardData,
    output: str | Path,
    title: str = "Training reward across seeds",
    xlabel: str = "Iteration",
    ylabel: str = "Average training return",
    label: str = "Mean",
) -> Path:
    """Plot a mean and 95% Student-t CI across the selected curves."""
    plt.figure(figsize=(8, 5))
    x_values = np.arange(data.curves.shape[1])
    if len(data.curves) == 1:
        plt.plot(x_values, data.curves[0], label=label)
    else:
        mean, lower, upper = mean_confidence_interval(data.curves)
        line, = plt.plot(x_values, mean, label=label)
        plt.fill_between(
            x_values,
            lower,
            upper,
            color=line.get_color(),
            alpha=0.25,
            label="95% CI",
        )
    return _finish_plot(output, title, xlabel, ylabel)


def plot_comparison(
    groups: dict[str, RewardData],
    output: str | Path,
    title: str = "Run comparison",
    xlabel: str = "Iteration",
    ylabel: str = "Average training return",
    mode: str = "mean",
    final_window: int = 100,
    truncate: bool = False,
) -> Path:
    """Compare labeled runs using their mean/CI or best selected curve.

    truncate=True clips every group to the shortest length instead of raising,
    for comparing runs that stopped at different iteration counts.
    """
    if not groups:
        raise ValueError("Select at least one run to compare")

    lengths = {
        data.curves.shape[1] for data in groups.values()
    }
    if len(lengths) != 1:
        details = ", ".join(
            f"{label}={data.curves.shape[1]}" for label, data in groups.items()
        )
        if not truncate:
            raise ValueError(
                f"Compared runs must have equal length: {details}. "
                f"Pass truncate=True to clip them to the shortest."
            )
        shortest = min(lengths)
        print(f"truncating every run to {shortest} iterations ({details})")
        groups = {
            label: RewardData(
                data.curves[:, :shortest], data.labels, data.seeds, data.sources
            )
            for label, data in groups.items()
        }
    if final_window <= 0:
        raise ValueError("final_window must be greater than zero")

    plt.figure(figsize=(8, 5))
    for label, data in groups.items():
        x_values = np.arange(data.curves.shape[1])
        if mode == "best":
            window = min(final_window, data.curves.shape[1])
            scores = data.curves[:, -window:].mean(axis=1)
            index = int(np.argmax(scores))
            curve_label = label
            if data.seeds is not None:
                curve_label = f"{label} (seed {data.seeds[index]})"
            plt.plot(x_values, data.curves[index], label=curve_label)
        elif len(data.curves) == 1:
            plt.plot(x_values, data.curves[0], label=label)
        else:
            mean, lower, upper = mean_confidence_interval(data.curves)
            line, = plt.plot(x_values, mean, label=label)
            plt.fill_between(
                x_values,
                lower,
                upper,
                color=line.get_color(),
                alpha=0.25,
            )
    return _finish_plot(output, title, xlabel, ylabel)


def _add_plot_arguments(
    parser: argparse.ArgumentParser,
    default_title: str,
    default_output: str,
) -> None:
    parser.add_argument(
        "--output", type=Path, default=default_output, help="Output image path."
    )
    parser.add_argument("--title", default=default_title, help="Graph title.")
    parser.add_argument("--xlabel", default="Iteration", help="X-axis title.")
    parser.add_argument(
        "--ylabel", default="Average training return", help="Y-axis title."
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help=(
            "Clip every curve to the shortest instead of failing when the "
            "selection mixes different iteration counts."
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot saved training reward files and run directories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    single = subparsers.add_parser(
        "single", help="Plot selected curves without aggregation."
    )
    single.add_argument("inputs", nargs="+", help="Reward files or run directories.")
    _add_plot_arguments(single, "Training rewards", "analysis_single.png")

    ci = subparsers.add_parser(
        "ci", help="Plot mean and 95%% CI for an exact input selection."
    )
    ci.add_argument("inputs", nargs="+", help="Reward files or run directories.")
    ci.add_argument("--label", default="Mean", help="Mean curve legend label.")
    _add_plot_arguments(ci, "Training reward across seeds", "analysis_ci.png")

    compare = subparsers.add_parser(
        "compare", help="Compare independently labeled groups of reward files."
    )
    compare.add_argument(
        "--run",
        action="append",
        nargs="+",
        metavar=("LABEL", "INPUT"),
        required=True,
        help=(
            "A label followed by one or more reward files/run directories. "
            "Repeat --run for each comparison group."
        ),
    )
    compare.add_argument(
        "--mode",
        choices=("mean", "best"),
        default="mean",
        help="Compare each group's mean/CI or its best final-window curve.",
    )
    compare.add_argument(
        "--final-window",
        type=int,
        default=100,
        help="Iterations used to choose a best curve.",
    )
    _add_plot_arguments(compare, "Run comparison", "comparison.png")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "single":
        output = plot_single(
            load_reward_files(args.inputs, truncate=args.truncate),
            args.output,
            args.title,
            args.xlabel,
            args.ylabel,
        )
    elif args.command == "ci":
        output = plot_ci(
            load_reward_files(args.inputs, truncate=args.truncate),
            args.output,
            args.title,
            args.xlabel,
            args.ylabel,
            args.label,
        )
    else:
        groups = {}
        for run in args.run:
            if len(run) < 2:
                raise ValueError("--run requires LABEL followed by at least one input")
            label, *inputs = run
            if label in groups:
                raise ValueError(f"Duplicate comparison label: {label}")
            groups[label] = load_reward_files(inputs, truncate=args.truncate)
        output = plot_comparison(
            groups,
            args.output,
            args.title,
            args.xlabel,
            args.ylabel,
            args.mode,
            args.final_window,
            truncate=args.truncate,
        )

    print(f"Saved {args.command} analysis: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
