"""Plots for empirical-Fisher eigenspectrum experiments."""

from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt


_COLORS = ["#176B87", "#C2410C", "#3F6212", "#7E22CE", "#374151"]


def plot_spectra(
    spectra: Sequence[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Save raw, normalized, and cumulative Fisher-spectrum plots."""

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for index, spectrum in enumerate(spectra):
        eigenvalues = spectrum["eigenvalues"]
        floor = max(spectrum["rank_tolerance"], np.finfo(np.float64).tiny)
        axis.semilogy(
            np.arange(1, eigenvalues.size + 1),
            np.maximum(eigenvalues, floor),
            color=_COLORS[index % len(_COLORS)],
            label=f"width {spectrum['width']} (P={eigenvalues.size})",
        )
    _finish_axis(
        figure,
        axis,
        output_dir / "raw_eigenspectrum.png",
        xlabel="Principal-component index",
        ylabel="Eigenvalue",
        title="Undamped empirical Fisher eigenspectrum",
        log_grid=True,
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for index, spectrum in enumerate(spectra):
        eigenvalues = spectrum["eigenvalues"]
        trace = spectrum["trace"]
        normalized = eigenvalues / trace if trace > 0 else np.zeros_like(eigenvalues)
        floor = (
            spectrum["rank_tolerance"] / trace
            if trace > 0
            else np.finfo(np.float64).tiny
        )
        axis.semilogy(
            np.arange(1, eigenvalues.size + 1),
            np.maximum(normalized, floor),
            color=_COLORS[index % len(_COLORS)],
            label=f"width {spectrum['width']}",
        )
    _finish_axis(
        figure,
        axis,
        output_dir / "trace_normalized_eigenspectrum.png",
        xlabel="Principal-component index",
        ylabel="Eigenvalue / Fisher trace",
        title="Trace-normalized Fisher eigenspectrum",
        log_grid=True,
    )

    figure, axis = plt.subplots(figsize=(7.2, 4.5))
    for index, spectrum in enumerate(spectra):
        eigenvalues = spectrum["eigenvalues"].copy()
        near_zero = np.abs(eigenvalues) <= spectrum["psd_tolerance"]
        eigenvalues[near_zero] = np.maximum(eigenvalues[near_zero], 0.0)
        total = float(np.sum(eigenvalues))
        cumulative = (
            np.cumsum(eigenvalues) / total
            if total > 0
            else np.zeros_like(eigenvalues)
        )
        axis.plot(
            np.arange(1, eigenvalues.size + 1),
            cumulative,
            color=_COLORS[index % len(_COLORS)],
            label=f"width {spectrum['width']}",
        )
    for threshold in (0.90, 0.95, 0.99):
        axis.axhline(threshold, color="#6B7280", linewidth=0.8, linestyle="--")
    axis.set_ylim(0.0, 1.01)
    _finish_axis(
        figure,
        axis,
        output_dir / "cumulative_explained_trace.png",
        xlabel="Number of principal components",
        ylabel="Cumulative explained Fisher trace",
        title="Cumulative Fisher trace",
    )


def _finish_axis(
    figure,
    axis,
    output_path: Path,
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    log_grid: bool = False,
) -> None:
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(True, which="both" if log_grid else "major", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
