# Run the eigenvalue notebook locally

See [`fisher_analysis.md`](fisher_analysis.md) for the derivation, results, and
limitations of this analysis. This document only covers re-running the notebook.

The notebook [`eigenvalue_analysis.ipynb`](eigenvalue_analysis.ipynb) does **not**
collect trajectories or recompute the Fisher matrices. It validates the saved
`.npz` files, displays `summary.csv`, and regenerates the spectrum plots.

Run every command below from the repository root.

## 1. Start Jupyter with the existing environment

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook fisher_analysis\eigenvalue_analysis.ipynb
```

If this opens Jupyter, select **Run All Cells**. No MuJoCo process is started by
the notebook.

## 2. Fallback if the existing environment is broken

Install a normal Python 3.10+ distribution first. Then create a separate,
lightweight environment; this avoids deleting or changing the training
environment:

```powershell
py -3.13 -m venv .venv-analysis
.\.venv-analysis\Scripts\python.exe -m pip install --upgrade pip
.\.venv-analysis\Scripts\python.exe -m pip install numpy matplotlib jupyter ipykernel
.\.venv-analysis\Scripts\python.exe -m jupyter notebook fisher_analysis\eigenvalue_analysis.ipynb
```

If the installed version is not Python 3.13, replace `py -3.13` with the
available launcher, for example `py -3.12`, or use the full path to
`python.exe`.

Only NumPy, Matplotlib, Jupyter, and IPython are needed to analyze the existing
artifacts. PyTorch, Gymnasium, and MuJoCo are needed only if the Fisher matrices
themselves must be recomputed.

## 3. Choose the results in the first code cell

Hopper:

```python
RESULTS_DIR = "fisher_analysis/results/hopper_width_sweep"
OUTPUT_DIR = "fisher_analysis/results/hopper_width_sweep"
```

HalfCheetah:

```python
RESULTS_DIR = "fisher_analysis/results/halfcheetah_width_sweep"
OUTPUT_DIR = "fisher_analysis/results/halfcheetah_width_sweep"
```

Swimmer:

```python
RESULTS_DIR = "fisher_analysis/results/swimmer_width_sweep"
OUTPUT_DIR = "fisher_analysis/results/swimmer_width_sweep"
```

Keep:

```python
WIDTHS = None       # Load widths 4, 8, and 16 from config.json.
SHOW_NPG_DAMPING = True
NPG_DAMPING = 0.01 # Plot-only shift; it does not modify the saved Fisher.
DPI = 180
```

The notebook searches its working directory and parents for the repository
root, so it works when launched from either the repository root or the
`fisher_analysis` directory.

## 4. Expected outputs

The selected output directory receives:

- `raw_eigenspectrum.png`
- `trace_normalized_eigenspectrum.png`
- `cumulative_explained_trace.png`
- `damped_eigenspectrum.png` when `SHOW_NPG_DAMPING = True`

The notebook also displays the saved summary metrics, including numerical rank,
effective rank, condition number, and the component counts for 90%, 95%, and
99% of Fisher trace.

The plot files are overwritten when rerun. The source files `config.json`,
`summary.csv`, and `fisher_width_*.npz` are read only and are not modified.

## 5. Input readiness already checked

All three result directories currently contain:

- `config.json`
- `summary.csv`
- `iteration_stats.csv`
- `fisher_width_4.npz`
- `fisher_width_8.npz`
- `fisher_width_16.npz`

The notebook additionally checks that the matrices are finite and symmetric,
the eigenvalues are descending, and each Fisher trace matches the sum of its
eigenvalues. A failed check stops execution rather than silently drawing an
invalid plot.

## 6. Optional non-interactive execution

To execute the configured notebook without using the browser UI:

```powershell
.\.venv-analysis\Scripts\python.exe -m jupyter nbconvert `
  --to notebook --execute fisher_analysis\eigenvalue_analysis.ipynb `
  --output eigenvalue_analysis_executed.ipynb `
  --output-dir fisher_analysis
```

This creates an executed notebook copy and writes the same configured plot
files. Change the first configuration cell before running this command.

