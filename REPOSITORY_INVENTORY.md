# Repository inventory for the university deliverable

Status: inventory plus requested index cleanup. `exploration/` is ignored and
has been removed from the Git index, while its complete local filesystem tree
has been retained. No git tag, commit, or push has been created.

This inventory adapts `plan.md` to the existing repository. In particular,
`vpg/` remains the canonical university-project package; there is no reason to
rename it to `src/`.

## 1. Current safety state

The working tree is not ready for a freeze tag or a reorganization commit.
It contains existing modified and untracked research files and many tracked
deletions inside historical sampled-tabular result trees. These changes predate
this inventory and must be classified deliberately before any move or archival
commit.

The requested index removal now appears as staged deletions until it is
eventually committed. Until the repository scope is finalized:

- do not tag the repository;
- do not commit a repository-wide snapshot;
- do not move or delete the local exploration result directories;
- do not use destructive git commands;
- do not modify `vpg/` as part of the reorganization.

## 2. Canonical university core

Classification: `CORE`

The complete core already lives in `vpg/`:

| Path | Canonical responsibility |
|---|---|
| `vpg/policy.py` | categorical and Gaussian policies; policy construction |
| `vpg/data_collection.py` | trajectory collection and rollout data |
| `vpg/gpomdp.py` | discounted returns, GPOMDP loss, empirical Fisher, NPG preconditioning |
| `vpg/train.py` | canonical training loop, logging, checkpoints, evaluation diagnostics |
| `vpg/run.py` | training CLI and configuration assembly |
| `vpg/stats.py` | statistical summaries and confidence intervals |
| `vpg/analysis.py` | post-hoc reward analysis |
| `vpg/plotting.py` | training plots |
| `vpg/video.py` | policy video generation |
| `vpg/env_wrappers.py` | environment wrappers |
| `vpg/tune.py`, `vpg/tune_all.py` | Optuna tuning workflows; useful but not part of the minimal final path |

Canonical mathematical objects already implemented in `vpg/gpomdp.py`:

- trajectory-averaged GPOMDP/REINFORCE loss;
- empirical Fisher from valid transition scores,
  `F = S.T @ S / M + damping * I`;
- NPG through a linear solve rather than an explicit inverse.

Decision: preserve this package and its public behavior. Any final runner or
analysis layer should import `vpg`; it should not duplicate the core.

## 3. Existing top-level entry points

| Path | Classification | Decision |
|---|---|---|
| `run.py` | `CORE` | keep as the canonical thin training entry point |
| `analysis.py` | `FINAL_ANALYSIS` | keep as the canonical reward-analysis wrapper |
| `tune.py` | `OBSOLETE_BUT_PRESERVE` for delivery | retain, but omit from the primary reproduction path |
| `tune_all.py` | `OBSOLETE_BUT_PRESERVE` for delivery | retain, but omit from the primary reproduction path |
| `config.json` | `CORE`, currently generic | do not treat as the final frozen experiment config |

The missing final entry point is a small reproducibility wrapper, provisionally
`experiments/reproduce_final.py`, which should call the existing core rather
than replace it.

## 4. Fisher analysis

Classification: `FINAL_ANALYSIS` and candidate `FINAL_RESULT`

| Path | Role |
|---|---|
| `fisher_analysis/run_fisher_analysis.py` | fixed-policy empirical-Fisher estimation and eigenspectrum analysis |
| `fisher_analysis/eigenvalue_analysis.ipynb` | interactive reproduction of saved spectra |
| `fisher_analysis/tests/` | Fisher-analysis verification |
| `fisher_analysis/results/*_width_sweep/` | current Hopper, HalfCheetah, and Swimmer spectral artifacts |

This package is already independent of training and explicitly checks that the
policy is unchanged during analysis. It is the strongest candidate for the
school-facing spectral-analysis component.

Before delivery, its output schema should be checked against the desired final
columns: trace, minimum and maximum eigenvalues, numerical rank, condition
number, damped condition number, effective rank, and log determinant.

Observed documentation issue: the top-level README links to
`fisher_analysis/fisher_analysis.md`, but that file is not currently present in
the filesystem inventory. The link must be restored or changed later.

## 5. Final log-barrier experiment candidate

Classification: `FINAL_EXPERIMENT_CANDIDATE`, not yet promoted.

The most relevant existing neural implementations are under:

- `exploration/neural_discrete_log_barrier/`;
- `exploration/npg_logbarrier_factorial/`.

They include Acrobot reliability studies, GPOMDP/barrier training, NPG/barrier
factorial experiments, Fisher diagnostics, handoff controls, and reporting.
They remain exploratory until one exact four-cell comparison is selected:

| Optimizer | Regularization |
|---|---|
| GPOMDP | none |
| GPOMDP | declared log barrier |
| NPG | none |
| NPG | the same declared log barrier |

Promotion must copy or wrap only the selected tested implementation. Historical
oracle, handoff, entropy, reliability, and ablation variants remain in
`exploration/`.

Open decision before promotion: state precisely whether the university-facing
regularizer is the sampled conditional categorical barrier, a damped Fisher
log determinant, or another spectral scalar. The final report and runner must
use exactly the same definition.

## 6. Exploration archive

Classification: `EXPLORATION` or `OBSOLETE_BUT_PRESERVE`

All of the following remain locally under `exploration/`, but none are now
tracked by Git:

- `categorical_bandit/`: exact categorical identity and stochastic-bandit work;
- `tabular_mdp/`: exact two-state transition-pooled geometry;
- `sampled_tabular_mdp/`: finite-batch estimator audits and handoff studies;
- `neural_discrete_log_barrier/`: Acrobot neural experiments and reliability work;
- `npg_logbarrier_factorial/`: sampled/exact NPG and barrier factorial studies;
- `joint_state_action_fisher/`: policy/state/joint Fisher theory, arbitrary-depth chains, critical-beta results, and Arb/Krawczyk certification;
- `fisher_log_barrier/`: planning/history if present in the checked-out branch;
- `exploration/results/`: historical numerical artifacts;
- `exploration/tests/`: tests for exploratory packages;
- experiment catalog and report-digest scripts/CSVs.

The arbitrary-depth and certification material is research-facing and must not
be required to reproduce the university result.

Index-cleanup verification:

- tracked paths below `exploration/`: `0`;
- local files retained below `exploration/`: `32,299`;
- local result files retained below `exploration/results/`: `32,057`;
- approximate local exploration size: `2.26 GB`.

## 7. Reports and documentation

| Path | Classification | Current state |
|---|---|---|
| `README.md` | `REPORT` | documents the original GPOMDP/NPG project and fixed-policy Fisher analysis; needs a later delivery rewrite |
| `report/README.md` | `REPORT` | small report-directory guide |
| `report/log_barrier_analysis_plan.md` | `EXPLORATION`/research planning | not the final university report |
| `exploration/**/*.tex` | `EXPLORATION` | step-specific technical reports and experiment records |

There is currently no top-level final `report/report.tex` or bibliography in
the inventory. These are future delivery tasks, not Phase-B moves.

## 8. Tests

| Path | Classification |
|---|---|
| `tests/test_analysis.py` | `TEST`, core reward-analysis behavior |
| `fisher_analysis/tests/test_fisher_analysis.py` | `TEST`, fixed-policy empirical Fisher and CLI |
| `exploration/tests/` | `EXPLORATION_TEST`, preserve with research packages |
| `exploration/joint_state_action_fisher/test_rigorous_certification.py` | `EXPLORATION_TEST`, proof-package replay checks |

The university core still needs explicit focused tests for policy probability
normalization, GPOMDP reproducibility, the empirical-Fisher toy identity, NPG
damping/linear solve behavior, and the selected log-barrier implementation.

## 9. Results classification

| Current location | Classification | Proposed treatment |
|---|---|---|
| `fisher_analysis/results/` | candidate `FINAL_RESULT` | retain; select only report-used artifacts for the final manifest |
| `runs/`, `runs_new/`, `output/` | mixed `FINAL_RESULT`/historical | inventory experiment-by-experiment before selecting anything |
| `exploration/results/` | `EXPLORATION` | preserve in place; never make it a dependency of the final runner |
| `tmp/` | likely `OBSOLETE_BUT_PRESERVE` until inspected | do not delete during reorganization |

No result is promoted merely because it exists. A final result must have an
explicit configuration, seeds, raw data, metric definitions, and a reproducible
figure-generation path.

## 10. Revised target structure

The least disruptive target keeps the existing core:

```text
vanilla-policy-gradient/
|-- vpg/                         # unchanged canonical implementation
|-- run.py                       # existing training entry point
|-- analysis.py                  # existing reward analysis
|-- experiments/
|   |-- reproduce_final.py       # future thin orchestrator
|   `-- configs/final.json
|-- fisher_analysis/             # canonical spectral analysis
|-- results/final/               # curated final numerical artifacts
|-- report/                      # final report plus figures
|-- tests/                       # university-core tests
`-- exploration/                 # preserved research archive
```

Creating `src/` would duplicate or rename a sensible existing package and is
therefore explicitly rejected.

## 11. Canonical-selection decisions still required

Before any code promotion or movement:

1. Select the primary environment and optional validation environment.
2. Select the exact four final method configurations and their tested source
   implementation.
3. Freeze the log-barrier definition and coefficient naming.
4. Decide whether the fixed-random-policy width sweep is a main result or a
   background spectral analysis.
5. Select report-relevant existing runs, if any, from `runs/`, `runs_new/`, and
   the neural exploration result tree.
6. Reconcile the currently deleted historical sampled-tabular artifacts.

The log-barrier implementation has not yet been promoted. A repository-wide
search confirms there is currently no tracked non-exploration implementation
of the categorical log barrier. The final university extraction must therefore
happen before the university branch can be considered reproducible. The local
source remains available at:

- `exploration/neural_discrete_log_barrier/barrier.py`;
- `exploration/npg_logbarrier_factorial/acrobot.py`;
- `exploration/npg_logbarrier_factorial/natural_step.py`.

The factorial Acrobot runner also depends on neural training/evaluation helpers
and a saved reference-state bank, so copying only `acrobot.py` would create a
hidden dependency. Promotion must extract a self-contained minimal package.

## 12. Safe next sequence

1. Review this inventory and make the six selection decisions above.
2. Extract the selected categorical barrier and final four-method runner into a
   self-contained tracked package outside `exploration/`.
3. Reconcile the remaining dirty tree without discarding local historical results.
4. Write a final experiment specification and immutable config.
5. Add the canonical spectral-summary pipeline.
6. Run the selected final matrix and curate `results/final/`.
7. Update the report and top-level README.
8. Perform clean-room reproduction.
9. Only after explicit approval: commit, tag, and push.

## 13. Actions deliberately not taken

- no edits inside `vpg/`;
- no file or result moves;
- no deletions or restoration of existing deletions;
- no dependency changes;
- no experiment execution;
- the requested Git-index removal of `exploration/` was performed without
  deleting its local files;
- no git tag;
- no git commit;
- no git push.
