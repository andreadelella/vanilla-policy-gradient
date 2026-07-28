# Fixed-Policy Fisher Eigenspectrum Analysis

## Purpose

This experiment measures the local behavioral dimensionality of randomly
initialized Gaussian policies. It asks a compression-oriented question:

> Although a policy has \(P\) trainable parameters, how many parameter-space
> directions account for most of the policy's local change in behavior?

We evaluated fixed two-hidden-layer policies on `Hopper-v5`,
`HalfCheetah-v5`, and `Swimmer-v5` with widths 4, 8, and 16. All three use 32
asynchronously stepped environments and four trajectories per environment to
reduce sampling error. No policy was trained. Each policy was initialized
once, held fixed, and evaluated on fresh on-policy trajectories.

This isolates architecture and policy-distribution geometry from the effects
of optimization. The result is a baseline description of the local geometry
present at random initialization, not a comparison of learned performance.

## Plain-language summary

The experiment asks how many independent ways a policy can meaningfully change
its behavior. A neural policy may have hundreds of parameters, but many
parameter combinations can have almost the same effect or almost no effect on
the action distribution.

For each fixed random policy, we:

1. ran fresh trajectories without training the policy;
2. measured how every parameter affected every sampled action probability;
3. combined those sensitivities into a Fisher information matrix;
4. decomposed the matrix into directions ordered from most to least important;
5. counted how many directions explained 90%, 95%, and 99% of total
   sensitivity.

A rapidly falling eigenspectrum means that most local behavior can be
described by far fewer directions than the network has parameters. The main
90%-trace results are:

| Environment | Policy parameters across widths | Directions explaining 90% |
|---|---:|---:|
| Hopper | 86 to 518 | 6 to 10 |
| HalfCheetah | 128 to 668 | 32 to 70 |
| Swimmer | 68 to 452 | 6 to 7 |

Swimmer is the most spectrally concentrated. Hopper is similarly concentrated,
although its random policies terminate early and therefore produce fewer
samples per trajectory. HalfCheetah uses a broader set of meaningful
directions, although that set is still much smaller than its parameter count.

This identifies local compression potential; it does not itself create a
smaller neural network. A separate projection, low-rank parameterization,
pruning, or distillation step would be needed to turn the spectral result into
deployable model compression.

## Mathematical perspective

### Score vectors and the empirical Fisher

For policy parameters \(\theta \in \mathbb{R}^P\), state \(s_i\), and action
\(a_i\), the per-sample score vector is

\[
g_i = \nabla_\theta \log \pi_\theta(a_i \mid s_i).
\]

Given \(M\) valid on-policy state/action samples, the undamped empirical Fisher
matrix is

\[
\widehat{F}
=
\frac{1}{M}
\sum_{i=1}^{M} g_i g_i^\top
=
\frac{1}{M} S^\top S,
\]

where row \(i\) of \(S \in \mathbb{R}^{M \times P}\) is \(g_i^\top\).

The matrix is positive semidefinite because, for any parameter perturbation
\(\delta\),

\[
\delta^\top \widehat{F}\delta
=
\frac{1}{M}
\sum_i (g_i^\top \delta)^2
\geq 0.
\]

Therefore, the Fisher does not measure reward curvature. It measures how
strongly the policy's action probabilities or densities respond to local
parameter changes on the states visited by the current policy.

### Connection to local policy change

The Fisher is the second-order local metric induced by the policy's KL
divergence. For a small perturbation \(\delta\),

\[
\mathbb{E}_{s}
\left[
D_{\mathrm{KL}}
\left(
\pi_\theta(\cdot \mid s)
\;\|\;
\pi_{\theta+\delta}(\cdot \mid s)
\right)
\right]
\approx
\frac{1}{2}\delta^\top F\delta.
\]

This gives the eigenvalues a direct interpretation. Let

\[
F = Q \Lambda Q^\top,
\qquad
\Lambda = \operatorname{diag}(\lambda_1,\ldots,\lambda_P),
\qquad
\lambda_1 \geq \cdots \geq \lambda_P \geq 0.
\]

For eigenvector \(q_j\), a perturbation \(\delta = \alpha q_j\) produces the
local KL change

\[
\frac{1}{2}\alpha^2\lambda_j.
\]

- A large \(\lambda_j\) identifies a sensitive direction: a small movement
  changes the policy distribution substantially.
- A small \(\lambda_j\) identifies a locally insensitive direction: a similar
  movement changes the observed policy behavior very little.
- A rapidly decaying spectrum means that local behavioral variation is
  concentrated in a subspace much smaller than the raw parameter space.

### The compression lens

A rank-\(k\) approximation keeps the leading Fisher directions:

\[
F_k = Q_k \Lambda_k Q_k^\top.
\]

The retained fraction of Fisher trace is

\[
R_k
=
\frac{\sum_{j=1}^{k}\lambda_j}
{\sum_{j=1}^{P}\lambda_j}.
\]

Under this criterion, \(k\) is a local behavioral compression dimension. A
small \(k\) with \(R_k \approx 1\) means that most infinitesimal policy
sensitivity can be represented in a low-dimensional parameter subspace.
Discarding the remaining directions loses little Fisher trace on the sampled
state distribution.

This is a sensitivity-weighted notion of compression. It differs from:

- compressing weights by magnitude;
- counting algebraically nonzero parameters;
- measuring the rank of individual network layers;
- proving that a smaller network can reproduce the same policy globally.

The Fisher eigenvectors are generally dense combinations of parameters across
layers. Consequently, a low-dimensional Fisher spectrum suggests opportunities
for low-rank updates, Fisher-aware pruning, or distillation, but it does not by
itself provide a deployable smaller network.

### Reported spectral statistics

The analysis reports:

- **Trace:** \(\operatorname{tr}(F)=\sum_j\lambda_j\), the total sampled local
  sensitivity.
- **Numerical rank:** the number of eigenvalues above
  \(P\,\epsilon_{64}\max_j|\lambda_j|\).
- **Effective rank:** entropy-based spectral dimensionality,

  \[
  r_{\mathrm{eff}}
  =
  \exp\left(-\sum_j p_j\log p_j\right),
  \qquad
  p_j = \frac{\lambda_j}{\sum_l\lambda_l},
  \]

  using eigenvalues above the rank tolerance.
- **Stable rank:** spectral concentration measured by

  \[
  r_{\mathrm{stable}}
  =
  \frac{\sum_j\lambda_j^2}{\lambda_1^2}.
  \]

- **Positive-spectrum condition number:**
  \(\lambda_1/\lambda_{\min,+}\), where \(\lambda_{\min,+}\) is the smallest
  eigenvalue above the numerical-rank tolerance.
- **Components for 90%, 95%, and 99% trace:** the smallest \(k\) for which
  \(R_k\) reaches the corresponding threshold.

Numerical rank and effective rank answer different questions. A matrix may be
numerically full rank while still having a small effective rank because many
positive eigenvalues contribute almost no trace.

## Experimental design

All experiments share the following settings:

| Setting | Value |
|---|---:|
| Policy widths | 4, 8, 16 |
| Hidden layers | 2 |
| Iterations | 10 |
| Horizon | 200 |
| Base seed | 23 |
| Fisher damping | 0 |
| Fisher dtype | float64 |

The rollout budgets are:

| Environment | Parallel environments | Trajectories per environment and iteration | Maximum samples per width |
|---|---:|---:|---:|
| `Hopper-v5` | 32 | 4 | 256,000 |
| `HalfCheetah-v5` | 32 | 4 | 256,000 |
| `Swimmer-v5` | 32 | 4 | 256,000 |

For observation dimension \(d_s\), action dimension \(d_a\), and hidden width
\(w\), the two-hidden-layer Gaussian policy has

\[
P
=
(d_s w+w)
+(w^2+w)
+(w d_a+d_a)
+d_a.
\]

The final term is the learned state-independent `log_std`. The resulting
dimensions are:

| Environment | \(d_s\) | \(d_a\) | Width 4 | Width 8 | Width 16 |
|---|---:|---:|---:|---:|---:|
| Hopper | 11 | 3 | 86 | 198 | 518 |
| HalfCheetah | 17 | 6 | 128 | 276 | 668 |
| Swimmer | 8 | 2 | 68 | 164 | 452 |

Each width receives a deterministic but distinct policy seed. Environment
resets and action samples also have deterministic seed schedules. Iterations
produce independent rollout batches, and all valid samples from all iterations
are pooled into one Fisher estimate per width.

Hopper can terminate before the 200-step horizon when it reaches an unhealthy
state. Therefore, the Hopper widths produced different sample counts.
HalfCheetah and Swimmer do not normally terminate early under these settings,
so every one of their width runs used the full 256,000 samples.

## Code perspective

The implementation is in
[`run_fisher_analysis.py`](run_fisher_analysis.py).

### 1. Policy construction

The analysis reuses the repository's policy builder from
[`policy.py`](../policy.py):

- All three continuous-control environments use `GaussianPolicy`.
- The MLP produces the state-dependent Gaussian mean.
- `log_std` is a learned, state-independent parameter vector.
- Every width uses two hidden layers of that width.
- The policy is moved to CPU float64 and put in evaluation mode.

The complete initial `state_dict` is cloned before collecting data. Exact
tensor equality and empty `.grad` fields are checked after every rollout batch
and after Fisher construction.

### 2. Fixed-policy trajectory collection

For every iteration:

1. A Gymnasium `AsyncVectorEnv` steps all environment workers in parallel.
2. Each environment is reset with its deterministic episode seed.
3. The fixed policy samples fresh stochastic actions from its current
   distribution.
4. Continuous actions are clipped to the environment's action bounds before
   `env.step`.
5. The unclipped Gaussian samples are retained for likelihood evaluation.
6. Finite state/action pairs are pooled across environments and trajectories.

Keeping the raw action is important. The score must be evaluated at the random
variable sampled from the Gaussian:

\[
\nabla_\theta \log \pi_\theta(a_{\mathrm{raw}}\mid s).
\]

The clipped action is only an environment command. Treating it as an ordinary
Gaussian sample would assign the wrong likelihood to probability mass that was
collapsed onto the action-space boundary.

No objective, optimizer, `.backward()` call, or parameter update appears in
this collection and analysis path.

### 3. Per-sample score calculation

The implementation uses PyTorch functional transforms:

- `functional_call` evaluates the policy with an explicit parameter mapping.
- `torch.func.grad` differentiates one scalar sample log probability.
- `vmap` evaluates those score gradients over a sample batch.

For each chunk, the named parameter gradients are flattened in
`named_parameters()` order to form a score matrix. The code accumulates

\[
S_{\mathrm{chunk}}^\top S_{\mathrm{chunk}}
\]

in float64 and divides by the total sample count only after all chunks have
been processed. Chunking avoids retaining the entire \(M \times P\) score
matrix. Score tensors are detached before accumulation, so the resulting
Fisher carries no higher-order autograd graph.

No damping term is added. This is deliberate: damping would replace each
eigenvalue by \(\lambda_j+\delta\), hiding the intrinsic small-eigenvalue tail
that is central to the compression analysis.

### 4. Numerical validation

Before saving results, the implementation verifies:

- the matrix is square and finite;
- symmetry holds within a float64 scale-aware tolerance;
- the smallest eigenvalue satisfies a numerical PSD tolerance;
- eigenvalues are stored in descending order;
- the matrix trace equals the sum of eigenvalues;
- policy parameters and buffers exactly match their initial values;
- no `.grad` fields were populated.

The original matrix and eigenvalues are saved without clipping. Only values at
the numerical tolerance floor are clamped for logarithmic plotting.

### 5. Outputs

The result directories are:

- [`results/hopper_width_sweep/`](results/hopper_width_sweep/)
- [`results/halfcheetah_width_sweep/`](results/halfcheetah_width_sweep/)
- [`results/swimmer_width_sweep/`](results/swimmer_width_sweep/)

Each directory contains:

- `config.json`: complete experiment and seed configuration;
- `iteration_stats.csv`: rollout counts, returns, lengths, and timings;
- `summary.csv`: spectral statistics for every width;
- `fisher_width_<width>.npz`: original Fisher, eigenvalues, normalized and
  cumulative spectra, parameter layout, sample counts, and tolerances;
- `checkpoints/policy_width_<width>.pt`: the fixed policy state dictionaries;
- `raw_eigenspectrum.png`;
- `trace_normalized_eigenspectrum.png`;
- `cumulative_explained_trace.png`.

[`eigenvalue_analysis.ipynb`](eigenvalue_analysis.ipynb)
reloads these files, reproduces the plots, and optionally shows how NPG
diagonal damping shifts the spectrum and condition number. Its single
configuration cell selects the result directory, an optional subset of saved
widths, output directory, plot titles, damping value, and image resolution.
All relative paths are resolved from the repository root.

## Results

All nine empirical Fishers are numerically full rank. Their effective ranks
and trace thresholds show that full numerical rank does not imply that all
directions carry comparable local policy sensitivity.

| Environment | Width | Parameters | Samples | Trace | Effective rank | Stable rank | Condition number | PCs for 90% | PCs for 95% | PCs for 99% |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hopper | 4 | 86 | 29,389 | 41.104 | 6.16 | 1.56 | \(2.49\times10^8\) | 6 | 8 | 15 |
| Hopper | 8 | 198 | 11,549 | 35.023 | 7.30 | 1.86 | \(1.56\times10^{10}\) | 7 | 9 | 15 |
| Hopper | 16 | 518 | 21,107 | 47.157 | 9.93 | 1.94 | \(9.63\times10^9\) | 10 | 13 | 25 |
| HalfCheetah | 4 | 128 | 256,000 | 138.748 | 36.13 | 3.90 | \(1.48\times10^5\) | 32 | 40 | 57 |
| HalfCheetah | 8 | 276 | 256,000 | 175.847 | 48.23 | 4.18 | \(4.36\times10^5\) | 45 | 60 | 100 |
| HalfCheetah | 16 | 668 | 256,000 | 159.042 | 76.13 | 9.18 | \(6.84\times10^5\) | 70 | 105 | 198 |
| Swimmer | 4 | 68 | 256,000 | 15.861 | 7.12 | 2.36 | \(8.92\times10^7\) | 6 | 8 | 13 |
| Swimmer | 8 | 164 | 256,000 | 23.231 | 5.72 | 1.74 | \(1.50\times10^9\) | 6 | 8 | 13 |
| Swimmer | 16 | 452 | 256,000 | 29.309 | 8.19 | 2.35 | \(2.87\times10^{10}\) | 7 | 9 | 15 |

### Hopper

The leading component explains 39.3%, 32.3%, and 28.7% of trace for widths 4,
8, and 16. The leading ten components explain 97.4%, 97.1%, and 90.6%.
Only 6 to 10 directions explain 90% of trace, indicating strong local
spectral compression. The 32-by-4 setup increased the pooled sample counts by
roughly 15 to 16 times over the original Hopper run. Hopper still has fewer
samples than the other environments because its random policies terminate
early.

### HalfCheetah

HalfCheetah has a substantially broader spectrum. Its leading component
explains only 10.0%, 8.9%, and 4.7% of trace as width increases. The leading
ten components explain 51.3%, 46.6%, and 33.4%. Reaching 90% requires 32, 45,
and 70 directions.

HalfCheetah is still compressible relative to raw parameter count: the
width-16 policy needs 70 of 668 directions for 90% trace. However, its
effective rank grows from 36.13 to 76.13, showing that its local policy
geometry uses many more independent directions than Hopper or Swimmer.

### Swimmer

Swimmer remains sharply concentrated even with 256,000 samples per width.
Only 6, 6, and 7 directions explain 90% of trace, while 13, 13, and 15 explain
99%. The leading ten components explain 97.6%, 98.0%, and 97.7%.

The width-8 effective rank being lower than width 4 is possible because each
width is a distinct random policy with a distinct state distribution. Fisher
intrinsic dimension need not increase monotonically for one initialization.

### Cross-environment compression interpretation

The comparison shows that spectral compressibility is environment- and
occupancy-dependent, not just a function of network width:

- Swimmer has the strongest concentration: 6 to 7 directions retain 90%.
- Hopper is similarly concentrated, but early termination still makes its
  precise spectral tail less certain than the 256,000-sample runs.
- HalfCheetah has a broader policy-sensitivity subspace: 32 to 70 directions
  are needed for 90%.

All environments still use far fewer 90%-trace directions than parameters.
As width grows, the absolute number of meaningful directions generally grows,
but much more slowly than parameter count. This supports low-rank
Fisher-aware updates or local policy projections, while not yet proving that
the underlying neural networks can be structurally reduced to those sizes.

The condition numbers also differ materially. HalfCheetah's
\(1.5\times10^5\) to \(6.8\times10^5\) range is ill-conditioned but much less
extreme than Swimmer and Hopper. Swimmer reaches \(2.9\times10^{10}\), so an
undamped natural-gradient solve would strongly amplify its weakest
directions. NPG damping remains necessary even when the Fisher is estimated
from 256,000 samples.

## What the analysis does not establish

These results should not be read as proof that a policy with as many
parameters as its 90% or 99% component count can replace the corresponding
network.

1. **The result is local.** The Fisher describes infinitesimal perturbations
   around one parameter vector, not large parameter movements or global policy
   equivalence.
2. **The result is state-distribution dependent.** It only measures states
   visited by each random policy. A trained policy or a different environment
   occupancy can have a different spectrum.
3. **Each width uses one initialization.** Differences between widths may
   partly reflect the selected random policies. Multiple initialization seeds
   are needed for confidence intervals.
4. **Sample precision differs across environments.** Early Hopper termination
   produced 29,389, 11,549, and 21,107 samples despite the same 32-by-4 rollout
   budget. HalfCheetah and Swimmer each used 256,000 samples per width, but
   temporal and within-trajectory correlation means these are not 256,000
   independent state draws.
5. **The Fisher is not a reward Hessian.** High Fisher sensitivity does not
   imply high reward importance, and low sensitivity does not guarantee that a
   large finite perturbation is harmless.
6. **Dense spectral directions are not structural compression.** Turning this
   observation into a smaller deployable network requires a mechanism such as
   low-rank parameterization, structured pruning, projection, or distillation.

## Why this analysis is useful

The experiment establishes a clean baseline for subsequent compression work:

- It quantifies the local intrinsic dimension before training.
- It identifies how much Fisher geometry a rank-\(k\) approximation would
  retain.
- It motivates low-rank approximations to natural-gradient or trust-region
  updates.
- It shows why damping is required when inverting the Fisher.
- It provides a reproducible way to compare spectral concentration across
  policy widths.

A stronger compression study would repeat this analysis across initialization
seeds and training checkpoints, evaluate whether the leading eigenspaces are
stable, compress or project the policy using those directions, and then
measure both KL divergence and task return after compression.

## Verification performed

The implementation and generated artifacts were checked at several levels:

- A small Gaussian-policy Fisher was compared against a manual
  `torch.autograd.grad` calculation.
- Matrix shape, float64 dtype, symmetry, PSD tolerance, descending
  eigenvalues, trace equality, and rank metrics were tested.
- Policy state dictionaries were compared exactly before and after collection
  and Fisher construction, and `.grad` fields remained empty.
- A CartPole CLI smoke test verified configs, checkpoints, CSV files, NumPy
  archives, and plots.
- A repeated seeded smoke run produced identical Fisher arrays, eigenspectra,
  metadata, checkpoints, and rollout statistics apart from wall-clock timing.
- The analysis notebook was executed headlessly against smoke-test results.
- Every Hopper, HalfCheetah, and Swimmer matrix was independently reloaded and
  checked for symmetry, PSD tolerance, trace equality, parameter dimension,
  checkpoint size, and the expected pooled sample count.
- All four automated tests passed after the asynchronous collector was added.

## Reproduction

Run the high-resource Hopper experiment with:

```bash
.venv/bin/python -m fisher_analysis.run_fisher_analysis
```

Run the matching HalfCheetah and Swimmer experiments with:

```bash
.venv/bin/python -m fisher_analysis.run_fisher_analysis \
  --env-id HalfCheetah-v5 \
  --n-envs 32 \
  --trajectories-per-env 4 \
  --output-dir fisher_analysis/results/halfcheetah_width_sweep

.venv/bin/python -m fisher_analysis.run_fisher_analysis \
  --env-id Swimmer-v5 \
  --n-envs 32 \
  --trajectories-per-env 4 \
  --output-dir fisher_analysis/results/swimmer_width_sweep
```

The test suite includes a manual autograd comparison, matrix and metric checks,
policy immutability checks, and a CartPole CLI smoke test:

```bash
MPLCONFIGDIR=/tmp/fisher-mpl \
  .venv/bin/python -m unittest discover -s fisher_analysis/tests -v
```
