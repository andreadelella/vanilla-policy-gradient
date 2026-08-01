import os

# Rollout is multiprocessing-IPC-bound for cheap envs (e.g. Swimmer): the policy
# is tiny, so BLAS/OpenMP multithreading gives no speedup and 32 worker processes
# each spawning threads on a many-core box thrash each other. Pinning every process
# to a single BLAS/OpenMP thread removed that oversubscription and cut rollout time
# by ~20% on this workload with zero change to results.
#
# These must be set before numpy/torch/BLAS are first imported -- they read the
# thread counts at init. This package __init__ runs before any vpg submodule (and
# thus before their `import torch`), and AsyncVectorEnv workers inherit the env, so
# setting it here covers every entry point. setdefault lets an explicit shell
# override still win.
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
