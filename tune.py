import argparse
import json
import math
import os
import shutil
import tempfile

import gymnasium as gym
import numpy as np
import optuna
import torch

from data_collection import collect_parallel_trajectories
from train import run_single_training, make_env


# Seeds are spaced far apart (stride 10_000) so a trial's per-seed training-env range
# (seed + [0, n_envs)) and its evaluation-env range never collide with each other or across
# seeds. The number of tuning seeds is configurable: screening uses one seed, while the
# validation stage should use three.
_SEED_STRIDE = 10_000
_EVAL_SEED_OFFSET = _SEED_STRIDE // 2

_N_EVAL_ENVS = 8
_N_EVAL_EPISODES_PER_ENV = 2

# Keep incompatible objective/config definitions in separate persistent studies. Reusing an
# older study would mix scores from different grids and training budgets.
_OBJECTIVE_VERSION_BY_ALGORITHM = {
    "gpomdp": "v4_expanded_stable_grid_best_checkpoint_eval2",
    "npg": "v4_expanded_stable_grid_best_checkpoint_eval2",
}

_TARGET_ENVS = ("Swimmer-v5", "Hopper-v5", "HalfCheetah-v5")

# Grid search, not a continuous/Bayesian search: every parameter below is an explicit list.
# The full Cartesian product is intentionally larger than one search run. By default, the
# GridSampler samples 15 distinct points from it. NPG uses smaller learning rates and stronger
# damping than the previous grid because the old long run eventually produced non-finite
# policy means.
_LR_GRID = {
    "gpomdp": [3e-5, 1e-4, 3e-4, 1e-3],
    "npg": [1e-5, 3e-5, 1e-4, 3e-4],
}
_NPG_DAMPING_GRID = [0.1, 0.3, 1.0]
_INIT_LOG_STD_GRID = [-1.5, -1.0, -0.5]
_ENTROPY_COEFF_GRID = {
    "gpomdp": [0.0, 1e-3, 1e-2],
    "npg": [1e-3, 1e-2],
}
_GPOMDP_NORMALIZE_RETURNS_GRID = [False, True]

# Gamma is both environment- and algorithm-specific. NPG omits 0.999 on Hopper and
# HalfCheetah because its unconstrained update has already shown long-run instability there.
_GAMMA_CHOICES_BY_ALGORITHM_AND_ENV_PREFIX = {
    "gpomdp": {
        "Swimmer": [0.995, 0.999],
        "Hopper": [0.99, 0.995, 0.999],
        "HalfCheetah": [0.99, 0.995, 0.999],
    },
    "npg": {
        "Swimmer": [0.995, 0.999],
        "Hopper": [0.99, 0.995],
        "HalfCheetah": [0.99, 0.995],
    },
}
_DEFAULT_GAMMA_CHOICES = [0.95, 0.97, 0.99, 0.995, 0.999]  # fallback for envs not listed above


def _gamma_choices_for_env(env_id: str, algorithm: str):
    for prefix, choices in _GAMMA_CHOICES_BY_ALGORITHM_AND_ENV_PREFIX[algorithm].items():
        if env_id.startswith(prefix):
            return choices
    return _DEFAULT_GAMMA_CHOICES


def _grid_search_space(env_id: str, algorithm: str) -> dict:
    space = {
        "lr": _LR_GRID[algorithm],
        "gamma": _gamma_choices_for_env(env_id, algorithm),
        "init_log_std": _INIT_LOG_STD_GRID,
        "entropy_coeff": _ENTROPY_COEFF_GRID[algorithm],
    }
    if algorithm == "npg":
        space["npg_damping"] = _NPG_DAMPING_GRID
    else:
        space["normalize_returns"] = _GPOMDP_NORMALIZE_RETURNS_GRID
    return space


def parse_hidden_sizes(value: str):
    sizes = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("hidden_sizes must contain positive integers, e.g. '32,32'")
    return sizes


def _validate_args(args) -> None:
    for name in ("n_iterations", "n_envs", "n_trajectories", "n_tuning_seeds"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be greater than zero")
    if args.horizon < 0:
        raise ValueError("horizon must be zero (environment default) or a positive integer")
    if args.n_trials is not None and args.n_trials <= 0:
        raise ValueError("n_trials must be greater than zero when specified")


# Fixed, non-searched training settings. Each trial is long enough to expose learning behavior,
# but plots/videos are disabled. A best/final checkpoint pair is written only inside each
# trial's scratch directory and removed immediately after out-of-sample evaluation. Architecture
# and batch size are fixed rather than searched: for NPG in particular, the Fisher matrix is
# P x P in the parameter count, so varying network size would make trials wildly different
# costs and destabilize damping. NPG return normalization remains fixed on because the raw
# long-horizon return scale made its unconstrained update numerically diverge. GPOMDP return
# normalization and both algorithms' entropy coefficients are searched.
def _base_cfg(args, algorithm: str) -> dict:
    use_npg = algorithm == "npg"
    return {
        "env_id": args.env_id,

        "n_iterations": args.n_iterations,
        "n_envs": args.n_envs,
        "n_trajectories": args.n_trajectories,
        "horizon": args.horizon,

        "policy": "mlp",
        "hidden_sizes": args.hidden_sizes,
        "learn_std": True,

        "center_returns": True,
        "normalize_returns": use_npg,
        "returns_implementation": "recursive",
        "clip_actions": True,
        "entropy_coeff": 0.0,

        "use_npg": use_npg,

        "save_plots": False,
        "save_checkpoints": True,
        "record_video": False,
    }


def _suggest_cfg(trial: optuna.Trial, args, algorithm: str) -> dict:
    cfg = _base_cfg(args, algorithm)

    space = _grid_search_space(args.env_id, algorithm)
    cfg["lr"] = trial.suggest_categorical("lr", space["lr"])
    cfg["gamma"] = trial.suggest_categorical("gamma", space["gamma"])
    cfg["init_log_std"] = trial.suggest_categorical("init_log_std", space["init_log_std"])
    cfg["entropy_coeff"] = trial.suggest_categorical(
        "entropy_coeff", space["entropy_coeff"]
    )

    if algorithm == "npg":
        cfg["npg_damping"] = trial.suggest_categorical("npg_damping", space["npg_damping"])
    else:
        cfg["normalize_returns"] = trial.suggest_categorical(
            "normalize_returns", space["normalize_returns"]
        )
        cfg["npg_damping"] = 1e-2  # unused by run_single_training when use_npg is False

    return cfg


def _evaluate_policy(policy, cfg: dict, seed: int) -> float:
    """Runs a fresh batch of rollouts with the trained policy and returns the mean episode
    reward. Deliberately separate from the training reward curve: training rewards are the
    return of the exact stochastic batch the policy was just updated on, which is an optimistic,
    noisy estimate of how good the policy actually is -- this re-evaluates on freshly sampled
    episodes (still under the stochastic policy, but out-of-sample) with their own env seeds.
    """
    # Evaluation batch size is deliberately independent from the training batch size. Reducing
    # --n_envs to make a smoke test cheap should not silently change the objective definition.
    n_eval_envs = _N_EVAL_ENVS
    env_fns = [
        make_env(env_id=cfg["env_id"], seed=seed + i, horizon=cfg.get("horizon") or None)
        for i in range(n_eval_envs)
    ]
    eval_envs = gym.vector.AsyncVectorEnv(env_fns)
    try:
        trajectories = collect_parallel_trajectories(
            envs=eval_envs,
            policy=policy,
            n_trajectories_per_env=_N_EVAL_EPISODES_PER_ENV,
            clip_actions=cfg.get("clip_actions", True),
        )
    finally:
        eval_envs.close()

    return float(np.mean([sum(traj.rewards) for traj in trajectories]))


def _objective(trial: optuna.Trial, args, algorithm: str) -> float:
    cfg = _suggest_cfg(trial, args, algorithm)

    seed_scores = []
    for seed_idx in range(args.n_tuning_seeds):
        trial_seed = args.seed + seed_idx * _SEED_STRIDE
        cfg_i = dict(cfg, seed=trial_seed)

        trial_dir = tempfile.mkdtemp(prefix=f"optuna_trial_{trial.number}_seed{trial_seed}_")
        cfg_i["output_dir"] = trial_dir
        try:
            policy, _ = run_single_training(cfg_i)

            # run_single_training returns the final policy, but project runs save their highest
            # training-return policy as checkpoints/best.pt. Evaluate the same artifact a real
            # tuned run will use, otherwise late-training regressions can select the wrong grid
            # point (especially on the less stable MuJoCo tasks).
            best_checkpoint = os.path.join(trial_dir, "checkpoints", "best.pt")
            state_dict = torch.load(best_checkpoint, map_location="cpu", weights_only=True)
            policy.load_state_dict(state_dict)

            score = _evaluate_policy(policy, cfg_i, seed=trial_seed + _EVAL_SEED_OFFSET)
        finally:
            shutil.rmtree(trial_dir, ignore_errors=True)

        if not math.isfinite(score):
            # A diverged run (e.g. loss -> NaN from an unstable lr/damping combo) doesn't
            # always raise -- it just produces garbage rewards. Prune explicitly rather than
            # letting a NaN silently pollute the trial's average.
            raise optuna.TrialPruned(
                f"Non-finite evaluation score ({score}) at seed {trial_seed}"
            )

        seed_scores.append(score)

    mean_score = float(np.mean(seed_scores))
    trial.set_user_attr("seed_scores", seed_scores)
    trial.set_user_attr("score_std", float(np.std(seed_scores)))
    return mean_score


def _recommended_config(args, algorithm: str, best_params: dict) -> dict:
    """Build the complete config represented by the winning trial.

    Saving only Optuna's searched fields is unsafe in this project because config.json supplies
    different defaults for normalize_returns and entropy_coeff. This complete config can be
    inspected or passed directly to train.run_single_training without silently changing the
    conditions under which the parameters won.
    """
    cfg = _base_cfg(args, algorithm)
    cfg.update(best_params)
    cfg["seed"] = args.seed
    cfg["npg_damping"] = best_params.get("npg_damping", 1e-2)
    cfg["output_dir"] = os.path.join("runs", "tuned", args.env_id, algorithm)
    cfg["save_plots"] = True
    cfg["save_checkpoints"] = True
    return cfg


def _run_command(args, algorithm: str, best_params: dict) -> str:
    """Return a run.py command that overrides every fixed setting used during tuning."""
    hidden_sizes = ",".join(str(size) for size in args.hidden_sizes)
    lr_flag = "--lr_npg" if algorithm == "npg" else "--lr"
    normalize_returns = (
        "1" if best_params.get("normalize_returns", algorithm == "npg") else "0"
    )
    entropy_coeff = str(best_params["entropy_coeff"])
    parts = [
        ".\\.venv\\Scripts\\python.exe", "run.py",
        "--algorithm", algorithm,
        "--env_id", args.env_id,
        "--seed", str(args.seed),
        "--n_iterations", str(args.n_iterations),
        "--n_envs", str(args.n_envs),
        "--n_trajectories", str(args.n_trajectories),
        "--horizon", str(args.horizon),
        "--hidden_sizes", hidden_sizes,
        "--gamma", str(best_params["gamma"]),
        lr_flag, str(best_params["lr"]),
        "--init_log_std", str(best_params["init_log_std"]),
        "--learn_std", "1",
        "--center_returns", "1",
        "--normalize_returns", normalize_returns,
        "--returns_implementation", "recursive",
        "--clip_actions", "1",
        "--entropy_coeff", entropy_coeff,
        "--save_plots", "1",
        "--save_checkpoints", "1",
        "--record_video", "0",
        "--output_dir", os.path.join("runs", "tuned", args.env_id, algorithm),
    ]
    if algorithm == "npg":
        parts.extend(["--npg_damping", str(best_params["npg_damping"])])
    return " ".join(parts)


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search over GPOMDP or NPG training configs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--algorithm", type=str, required=True, choices=["gpomdp", "npg"],
        help="Which algorithm's hyperparameters to search.",
    )
    parser.add_argument(
        "--env_id", type=str, default="Swimmer-v5",
        help=(
            "Gymnasium environment ID. This tuner is configured for "
            f"{', '.join(_TARGET_ENVS)}; other IDs use the fallback gamma grid."
        ),
    )
    parser.add_argument(
        "--seed", type=int, default=23,
        help=(
            "Base seed. Additional tuning seeds are derived from this one; this also seeds "
            "the grid traversal order."
        ),
    )
    parser.add_argument(
        "--n_tuning_seeds", type=int, default=1,
        help="Training/evaluation seeds per grid point. Use 1 for search and 3 for validation.",
    )

    parser.add_argument(
        "--n_trials", type=int, default=15,
        help=(
            "Number of distinct grid points to run. The default is the 15-point screening "
            "phase. Pass the displayed full grid size to exhaust the Cartesian product."
        ),
    )
    parser.add_argument(
        "--n_iterations", type=int, default=2000,
        help=(
            "Policy-update cycles per seed. Every cycle collects n_envs * n_trajectories "
            "complete trajectories."
        ),
    )
    parser.add_argument(
        "--n_envs", type=int, default=8,
        help="Number of parallel environments per trial.",
    )
    parser.add_argument(
        "--n_trajectories", type=int, default=1,
        help="Episodes collected per environment per iteration.",
    )
    parser.add_argument(
        "--horizon", type=int, default=1000,
        help="Maximum episode length. Defaults to 1000, the standard episode length for the "
             "MuJoCo continuous-control envs this is ultimately tuned for. 0 uses the "
             "environment default.",
    )
    parser.add_argument(
        "--hidden_sizes", type=str, default="32,32",
        help="Fixed (not searched) hidden layer sizes, comma-separated.",
    )

    parser.add_argument(
        "--study_name", type=str, default=None,
        help=(
            "Optuna study base name. Defaults to '<algorithm>_<env_id>'; the objective-version "
            "suffix is added automatically so incompatible scores cannot be mixed."
        ),
    )
    parser.add_argument(
        "--storage", type=str, default=None,
        help=(
            "Optuna storage URL for a persistent/resumable study, e.g. "
            "'sqlite:///runs/optuna/study.db'. Defaults to in-memory (not resumable)."
        ),
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Where to save best_params.json. Defaults to runs/optuna/<env_id>/<algorithm>/.",
    )

    return parser


def run_search(args) -> dict:
    """Runs one full Optuna search (one env, one algorithm) and returns the result dict.

    `args` is whatever build_arg_parser() produces -- either from real CLI parsing, or built
    programmatically (e.g. by tune_all.py) as an argparse.Namespace with the same fields.
    `args.hidden_sizes` is expected as a comma-separated string, same as the CLI would give it.
    """
    if isinstance(args.hidden_sizes, str):
        args.hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    else:
        args.hidden_sizes = list(args.hidden_sizes)
        if not args.hidden_sizes or any(size <= 0 for size in args.hidden_sizes):
            raise ValueError("hidden_sizes must contain positive integers")
    _validate_args(args)

    objective_version = (
        f"{_OBJECTIVE_VERSION_BY_ALGORITHM[args.algorithm]}"
        f"__it{args.n_iterations}_seeds{args.n_tuning_seeds}"
        f"_envs{args.n_envs}_traj{args.n_trajectories}_h{args.horizon}"
    )
    study_base_name = args.study_name or f"{args.algorithm}_{args.env_id}"
    study_name = f"{study_base_name}__{objective_version}"
    output_dir = args.output_dir or os.path.join("runs", "optuna", args.env_id, args.algorithm)
    os.makedirs(output_dir, exist_ok=True)

    search_space = _grid_search_space(args.env_id, args.algorithm)
    grid_size = math.prod(len(v) for v in search_space.values())

    # Accept None for programmatic/backward-compatible callers that explicitly want the full
    # grid. CLI calls default to the 15-point screening pass.
    n_trials = args.n_trials if args.n_trials is not None else grid_size
    policy_updates_budget = n_trials * args.n_iterations * args.n_tuning_seeds
    trajectories_budget = (
        policy_updates_budget * args.n_envs * args.n_trajectories
    )
    max_environment_steps_budget = (
        trajectories_budget * (args.horizon or 1000)
    )
    print(f"Grid for {study_name}: {search_space} ({grid_size} combinations, running {n_trials})")
    print(
        f"Budget: {policy_updates_budget:,} policy updates, "
        f"{trajectories_budget:,} complete trajectories, "
        f"at most {max_environment_steps_budget:,} environment steps"
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=args.storage,
        direction="maximize",
        load_if_exists=args.storage is not None,
        sampler=optuna.samplers.GridSampler(search_space, seed=args.seed),
    )

    study.optimize(
        lambda trial: _objective(trial, args, args.algorithm),
        n_trials=n_trials,
        # Targeted at known numerical-failure modes rather than swallowing everything:
        # RuntimeError covers apply_npg_preconditioning's explicit singular-Fisher-matrix
        # error (gpomdp.py), ValueError covers invalid distribution parameters. A real bug
        # elsewhere should still surface instead of silently being logged as a failed trial.
        catch=(RuntimeError, ValueError),
    )

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if not completed:
        # study.best_value raises an opaque ValueError from inside optuna's storage layer if
        # every trial was pruned/failed (e.g. every hyperparameter combo diverged) -- surface
        # a message that actually says what happened instead.
        n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
        n_failed = sum(t.state == optuna.trial.TrialState.FAIL for t in study.trials)
        raise RuntimeError(
            f"No trial completed for {study_name}: {n_pruned} pruned (non-finite score), "
            f"{n_failed} failed (raised RuntimeError/ValueError), out of {len(study.trials)} "
            "total. Every hyperparameter combination diverged -- check n_iterations isn't too "
            "short for the search ranges in use, or widen/shift them."
        )

    print(f"\nBest value : {study.best_value:.3f}")
    print(f"Best params: {study.best_params}")

    recommended_config = _recommended_config(args, args.algorithm, study.best_params)
    run_command = _run_command(args, args.algorithm, study.best_params)
    best_seed_scores = study.best_trial.user_attrs.get("seed_scores", [])
    best_score_std = study.best_trial.user_attrs.get("score_std")

    result = {
        "algorithm": args.algorithm,
        "env_id": args.env_id,
        "study_name": study_name,
        "objective_version": objective_version,
        "grid_size": grid_size,
        "n_trials": n_trials,
        "n_trials_requested_this_run": n_trials,
        "n_trials_total": len(study.trials),
        "n_trials_completed": len(completed),
        "n_iterations": args.n_iterations,
        "n_tuning_seeds": args.n_tuning_seeds,
        "training_iterations_budget": policy_updates_budget,
        "policy_updates_budget": policy_updates_budget,
        "trajectories_budget": trajectories_budget,
        "max_environment_steps_budget": max_environment_steps_budget,
        "n_evaluation_episodes_per_seed": _N_EVAL_ENVS * _N_EVAL_EPISODES_PER_ENV,
        "best_value": study.best_value,
        "best_seed_scores": best_seed_scores,
        "best_score_std": best_score_std,
        "best_params": study.best_params,
        "recommended_config": recommended_config,
        "run_command": run_command,
    }
    result_path = os.path.join(output_dir, "best_params.json")
    with open(result_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved best params to: {result_path}")
    print(f"Run the winning configuration with:\n{run_command}")

    return result


def main():
    args = build_arg_parser().parse_args()
    run_search(args)


if __name__ == "__main__":
    main()
