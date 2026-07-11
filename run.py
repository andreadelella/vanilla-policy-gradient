import argparse
import json
import os

import numpy as np

from train import main
from utils import plot_comparison


def str_to_bool(x):
    return bool(int(x))


def parse_hidden_sizes(value: str):
    return [int(v) for v in value.split(",")]


def build_config(args):
    return {
        "run_mode": args.run_mode,

        "env_id": args.env_id,
        "seed": args.seed,
        "seeds": args.seeds,

        "n_iterations": args.n_iterations,
        "n_envs": args.n_envs,
        "n_trajectories": args.n_trajectories,
        "horizon": args.horizon,

        "gamma": args.gamma,
        "lr": args.lr,
        "lr_npg": args.lr_npg,

        "center_returns": str_to_bool(args.center_returns),
        "normalize_returns": str_to_bool(args.normalize_returns),
        "clip_actions": str_to_bool(args.clip_actions),

        "hidden_sizes": parse_hidden_sizes(args.hidden_sizes),
        "hidden_dim": args.hidden_dim,

        "init_log_std": args.init_log_std,
        "learn_std": str_to_bool(args.learn_std),

        "save_plots": str_to_bool(args.save_plots),
        "save_checkpoints": str_to_bool(args.save_checkpoints),
        "record_video": str_to_bool(args.record_video),

        "use_npg": str_to_bool(args.use_npg),
        "npg_damping": args.npg_damping,
        "entropy_coeff": args.entropy_coeff,
    }


# Keys computed by run.py — must not be overridden by config.json.
_SKIP_FROM_FILE = {"output_dir", "scored_checkpoints"}

# Config keys stored as JSON booleans but argparse expects a 0/1 int.
_BOOL_KEYS = {
    "center_returns", "normalize_returns", "clip_actions",
    "learn_std", "save_plots", "save_checkpoints", "record_video", "use_npg",
}


def _apply_file_config(parser, config_path="config.json"):
    """Promote config.json values to argparse defaults.

    Priority: hardcoded defaults < config.json < CLI arguments.
    """
    if not os.path.exists(config_path):
        return

    with open(config_path) as f:
        file_cfg = json.load(f)

    overrides = {}
    for k, v in file_cfg.items():
        if k in _SKIP_FROM_FILE:
            continue
        if k == "hidden_sizes":
            overrides[k] = ",".join(str(x) for x in v)
        elif k in _BOOL_KEYS:
            overrides[k] = int(bool(v))
        else:
            overrides[k] = v

    parser.set_defaults(**overrides)


def _print_algo_header(cfg):
    print(f"Environment  : {cfg['env_id']}")
    print(f"Mode         : {cfg['run_mode']}")
    print(f"Horizon      : {cfg['horizon']}")
    print(f"Batch        : {cfg['n_envs']} envs × {cfg['n_trajectories']} traj")
    if cfg.get("use_npg", False):
        print(f"Algorithm    : NPG  (SGD, lr={cfg['lr']}, damping={cfg['npg_damping']})")
    else:
        print(f"Algorithm    : GPOMDP (Adam, lr={cfg['lr']})")


def main_run():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory where all outputs are saved. Defaults to runs/<env_id>/.",
    )
    parser.add_argument(
        "--run_mode",
        type=str,
        default="single",
        choices=["single", "multiseed"],
        help="'single' trains one seed; 'multiseed' loops over --seeds and produces CI plots.",
    )

    parser.add_argument(
        "--env_id",
        type=str,
        default="CartPole-v1",
        help="Gymnasium environment ID. Continuous spaces use a Gaussian policy; discrete use softmax MLP.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=23,
        help="Random seed for single-seed runs.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[23, 24, 25, 26, 27],
        help="List of seeds for multiseed runs.",
    )

    parser.add_argument(
        "--n_iterations",
        type=int,
        default=2000,
        help="Number of policy gradient update steps.",
    )
    parser.add_argument(
        "--n_envs",
        type=int,
        default=16,
        help="Number of parallel environments. Total batch size = n_envs * n_trajectories.",
    )
    parser.add_argument(
        "--n_trajectories",
        type=int,
        default=1,
        help="Episodes collected per environment per iteration.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=200,
        help="Maximum episode length. 0 uses the environment default.",
    )

    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor γ ∈ (0, 1]. Controls down-weighting of future rewards.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for GPOMDP (Adam).",
    )
    parser.add_argument(
        "--lr_npg",
        type=float,
        default=None,
        help="Learning rate for NPG (SGD). Falls back to --lr if not set.",
    )

    # --algorithms is the primary way to select algorithm(s).
    # --use_npg is kept for backward compatibility with config.json.
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=None,
        choices=["gpomdp", "npg"],
        help=(
            "Algorithm(s) to run. 'gpomdp' uses Adam; 'npg' uses natural gradient. "
            "Specifying both (--algorithms gpomdp npg) triggers a comparison run: "
            "each algorithm is trained in its own subdirectory and a joint CI plot is saved."
        ),
    )
    parser.add_argument(
        "--use_npg",
        type=int,
        default=0,
        choices=[0, 1],
        help=argparse.SUPPRESS,  # legacy; prefer --algorithms
    )
    parser.add_argument(
        "--npg_damping",
        type=float,
        default=1e-2,
        help="Tikhonov damping λ added to the Fisher diagonal: (F + λI)⁻¹. Increase if solve fails.",
    )
    parser.add_argument(
        "--entropy_coeff",
        type=float,
        default=0.01,
        help="Entropy bonus coefficient. Adds entropy_coeff * H[pi] to the objective to prevent policy collapse.",
    )

    parser.add_argument(
        "--center_returns",
        type=int,
        default=1,
        choices=[0, 1],
        help="Subtract mean return (baseline trick). Reduces gradient variance without bias.",
    )
    parser.add_argument(
        "--normalize_returns",
        type=int,
        default=0,
        choices=[0, 1],
        help="Divide returns by their std. Further reduces variance; can destabilize early training.",
    )
    parser.add_argument(
        "--clip_actions",
        type=int,
        default=1,
        choices=[0, 1],
        help="Clip continuous actions to the environment's action bounds before stepping.",
    )

    parser.add_argument(
        "--hidden_sizes",
        type=str,
        default="8,8",
        help="Hidden layer sizes for the Gaussian policy (continuous envs). Comma-separated, e.g. '64,64'.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Hidden layer size for the softmax MLP policy (discrete envs).",
    )

    parser.add_argument(
        "--init_log_std",
        type=float,
        default=-0.5,
        help="Initial log std of the Gaussian policy (σ ≈ 0.6). Controls initial exploration noise.",
    )
    parser.add_argument(
        "--learn_std",
        type=int,
        default=1,
        choices=[0, 1],
        help="1: log_std is a learnable parameter. 0: kept fixed at init_log_std.",
    )

    parser.add_argument(
        "--save_plots",
        type=int,
        default=1,
        choices=[0, 1],
        help="Save reward plots to output_dir.",
    )
    parser.add_argument(
        "--save_checkpoints",
        type=int,
        default=1,
        choices=[0, 1],
        help="Save best and final policy weights to output_dir/checkpoints/.",
    )
    parser.add_argument(
        "--record_video",
        type=int,
        default=0,
        choices=[0, 1],
        help="Record a video of the trained policy and save to output_dir/videos/.",
    )

    # Apply config.json values as defaults before parsing CLI arguments.
    _apply_file_config(parser)

    args = parser.parse_args()

    # Resolve which algorithm(s) to run.
    # --algorithms takes priority; fall back to legacy --use_npg / config.json use_npg.
    if args.algorithms is not None:
        algorithms = args.algorithms
    else:
        algorithms = ["npg"] if str_to_bool(args.use_npg) else ["gpomdp"]

    # Auto-generate output dir when the user did not specify one.
    output_dir_specified = args.output_dir is not None
    output_dir = args.output_dir if output_dir_specified else os.path.join("runs", args.env_id)

    comparing = len(algorithms) > 1

    if not comparing:
        # -- Single algorithm --------------------------------------------------
        algo = algorithms[0]
        cfg = build_config(args)
        cfg["use_npg"] = (algo == "npg")
        if algo == "npg" and cfg["lr_npg"] is not None:
            cfg["lr"] = cfg["lr_npg"]
        cfg["output_dir"] = output_dir
        cfg["scored_checkpoints"] = not output_dir_specified

        os.makedirs(output_dir, exist_ok=True)
        config_path = os.path.join(output_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(cfg, f, indent=2)

        print(f"Saved config to: {config_path}")
        _print_algo_header(cfg)

        main(config_path)

    else:
        # -- Comparison mode ---------------------------------------------------
        mode_label = args.run_mode
        seeds_label = str(args.seeds) if args.run_mode == "multiseed" else str([args.seed])
        print(f"Comparing    : {' vs '.join(a.upper() for a in algorithms)}")
        print(f"Environment  : {args.env_id}  |  mode: {mode_label}  |  seeds: {seeds_label}")
        print(f"Output       : {output_dir}/")

        for i, algo in enumerate(algorithms, 1):
            algo_dir = os.path.join(output_dir, algo)
            os.makedirs(algo_dir, exist_ok=True)

            algo_cfg = build_config(args)
            algo_cfg["use_npg"] = (algo == "npg")
            if algo == "npg" and algo_cfg["lr_npg"] is not None:
                algo_cfg["lr"] = algo_cfg["lr_npg"]
            algo_cfg["output_dir"] = algo_dir
            algo_cfg["scored_checkpoints"] = False

            config_path = os.path.join(algo_dir, "config.json")
            with open(config_path, "w") as f:
                json.dump(algo_cfg, f, indent=2)

            print(f"\n{'-'*55}")
            print(f"  [{i}/{len(algorithms)}] {algo.upper()}")
            print(f"{'-'*55}")
            print(f"Saved config to: {config_path}")
            _print_algo_header(algo_cfg)

            main(config_path)

        # Collect saved reward arrays and produce the comparison plot.
        rewards = {}
        for algo in algorithms:
            npy_path = os.path.join(output_dir, algo, "training_rewards.npy")
            if os.path.exists(npy_path):
                rewards[algo] = np.load(npy_path)
            else:
                print(f"Warning: missing {npy_path} — skipping {algo} from comparison plot")

        if len(rewards) > 1:
            plot_comparison(rewards, save_dir=output_dir, env_id=args.env_id)


if __name__ == "__main__":
    main_run()
