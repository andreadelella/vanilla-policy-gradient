import argparse
import json
import os

from train import main


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
    }


# Keys computed by run.py — must not be overridden by config.json.
_SKIP_FROM_FILE = {"output_dir", "scored_checkpoints"}

# Config keys stored as JSON booleans but argparse expects a 0/1 int.
_BOOL_KEYS = {
    "center_returns", "normalize_returns", "clip_actions",
    "learn_std", "save_plots", "save_checkpoints", "record_video",
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
        help="Adam learning rate.",
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

    # Auto-generate output dir when the user did not specify one.
    # The scored_checkpoints flag makes checkpoint filenames include the return value.
    output_dir_specified = args.output_dir is not None
    output_dir = args.output_dir if output_dir_specified else os.path.join("runs", args.env_id)

    os.makedirs(output_dir, exist_ok=True)

    cfg = build_config(args)
    cfg["output_dir"] = output_dir
    cfg["scored_checkpoints"] = not output_dir_specified

    config_path = os.path.join(output_dir, "config.json")

    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Saved config to: {config_path}")
    print(f"Running mode: {cfg['run_mode']}")
    print(f"Environment: {cfg['env_id']}")
    print(f"Horizon: {cfg['horizon']}")
    print(f"Batch trajectories: {cfg['n_envs']} x {cfg['n_trajectories']}")

    main(config_path)


if __name__ == "__main__":
    main_run()