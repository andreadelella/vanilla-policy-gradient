import argparse
import json
import os

from vpg.train import train_from_config

# Comment key: M says what the function does. A says how it works and why.


#M: Converts a command-line 0 or 1 into a Python Boolean.
#A: Changes the value to an integer first, then converts it to True or False.
def str_to_bool(x):
    return bool(int(x))


#M: Converts text such as "32,32" into a list of hidden-layer sizes.
#A: Splits at commas, converts each part to an integer, and rejects invalid sizes.
def parse_hidden_sizes(value: str):
    sizes = [int(v.strip()) for v in value.split(",") if v.strip()]
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError("hidden_sizes must contain positive integers, e.g. '32,32'")
    return sizes


#M: Builds the training configuration from the parsed command-line arguments.
#A: Collects all settings in one dictionary and converts special values to their final types.
def build_config(args):
    return {
        "run_mode": args.run_mode,
        "device": args.device,

        "env_id": args.env_id,
        "seed": args.seed,
        "seeds": args.seeds,

        "n_iterations": args.n_iterations,
        "n_envs": args.n_envs,
        "n_trajectories": args.n_trajectories,
        "horizon": args.horizon,

        "gamma": args.gamma,
        "returns_implementation": args.returns_implementation,
        "lr": args.lr,
        "lr_npg": args.lr_npg,

        "center_returns": str_to_bool(args.center_returns),
        "normalize_returns": str_to_bool(args.normalize_returns),
        "clip_actions": str_to_bool(args.clip_actions),

        "hidden_sizes": parse_hidden_sizes(args.hidden_sizes),
        "policy": args.policy,

        "init_log_std": args.init_log_std,
        "learn_std": str_to_bool(args.learn_std),

        "save_plots": str_to_bool(args.save_plots),
        "save_checkpoints": str_to_bool(args.save_checkpoints),
        "checkpoint_interval": args.checkpoint_interval,
        "log_metrics": str_to_bool(args.log_metrics),
        "record_video": str_to_bool(args.record_video),

        "npg_damping": args.npg_damping,
        "entropy_coeff": args.entropy_coeff,
    }


# Keys computed by run.py — must not be overridden by config.json.
_SKIP_FROM_FILE = {"output_dir", "scored_checkpoints"}

# Config keys stored as JSON booleans but argparse expects a 0/1 int.
_BOOL_KEYS = {
    "center_returns", "normalize_returns", "clip_actions",
    "learn_std", "save_plots", "save_checkpoints", "record_video",
}


#M: Uses values from config.json as command-line defaults.
#A: Loads the file, converts values to argparse's expected format, and lets explicit CLI values override them.
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


#M: Prints the main environment, batch, and algorithm settings before training.
#A: Reads the resolved configuration and shows the values most useful for checking a run.
def _print_algo_header(cfg):
    print(f"Environment  : {cfg['env_id']}")
    print(f"Mode         : {cfg['run_mode']}")
    print(f"Horizon      : {cfg['horizon']}")
    print(f"Batch        : {cfg['n_envs']} envs × {cfg['n_trajectories']} traj")
    if cfg.get("use_npg", False):
        print(f"Algorithm    : NPG  (SGD, lr={cfg['lr']}, damping={cfg['npg_damping']})")
    else:
        print(f"Algorithm    : GPOMDP (Adam, lr={cfg['lr']})")


#M: Reads command-line options and starts a training run.
#A: Builds the parser, merges defaults, saves the final config, and passes it to train.py.
def main():
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
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help=(
            "Compute device for the policy and gradient math. 'auto' prefers CUDA, "
            "then Apple MPS, then CPU. Note: environment rollouts always run on CPU "
            "(MuJoCo/gym); the GPU only helps when the policy network is large."
        ),
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
        "--returns_implementation",
        type=str,
        default="recursive",
        choices=["recursive", "vectorized"],
        help=(
            "Discounted-return backend. 'recursive' is numerically safe; "
            "'vectorized' uses a guarded float64 powers/reverse-cumsum formulation."
        ),
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

    parser.add_argument(
        "--algorithm",
        type=str,
        default="gpomdp",
        choices=["gpomdp", "npg"],
        help="Algorithm to run. 'gpomdp' uses Adam; 'npg' uses natural gradient.",
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
        help=(
            "Hidden layer sizes, comma-separated (e.g. '64,64'). Used by GaussianPolicy "
            "(continuous envs) and MLPSoftmaxPolicy (discrete envs, --policy mlp). "
            "Ignored when --policy linear."
        ),
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="mlp",
        choices=["mlp", "linear"],
        help=(
            "Policy architecture for discrete action spaces only ('mlp' or 'linear'). "
            "Continuous envs always use GaussianPolicy."
        ),
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
        help="Save best/final policy weights to output_dir/policy/ and periodic snapshots to output_dir/checkpoints/.",
    )
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=500,
        help="Save a policy snapshot to output_dir/checkpoints/ every N iterations. 0 disables periodic snapshots. Ignored if --save_checkpoints 0.",
    )
    parser.add_argument(
        "--log_metrics",
        type=int,
        default=1,
        choices=[0, 1],
        help="Write per-iteration diagnostics (KL, gradient norms, entropy, action std, "
             "return stats) to <seed_dir>/metrics.csv and mirror the console output to "
             "<seed_dir>/train.log. Adds one extra forward pass per iteration.",
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

    algorithm = args.algorithm

    # Auto-generate output dir when the user did not specify one.
    output_dir_specified = args.output_dir is not None
    output_dir = args.output_dir if output_dir_specified else os.path.join("runs", args.env_id)

    cfg = build_config(args)
    cfg["use_npg"] = (algorithm == "npg")
    if algorithm == "npg" and cfg["lr_npg"] is not None:
        cfg["lr"] = cfg["lr_npg"]
    cfg["output_dir"] = output_dir
    cfg["scored_checkpoints"] = not output_dir_specified

    os.makedirs(output_dir, exist_ok=True)
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Saved config to: {config_path}")
    _print_algo_header(cfg)

    train_from_config(config_path)


if __name__ == "__main__":
    main()
