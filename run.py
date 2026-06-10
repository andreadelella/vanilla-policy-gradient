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

        "eval_every": args.eval_every,
        "n_eval_episodes": args.n_eval_episodes,

        "save_plots": str_to_bool(args.save_plots),
        "record_video": str_to_bool(args.record_video),
    }


def main_run():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument("--dir", type=str, default="runs")
    parser.add_argument("--config_name", type=str, default="run_config.json")

    parser.add_argument(
        "--run_mode",
        type=str,
        default="single",
        choices=["single", "multiseed"],
    )

    parser.add_argument("--env_id", type=str, default="CartPole-v1")
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[23, 24, 25, 26, 27],
    )

    parser.add_argument("--n_iterations", type=int, default=2000)
    parser.add_argument("--n_envs", type=int, default=16)
    parser.add_argument("--n_trajectories", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=200)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lr", type=float, default=1e-4)

    parser.add_argument("--center_returns", type=int, default=1, choices=[0, 1])
    parser.add_argument("--normalize_returns", type=int, default=0, choices=[0, 1])
    parser.add_argument("--clip_actions", type=int, default=1, choices=[0, 1])

    parser.add_argument(
        "--hidden_sizes",
        type=str,
        default="8,8",
        help="Comma-separated hidden sizes, e.g. 8,8 or 64,64.",
    )
    parser.add_argument(
        "--hidden_dim",
        type=int,
        default=32,
        help="Used only by the current discrete MLP policy.",
    )

    parser.add_argument("--init_log_std", type=float, default=-0.5)
    parser.add_argument("--learn_std", type=int, default=1, choices=[0, 1])

    parser.add_argument("--eval_every", type=int, default=5)
    parser.add_argument("--n_eval_episodes", type=int, default=5)

    parser.add_argument("--save_plots", type=int, default=1, choices=[0, 1])
    parser.add_argument("--record_video", type=int, default=0, choices=[0, 1])

    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)

    cfg = build_config(args)

    config_path = os.path.join(args.dir, args.config_name)

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