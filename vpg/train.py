import contextlib
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import gymnasium as gym
import numpy as np
import torch

from vpg.artifacts import load_config, save_training_rewards
from vpg.diagnostics import (
    append_metrics_row as _append_metrics_row,
    freeze_distribution as _frozen_distribution,
    gradient_norm as _grad_norm,
    measure_kl as _measure_kl,
    policy_stats as _policy_stats,
)
from vpg.plotting import plot_training_curves
from vpg.video import record_policy_video
from vpg.data_collection import collect_parallel_trajectories
from vpg.policy import build_policy
from vpg.gpomdp import (
    compute_gpomdp_loss,
    apply_npg_preconditioning,
    compute_discounted_returns_matrix,
    trajectories_to_tensors,
)


class _Tee:
    """Duplicate writes to several streams (used to mirror stdout into a log file)."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()


@contextlib.contextmanager
def _tee_stdout(path, tee=True):
    """Redirect output to a log, optionally mirroring it to the console."""
    if path is None:
        yield
        return
    with open(path, "w") as log_file:
        original_out, original_err = sys.stdout, sys.stderr
        # _Tee flushes on every write, so a killed worker still leaves a complete log.
        sys.stdout = _Tee(original_out, log_file) if tee else _Tee(log_file)
        sys.stderr = _Tee(original_err, log_file) if tee else _Tee(log_file)
        try:
            yield
        finally:
            sys.stdout, sys.stderr = original_out, original_err


def resolve_device(name=None) -> torch.device:
    """Resolve a device string to a torch.device.

    'auto' (or None) prefers CUDA, then Apple MPS, then CPU.
    """
    if name in (None, "auto"):
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def make_env(env_id: str, seed: int | None, horizon: int | None = None):
    # AsyncVectorEnv requires a factory (thunk) rather than an env instance
    # so each worker can construct its own isolated copy.
    def thunk():
        env = gym.make(env_id)

        if horizon is not None and horizon > 0:
            env = gym.wrappers.TimeLimit(
                env,
                max_episode_steps=horizon,
            )

        if seed is not None:
            env.reset(seed=seed)

        return env

    return thunk


def run_single_training(cfg: dict):
    """Train and persist one independently seeded policy."""
    output_dir = cfg.get("output_dir", "runs")
    os.makedirs(output_dir, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    device = resolve_device(cfg.get("device", "auto"))
    print(f"Device       : {device}")

    env = gym.make(cfg["env_id"])

    policy = build_policy(cfg, env)
    policy.to(device)

    if cfg.get("use_npg", False):
        optimizer = torch.optim.SGD(policy.parameters(), lr=cfg["lr"])
    else:
        optimizer = torch.optim.Adam(policy.parameters(), lr=cfg["lr"])

    env_fns = [
        make_env(
            env_id=cfg["env_id"],
            seed=cfg["seed"] + i,
            horizon=cfg.get("horizon", None),
        )
        for i in range(cfg["n_envs"])
    ]

    # Rollout collection is the dominant wall-clock cost; N workers run concurrently to amortise it.
    train_envs = gym.vector.AsyncVectorEnv(env_fns)

    training_rewards = []
    best_reward = float("-inf")
    best_state_dict = None

    save_checkpoints = cfg.get("save_checkpoints", True)
    policy_dir = os.path.join(output_dir, "policy")
    live_best_path = os.path.join(policy_dir, "best.pt")
    if save_checkpoints:
        os.makedirs(policy_dir, exist_ok=True)

    # Periodic full-policy snapshots, independent of the best/final policy above.
    checkpoint_interval = cfg.get("checkpoint_interval", 500)
    snapshot_dir = os.path.join(output_dir, "checkpoints")
    save_snapshots = save_checkpoints and checkpoint_interval > 0
    if save_snapshots:
        os.makedirs(snapshot_dir, exist_ok=True)

    # Per-iteration diagnostics, written incrementally so a run that is killed or
    # collapses still leaves a complete record up to that point.
    log_metrics = cfg.get("log_metrics", True)
    metrics_path = os.path.join(output_dir, "metrics.csv")
    if log_metrics and os.path.exists(metrics_path):
        # Stale rows from an earlier run would be indistinguishable from this one's.
        os.remove(metrics_path)

    training_start = time.perf_counter()

    try:
        for iteration in range(cfg["n_iterations"]):
            t0 = time.perf_counter()

            trajectories = collect_parallel_trajectories(
                envs=train_envs,
                policy=policy,
                n_trajectories_per_env=cfg["n_trajectories"],
                clip_actions=cfg.get("clip_actions", True),
                device=device,
            )

            t1 = time.perf_counter()

            batch_reward = float(np.mean([
                sum(traj.rewards) for traj in trajectories
            ]))

            # `batch_reward` was generated by the current (pre-update) policy.
            # Preserve those exact weights when this is the best observed batch.
            if batch_reward > best_reward:
                best_reward = batch_reward
                best_state_dict = {
                    key: value.detach().cpu().clone()
                    for key, value in policy.state_dict().items()
                }
                if save_checkpoints:
                    torch.save(best_state_dict, live_best_path)

            debug = iteration == 0

            # Snapshot the pre-update distribution at this batch's states so the
            # KL of the step can be measured directly after optimizer.step().
            old_dist = None
            flat_states = None
            return_mean = return_std = float("nan")
            if log_metrics:
                states, _, raw_rewards, mask = trajectories_to_tensors(trajectories, device=device)
                valid = mask.reshape(-1).bool()
                flat_states = states.reshape(-1, states.shape[-1])[valid]
                old_dist = _frozen_distribution(policy, flat_states)

                valid_returns = compute_discounted_returns_matrix(
                    rewards=raw_rewards,
                    gamma=cfg["gamma"],
                    implementation=cfg.get("returns_implementation", "recursive"),
                ).reshape(-1)[valid]
                return_mean = float(valid_returns.mean())
                return_std = float(valid_returns.std())

            optimizer.zero_grad()

            loss = compute_gpomdp_loss(
                policy=policy,
                trajectories=trajectories,
                gamma=cfg["gamma"],
                center_returns=cfg["center_returns"],
                normalize_returns=cfg["normalize_returns"],
                entropy_coeff=cfg.get("entropy_coeff", 0.0),
                returns_implementation=cfg.get("returns_implementation", "recursive"),
                device=device,
                debug=debug,
            )

            loss.backward()

            grad_norm = _grad_norm(policy) if log_metrics else float("nan")

            if cfg.get("use_npg", False):
                # Note: re-derives states/actions/mask from `trajectories` internally
                # (trajectories_to_tensors runs again here) rather than reusing the ones
                # already built inside compute_gpomdp_loss above -- redundant but not incorrect.
                apply_npg_preconditioning(
                    policy=policy,
                    trajectories=trajectories,
                    damping=cfg.get("npg_damping", 1e-2),
                    device=device,
                    debug=debug,
                )

            # After preconditioning .grad holds F^-1 g; this is the step SGD applies.
            nat_grad_norm = _grad_norm(policy) if log_metrics else float("nan")

            optimizer.step()

            t2 = time.perf_counter()

            rollout_time = t1 - t0
            update_time = t2 - t1
            iteration_time = t2 - t0

            n_steps = sum(len(traj.rewards) for traj in trajectories)
            samples_per_sec = n_steps / iteration_time

            training_rewards.append(batch_reward)

            kl = entropy = mean_std = float("nan")
            if log_metrics:
                kl = _measure_kl(policy, flat_states, old_dist)
                entropy, mean_std = _policy_stats(policy, flat_states)
                _append_metrics_row(metrics_path, {
                    "iteration": iteration,
                    "train_reward": round(batch_reward, 4),
                    "best_reward": round(best_reward, 4),
                    "kl": f"{kl:.6e}",
                    "grad_norm": f"{grad_norm:.6e}",
                    "nat_grad_norm": f"{nat_grad_norm:.6e}",
                    "entropy": round(entropy, 6),
                    "mean_std": round(mean_std, 6),
                    "return_mean": round(return_mean, 4),
                    "return_std": round(return_std, 4),
                    "episode_len": round(n_steps / len(trajectories), 2),
                    "rollout_time": round(rollout_time, 4),
                    "update_time": round(update_time, 4),
                    "total_time": round(iteration_time, 4),
                })

            print(
                f"Iteration {iteration:04d} | "
                f"train reward: {batch_reward:.2f} | "
                f"best: {best_reward:.2f} | "
                f"KL: {kl:.3e} | "
                f"std: {mean_std:.4f} | "
                f"rollout: {rollout_time:.3f}s | "
                f"update: {update_time:.3f}s | "
                f"total: {iteration_time:.3f}s | "
                f"samples/s: {samples_per_sec:.0f}"
            )

            if save_snapshots and (iteration + 1) % checkpoint_interval == 0:
                snapshot_path = os.path.join(snapshot_dir, f"snapshot_iter_{iteration + 1:06d}.pt")
                torch.save(policy.state_dict(), snapshot_path)

            # Overwrite the same archive periodically so interrupted runs retain data.
            if checkpoint_interval > 0 and (iteration + 1) % checkpoint_interval == 0:
                save_training_rewards(
                    output_dir,
                    np.array([training_rewards], dtype=np.float32),
                    [cfg["seed"]],
                )

    finally:
        train_envs.close()
        env.close()

    training_time = time.perf_counter() - training_start
    print(f"training time {training_time:.2f}s")

    save_training_rewards(
        output_dir,
        np.array([training_rewards], dtype=np.float32),
        [cfg["seed"]],
    )

    if cfg.get("save_plots", True):
        plot_training_curves(
            training_rewards=training_rewards,
            save_dir=output_dir,
        )

    if save_checkpoints:
        scored = cfg.get("scored_checkpoints", False)
        if best_state_dict is not None:
            best_name = f"best_{best_reward:.1f}.pt" if scored else "best.pt"
            best_path = os.path.join(policy_dir, best_name)
            if best_path != live_best_path:
                os.replace(live_best_path, best_path)
        final_score = training_rewards[-1] if training_rewards else 0.0
        final_name = f"final_{final_score:.1f}.pt" if scored else "final.pt"
        torch.save(policy.state_dict(), os.path.join(policy_dir, final_name))

    if cfg.get("record_video", False):
        # Video recording feeds CPU state tensors to the policy, so run it on CPU.
        policy.to("cpu")
        # Restore the best weights found during training before recording.
        if best_state_dict is not None:
            policy.load_state_dict(best_state_dict)
        record_policy_video(
            env_id=cfg["env_id"],
            policy=policy,
            video_dir=os.path.join(output_dir, "videos"),
            seed=cfg["seed"] + 20_000,
        )

    return policy, training_rewards


def _record_best_seed_video(cfg, output_dir, all_training_rewards, seeds, final_window=100):
    """Record a video from the best-performing seed's checkpoint.

    "Best" = highest mean return over the last `final_window` iterations. The seed's
    weights are loaded from output_dir/seed<seed>/policy/best.pt (written during
    the multiseed loop) and replayed into output_dir/videos/.
    """
    rewards = np.asarray(all_training_rewards, dtype=np.float64)
    window = min(final_window, rewards.shape[1])
    scores = rewards[:, -window:].mean(axis=1)
    best_idx = int(np.argmax(scores))
    best_seed = seeds[best_idx]
    print(f"\nBest seed: {best_seed} (final-{window} mean {scores[best_idx]:.1f}) — recording video")

    checkpoint_path = os.path.join(output_dir, f"seed{best_seed}", "policy", "best.pt")
    if not os.path.exists(checkpoint_path):
        print(f"Warning: no checkpoint at {checkpoint_path}; skipping video")
        return

    # Replay runs on CPU (record_policy_video builds CPU state tensors).
    env = gym.make(cfg["env_id"])
    policy = build_policy(cfg, env)
    env.close()

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    policy.load_state_dict(state_dict)
    policy.eval()

    record_policy_video(
        env_id=cfg["env_id"],
        policy=policy,
        video_dir=os.path.join(output_dir, "videos"),
        seed=int(best_seed) + 20_000,
    )


def _train_one_seed(cfg: dict, seed: int, output_dir: str, tee: bool = True):
    """Train one seed in an isolated output directory."""
    seed_cfg = dict(cfg)
    seed_cfg["seed"] = seed
    seed_cfg["record_video"] = False   # never record during per-seed training
    seed_cfg["save_plots"] = False     # CI plot is produced downstream in the notebook

    # Every seed gets its own subdir for policy/, checkpoints/, and rewards, so seeds
    # never clobber each other and the best seed can be replayed after all finish.
    # save_checkpoints is inherited from cfg (respects --save_checkpoints); only the
    # filenames are forced deterministic so the best-seed video lookup always works.
    seed_cfg["scored_checkpoints"] = False
    seed_cfg["output_dir"] = os.path.join(output_dir, f"seed{seed}")
    os.makedirs(seed_cfg["output_dir"], exist_ok=True)

    # Mirror this seed's console output into its own directory, so one seed's
    # progress is not interleaved with the others in a single combined log.
    log_path = (
        os.path.join(seed_cfg["output_dir"], "train.log")
        if cfg.get("log_metrics", True)
        else None
    )
    with _tee_stdout(log_path, tee=tee):
        _, seed_rewards = run_single_training(seed_cfg)
    return seed_rewards


def _seed_worker(packed):
    cfg, seed, output_dir = packed
    try:
        rewards = _train_one_seed(cfg, seed, output_dir, tee=False)
        return seed, rewards, None
    except BaseException:  # report any failure back to the parent, don't crash the pool
        return seed, None, traceback.format_exc()


def _resolve_seed_workers(seed_workers, n_seeds: int, n_envs: int) -> int:
    """Resolve an explicit or automatic number of concurrent seed workers."""
    if seed_workers in (None, "auto"):
        cores = os.cpu_count() or 1
        target = max(1, (2 * cores) // max(1, n_envs))
        return max(1, min(n_seeds, target))
    n = int(seed_workers)
    return min(max(n, 1), n_seeds)


def _run_seeds_sequential(cfg: dict, seeds, output_dir: str):
    all_training_rewards = []
    for seed in seeds:
        print(f"\n========== Running seed {seed} ==========")
        seed_rewards = _train_one_seed(cfg, seed, output_dir, tee=True)
        all_training_rewards.append(seed_rewards)

        # One uniquely-named file per seed so each seed's curve is identifiable on disk.
        per_seed_path = save_training_rewards(
            output_dir,
            np.array([seed_rewards], dtype=np.float32),
            [seed],
            filename=f"training_rewards_seed{seed}.npz",
        )
        print(f"Saved per-seed rewards: {per_seed_path}")

        # Rewrite the combined matrix after every seed, so an interrupted multiseed run
        # still leaves a usable file covering the seeds that did finish. Seeds that ran
        # a different number of iterations cannot share one array, so only stack the
        # equal-length prefix group; the per-seed files above always hold the full curves.
        completed = seeds[:len(all_training_rewards)]
        if len({len(r) for r in all_training_rewards}) == 1:
            save_training_rewards(
                output_dir,
                np.asarray(all_training_rewards, dtype=np.float32),
                completed,
            )
    return all_training_rewards, list(seeds)


def _run_seeds_parallel(cfg: dict, seeds, output_dir: str, n_workers: int):
    """Train seeds in spawned processes and preserve requested seed order."""
    print(f"\n========== Training {len(seeds)} seeds, {n_workers} at a time ==========")
    print("Each seed's console output is redirected to its seed<N>/train.log")

    rewards_by_seed = {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
        futures = {
            ex.submit(_seed_worker, (cfg, seed, output_dir)): seed
            for seed in seeds
        }
        for fut in as_completed(futures):
            seed, seed_rewards, err = fut.result()
            if err is not None:
                print(f"[seed {seed}] FAILED -- other seeds continue:\n{err}")
                continue
            rewards_by_seed[seed] = seed_rewards
            per_seed_path = save_training_rewards(
                output_dir,
                np.array([seed_rewards], dtype=np.float32),
                [seed],
                filename=f"training_rewards_seed{seed}.npz",
            )
            print(f"[seed {seed}] done ({len(rewards_by_seed)}/{len(seeds)}) -> {per_seed_path}")

    # Reassemble in the requested seed order so downstream row i still maps to seeds[i];
    # any seed that failed is simply absent from both returned lists.
    completed_seeds = [s for s in seeds if s in rewards_by_seed]
    ordered_rewards = [rewards_by_seed[s] for s in completed_seeds]
    return ordered_rewards, completed_seeds


def run_multiseed(cfg: dict):
    seeds = cfg.get("seeds", [cfg["seed"]])
    output_dir = cfg.get("output_dir", "runs")
    os.makedirs(output_dir, exist_ok=True)

    want_video = cfg.get("record_video", False)

    n_workers = _resolve_seed_workers(
        cfg.get("seed_workers", 1), len(seeds), cfg.get("n_envs", 1)
    )

    if n_workers <= 1:
        all_training_rewards, completed_seeds = _run_seeds_sequential(cfg, seeds, output_dir)
    else:
        all_training_rewards, completed_seeds = _run_seeds_parallel(
            cfg, seeds, output_dir, n_workers
        )

    if not all_training_rewards:
        print("No seeds completed successfully; skipping combined outputs.")
        return

    # Final combined matrix (labeled) for the comparison notebook. Seeds that ran a
    # different number of iterations cannot share one array; guard the stack (all seeds
    # share cfg, so lengths match unless one was interrupted). Per-seed files above
    # always hold the full curves regardless.
    if len({len(r) for r in all_training_rewards}) == 1:
        combined = np.asarray(all_training_rewards, dtype=np.float32)
        save_training_rewards(output_dir, combined, completed_seeds)
        if want_video:
            _record_best_seed_video(cfg, output_dir, combined, completed_seeds)
    else:
        print("Seeds ran different iteration counts; combined matrix skipped "
              "(per-seed files saved).")


def train_from_config(config_path="config.json"):
    cfg = load_config(config_path)
    run_mode = cfg.get("run_mode", "single")

    if run_mode == "single":
        return run_single_training(cfg)

    if run_mode == "multiseed":
        return run_multiseed(cfg)

    raise ValueError(f"Unknown run_mode: {run_mode}")


if __name__ == "__main__":
    train_from_config()
