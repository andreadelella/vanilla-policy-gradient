import contextlib
import csv
import json
import multiprocessing as mp
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed

import gymnasium as gym
import numpy as np
import torch
from torch.distributions import kl_divergence

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

# Comment key: M says what the function does. A says how it works and why.


#M: Loads training settings from a JSON file.
#A: Opens the file and returns its contents as a Python dictionary.
def load_config(path="config.json"):
    with open(path, "r") as f:
        return json.load(f)


#M: Saves reward curves together with the seed that produced each curve.
#A: Converts values to fixed NumPy types and stores rewards and seeds in one NPZ file.
def save_training_rewards(output_dir, rewards, seeds, filename="training_rewards.npz"):
    """Save per-seed training curves to an .npz archive.

    rewards: array-like [n_seeds, n_iterations].
    seeds:   sequence of length n_seeds; seeds[i] is the unique id for rewards[i],
             so individual seed performance can be recovered downstream.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    seeds = np.asarray(seeds, dtype=np.int64)
    assert rewards.shape[0] == seeds.shape[0], "one seed id required per reward curve"

    path = os.path.join(output_dir, filename)
    np.savez(path, rewards=rewards, seeds=seeds)
    return path


#M: Column order for the per-iteration diagnostics log.
#A: Fixed so the CSV header is stable and downstream readers can rely on it.
METRIC_FIELDS = (
    "iteration",
    "train_reward",
    "best_reward",
    "kl",            # true KL(pi_old || pi_new) over the batch states, after the step
    "grad_norm",     # ||g|| before NPG preconditioning
    "nat_grad_norm", # ||F^-1 g|| after preconditioning (equals grad_norm for GPOMDP)
    "entropy",       # mean policy entropy, in nats
    "mean_std",      # mean action std; shrinking std drives Fisher curvature up
    "return_mean",   # mean discounted return over valid steps (pre-centering)
    "return_std",    # std of those returns; the divisor when normalize_returns is on
    "episode_len",   # mean steps per episode
    "rollout_time",
    "update_time",
    "total_time",
)


class _Tee:
    """Duplicate writes to several streams (used to mirror stdout into a log file)."""

    #M: Remembers every stream that should receive the output.
    #A: Stores the streams so each write can be forwarded to all of them.
    def __init__(self, *streams):
        self._streams = streams

    #M: Sends text to every stream.
    #A: Flushes immediately so a killed run still has a complete log on disk.
    def write(self, data):
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    #M: Flushes every stream.
    #A: Called by print() and at interpreter shutdown.
    def flush(self):
        for stream in self._streams:
            stream.flush()


#M: Redirects everything printed inside the block into a log file.
#A: Swaps sys.stdout/sys.stderr for a Tee, and always restores them on exit so a
#   failure in one seed cannot leave later seeds writing to a closed file.
#   tee=True also mirrors to the original console (sequential runs); tee=False writes
#   only to the file, so parallel seed workers don't interleave on the shared console.
@contextlib.contextmanager
def _tee_stdout(path, tee=True):
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


#M: Measures the total size of the current gradient across all parameters.
#A: Concatenates every .grad into one vector so ||g|| is comparable across runs.
def _grad_norm(policy):
    grads = [
        p.grad.reshape(-1) for p in policy.parameters() if p.grad is not None
    ]
    if not grads:
        return float("nan")
    return float(torch.cat(grads).norm())


#M: Copies the current action distribution so it survives the parameter update.
#A: Rebuilds the distribution from detached parameters, because a live distribution
#   holds references to the policy tensors and would change under optimizer.step().
def _frozen_distribution(policy, flat_states):
    with torch.no_grad():
        dist = policy.distribution(flat_states)
        if hasattr(dist, "scale"):
            # Gaussian: mean and std.
            return type(dist)(dist.loc.clone(), dist.scale.clone())
        if hasattr(dist, "logits"):
            # Categorical: class logits.
            return type(dist)(logits=dist.logits.clone())
    return None


#M: Measures the policy's action distribution over a batch of states.
#A: Averages entropy and std so exploration collapse is visible in the log.
def _policy_stats(policy, flat_states):
    with torch.no_grad():
        dist = policy.distribution(flat_states)
        entropy = dist.entropy()
        if entropy.dim() > 1:
            entropy = entropy.sum(-1)
        mean_entropy = float(entropy.mean())
        scale = getattr(dist, "scale", None)
        mean_std = float(scale.mean()) if scale is not None else float("nan")
    return mean_entropy, mean_std


#M: Measures how far the update moved the policy, in distribution space.
#A: Compares the pre- and post-update distributions at the same states, which is
#   the scale-free measure of NPG step size (the quadratic estimate is unreliable).
def _measure_kl(policy, flat_states, old_dist):
    if old_dist is None:
        return float("nan")
    with torch.no_grad():
        new_dist = policy.distribution(flat_states)
        kl = kl_divergence(old_dist, new_dist)
        if kl.dim() > 1:
            kl = kl.sum(-1)
        return float(kl.mean())


#M: Appends one row of diagnostics to the seed's metrics CSV.
#A: Opens in append mode per row so a killed run keeps everything written so far --
#   the reward .npz is only written after the full loop completes.
def _append_metrics_row(path, row):
    write_header = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


#M: Chooses the device used for policy and gradient calculations.
#A: Auto mode tries CUDA, then Apple MPS, and finally CPU.
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


#M: Prepares a function that can create one training environment.
#A: AsyncVectorEnv needs a factory so every worker can create its own environment.
def make_env(env_id: str, seed: int | None, horizon: int | None = None):
    # AsyncVectorEnv requires a factory (thunk) rather than an env instance
    # so each worker can construct its own isolated copy.
    #M: Creates, limits, and seeds one environment inside a worker.
    #A: Builds the environment only when AsyncVectorEnv starts that worker.
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


#M: Trains one policy using one random seed.
#A: Repeatedly collects trajectories, computes the loss, updates the policy,
#   records rewards, and saves the requested outputs.
def run_single_training(cfg: dict):
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

            #M: Periodically snapshots the policy so training progress can be inspected later.
            #A: Saves a numbered checkpoint every `checkpoint_interval` completed iterations.
            if save_snapshots and (iteration + 1) % checkpoint_interval == 0:
                snapshot_path = os.path.join(snapshot_dir, f"snapshot_iter_{iteration + 1:06d}.pt")
                torch.save(policy.state_dict(), snapshot_path)

            #M: Periodically saves the reward curve collected so far.
            #A: Overwrites the same file the final save uses, so a run that is killed
            #   mid-seed still leaves a loadable (truncated) curve instead of nothing.
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


#M: Records a video using the strongest seed from a multi-seed run.
#A: Chooses the best final-window reward, loads that seed's checkpoint, and replays it.
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


#M: Trains one seed and returns its reward curve.
#A: Builds the per-seed config and subdir, mirrors that seed's output to its own
#   train.log, and delegates to single-seed training. tee=True also echoes to the
#   console (sequential runs); parallel workers pass tee=False so the shared console
#   is not interleaved. Shared by the sequential and parallel seed paths so both
#   produce byte-identical per-seed layouts.
def _train_one_seed(cfg: dict, seed: int, output_dir: str, tee: bool = True):
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


#M: Process-pool entry point that trains one seed in a worker process.
#A: Returns (seed, rewards, error) so one seed's failure is reported to the parent
#   instead of tearing down the whole pool. Output goes to the seed's log file only.
def _seed_worker(packed):
    cfg, seed, output_dir = packed
    try:
        rewards = _train_one_seed(cfg, seed, output_dir, tee=False)
        return seed, rewards, None
    except BaseException:  # report any failure back to the parent, don't crash the pool
        return seed, None, traceback.format_exc()


#M: Decides how many seeds to train concurrently.
#A: 'auto' (or None) targets ~2x the core count in total env workers -- env workers are
#   IPC-bound, not CPU-bound, so mild oversubscription is fine -- bounded by the seed
#   count. An explicit integer is clamped to [1, n_seeds]. 1 (the default) means the
#   original sequential path runs untouched.
def _resolve_seed_workers(seed_workers, n_seeds: int, n_envs: int) -> int:
    if seed_workers in (None, "auto"):
        cores = os.cpu_count() or 1
        target = max(1, (2 * cores) // max(1, n_envs))
        return max(1, min(n_seeds, target))
    n = int(seed_workers)
    return min(max(n, 1), n_seeds)


#M: Trains every seed one after another (the original, crash-resilient path).
#A: After each seed, writes that seed's own file and rewrites the combined matrix, so
#   an interrupted run still leaves usable outputs for the seeds that did finish.
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


#M: Trains up to n_workers seeds concurrently in separate processes.
#A: Uses a 'spawn' pool so each worker is a fresh interpreter -- it re-imports vpg
#   (thread pinning applies) before torch loads, avoiding fork-after-torch issues with
#   the nested AsyncVectorEnv workers. Per-seed files are written as each seed finishes;
#   results are reassembled into the original seed order (failed seeds dropped). The
#   combined matrix is written once by the parent to avoid concurrent writers racing.
def _run_seeds_parallel(cfg: dict, seeds, output_dir: str, n_workers: int):
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


#M: Repeats training for several seeds and keeps every reward curve.
#A: Runs seeds sequentially (seed_workers <= 1) or concurrently, then saves the
#   combined matrix and optionally records the best seed. Concurrency does not change
#   any seed's result -- each seed is independent and self-seeded.
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


#M: Starts single-seed or multi-seed training from a saved configuration.
#A: Reads run_mode from the JSON config and calls the matching training function.
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
