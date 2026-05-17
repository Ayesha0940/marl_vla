#!/usr/bin/env python3
"""
Evaluate a diffusion policy checkpoint under three noise modes:
1. State (observation) noise only
2. Action noise only
3. Joint state + action noise

Results are saved as a CSV showing success rates across noise levels.

Usage:
    python evaluation/eval_noise_modes.py \
        --checkpoint /path/to/model_epoch_3500.pt \
        --alpha_s 0.0 0.1 0.2 0.3 0.4 0.5 0.6 \
        --alpha_a 0.05 0.1 0.2 \
        --n_rollouts 50 \
        --output_csv results/lift/custom_diffusion_policy/noise_modes.csv
"""

import argparse
import csv
import os
import sys
from collections import deque
from datetime import datetime
from typing import List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (PROJECT_ROOT, EVAL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.mujoco import configure_mujoco_env
from diffusion.lift_policy import DEFAULT_OBS_KEYS, load_lift_checkpoint, sample_action_sequence

# x0 sampler — needed for unet/unet_samdp/transformer checkpoints
sys.path.insert(0, os.path.join(PROJECT_ROOT, "training", "lift"))
_sample_x0 = None
for _mod in ("train_unet_diffusion_lift", "train_diffusion_lift_v2"):
    try:
        _sample_x0 = __import__(_mod, fromlist=["sample_action_sequence_x0"]).sample_action_sequence_x0
        break
    except (ImportError, AttributeError):
        continue


def _load_checkpoint_any(checkpoint_path: str, device):
    """Backbone-aware checkpoint loader (mlp / unet / unet_samdp / transformer)."""
    import torch
    backbone = torch.load(checkpoint_path, map_location="cpu", weights_only=False).get("backbone", "mlp")
    if backbone in ("unet", "unet_samdp"):
        from diffusion.lift_policy_unet import load_unet_checkpoint
        return load_unet_checkpoint(checkpoint_path, device)
    if backbone == "transformer":
        from diffusion.lift_policy_transformer import load_transformer_checkpoint
        return load_transformer_checkpoint(checkpoint_path, device)
    return load_lift_checkpoint(checkpoint_path, device)

DEFAULT_ENV_CKPT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "bc_rnn_lift",
    "bc_rnn_lift",
    "20260418190845",
    "models",
    "model_epoch_600.pth",
)

DEFAULT_ALPHA_S = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
DEFAULT_ALPHA_A = [0.0, 0.05, 0.1, 0.2]


def _load_env(env_ckpt: str):
    import torch
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=env_ckpt, device=device, verbose=False)
    env_meta = ckpt_dict["env_metadata"]
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    return env


def _flatten_obs(obs: dict, obs_keys) -> np.ndarray:
    return np.concatenate([np.asarray(obs[key]).reshape(-1) for key in obs_keys]).astype(np.float32)


def _run_rollout(
    model,
    checkpoint,
    alphas,
    alphas_bar,
    env,
    horizon: int,
    seed: int,
    t_start: Optional[int],
    alpha_s: float = 0.0,
    alpha_a: float = 0.0,
) -> bool:
    import torch

    device = next(model.parameters()).device
    obs_keys       = list(checkpoint.get("obs_keys") or DEFAULT_OBS_KEYS)
    obs_mean       = np.asarray(checkpoint["obs_mean"],    dtype=np.float32)
    obs_std        = np.asarray(checkpoint["obs_std"],     dtype=np.float32)
    action_mean    = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std     = np.asarray(checkpoint["action_std"],  dtype=np.float32)
    obs_horizon    = int(checkpoint["obs_horizon"])
    action_horizon = int(checkpoint["action_horizon"])
    diffusion_steps = int(checkpoint["diffusion_steps"])

    obs = env.reset()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize history with clean initial obs before the loop
    obs_vec = _flatten_obs(obs, obs_keys)
    history = deque([obs_vec.copy()] * obs_horizon, maxlen=obs_horizon)

    prediction_type = checkpoint.get("prediction_type", "epsilon")
    sampler = _sample_x0 if prediction_type == "x0" else sample_action_sequence
    if prediction_type == "x0" and _sample_x0 is None:
        raise RuntimeError("sample_action_sequence_x0 not found; check training/lift/ is on path")

    step = 0
    while step < horizon:
        # Build noisy obs for policy query; keep obs pointing to clean env obs
        if alpha_s > 0.0:
            noisy_obs = dict(obs)
            for key in obs_keys:
                if key in noisy_obs:
                    noisy_obs[key] = (
                        np.asarray(noisy_obs[key], dtype=np.float32)
                        + rng.normal(0, alpha_s, size=np.asarray(noisy_obs[key]).shape).astype(np.float32)
                    )
            obs_vec = _flatten_obs(noisy_obs, obs_keys)
        else:
            obs_vec = _flatten_obs(obs, obs_keys)
        history.append(obs_vec.copy())
        obs_hist = np.stack(history, axis=0)

        action_chunk = sampler(
            model=model,
            obs_history=obs_hist,
            obs_mean=obs_mean,
            obs_std=obs_std,
            action_mean=action_mean,
            action_std=action_std,
            alphas=alphas,
            alphas_bar=alphas_bar,
            diffusion_steps=diffusion_steps,
            t_start=t_start,
            device=device,
        )

        for action in action_chunk[:action_horizon]:
            if step >= horizon:
                break
            if alpha_a > 0.0:
                action = action + rng.normal(0, alpha_a, size=action.shape).astype(np.float32)
            obs, _, done, _ = env.step(np.clip(action, -1.0, 1.0))
            step += 1
            # Update history with clean obs after each step (rolling window)
            obs_vec = _flatten_obs(obs, obs_keys)
            history.append(obs_vec.copy())
            if env.is_success()["task"]:
                return True
            if done:
                return False

    return False


def _eval_noise_condition(
    model,
    checkpoint,
    alphas,
    alphas_bar,
    env,
    horizon: int,
    base_seed: int,
    n_rollouts: int,
    t_start: Optional[int],
    alpha_s: float,
    alpha_a: float,
) -> float:
    """Evaluate success rate for a specific (alpha_s, alpha_a) pair."""
    successes = []
    for index in range(n_rollouts):
        success = _run_rollout(
            model,
            checkpoint,
            alphas,
            alphas_bar,
            env,
            horizon,
            base_seed + index,
            t_start,
            alpha_s=alpha_s,
            alpha_a=alpha_a,
        )
        successes.append(success)
        print(f"    Rollout {index+1:3d}/{n_rollouts}: {'SUCCESS' if success else 'FAILURE'}", flush=True)
    return float(np.mean(successes))


def _run_evaluation(
    checkpoint_path: str,
    env_ckpt: str,
    alpha_s_list: List[float],
    alpha_a_list: List[float],
    n_rollouts: int,
    horizon: int,
    base_seed: int,
    t_start: Optional[int],
    mode: str = "all",
) -> dict:
    """
    Run evaluation under the selected noise mode(s).

    mode: "state_only" | "action_only" | "joint" | "all"

    Returns a dict containing only the keys for the modes that ran:
        "state_noise_only": {alpha_s: success_rate, ...}
        "action_noise_only": {alpha_a: success_rate, ...}
        "joint_noise":       {(alpha_s, alpha_a): success_rate, ...}
    """
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nLoading checkpoint... {checkpoint_path}", flush=True)
    model, checkpoint, alphas, alphas_bar = _load_checkpoint_any(checkpoint_path, device)
    backbone = checkpoint.get("backbone", "mlp")
    pred_type = checkpoint.get("prediction_type", "epsilon")
    print(f"  backbone={backbone}  prediction_type={pred_type}  device={device}", flush=True)

    results: dict = {}

    # Each mode gets a fresh env so all start from the same env RNG position.

    if mode in ("state_only", "all"):
        print("\n" + "=" * 80, flush=True)
        print("MODE: STATE NOISE ONLY", flush=True)
        print("=" * 80, flush=True)
        print("Loading environment...", flush=True)
        env = _load_env(env_ckpt)
        state_results: dict = {}
        for alpha_s in alpha_s_list:
            print(f"  Evaluating alpha_s={alpha_s:.3f}...", end=" ", flush=True)
            sr = _eval_noise_condition(
                model, checkpoint, alphas, alphas_bar, env, horizon, base_seed, n_rollouts, t_start,
                alpha_s=alpha_s, alpha_a=0.0,
            )
            state_results[alpha_s] = sr
            print(f"success_rate={sr:.4f}", flush=True)
        results["state_noise_only"] = state_results

    if mode in ("action_only", "all"):
        print("\n" + "=" * 80, flush=True)
        print("MODE: ACTION NOISE ONLY", flush=True)
        print("=" * 80, flush=True)
        print("Loading environment...", flush=True)
        env = _load_env(env_ckpt)
        action_results: dict = {}
        for alpha_a in alpha_a_list:
            print(f"  Evaluating alpha_a={alpha_a:.3f}...", end=" ", flush=True)
            sr = _eval_noise_condition(
                model, checkpoint, alphas, alphas_bar, env, horizon, base_seed, n_rollouts, t_start,
                alpha_s=0.0, alpha_a=alpha_a,
            )
            action_results[alpha_a] = sr
            print(f"success_rate={sr:.4f}", flush=True)
        results["action_noise_only"] = action_results

    if mode in ("joint", "all"):
        print("\n" + "=" * 80, flush=True)
        print("MODE: JOINT STATE + ACTION NOISE", flush=True)
        print("=" * 80, flush=True)
        print("Loading environment...", flush=True)
        env = _load_env(env_ckpt)
        joint_results: dict = {}
        total_cells = len(alpha_s_list) * len(alpha_a_list)
        cell_idx = 0
        for alpha_s in alpha_s_list:
            for alpha_a in alpha_a_list:
                cell_idx += 1
                print(
                    f"  [{cell_idx}/{total_cells}] alpha_s={alpha_s:.3f}, alpha_a={alpha_a:.3f}...",
                    end=" ", flush=True,
                )
                sr = _eval_noise_condition(
                    model, checkpoint, alphas, alphas_bar, env, horizon, base_seed, n_rollouts, t_start,
                    alpha_s=alpha_s, alpha_a=alpha_a,
                )
                joint_results[(alpha_s, alpha_a)] = sr
                print(f"success_rate={sr:.4f}", flush=True)
        results["joint_noise"] = joint_results

    return results


def _save_results_csv(results: dict, output_csv: str):
    """Save one CSV per mode that was actually run."""
    out_dir = os.path.dirname(output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    base_name = output_csv.replace(".csv", "")
    saved: List[str] = []

    if "state_noise_only" in results:
        path = f"{base_name}_state_only.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["alpha_s", "success_rate"])
            for alpha_s in sorted(results["state_noise_only"].keys()):
                writer.writerow([f"{alpha_s:.3f}", f"{results['state_noise_only'][alpha_s]:.4f}"])
        saved.append(path)

    if "action_noise_only" in results:
        path = f"{base_name}_action_only.csv"
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["alpha_a", "success_rate"])
            for alpha_a in sorted(results["action_noise_only"].keys()):
                writer.writerow([f"{alpha_a:.3f}", f"{results['action_noise_only'][alpha_a]:.4f}"])
        saved.append(path)

    if "joint_noise" in results:
        path = f"{base_name}_joint.csv"
        alpha_s_vals = sorted(set(k[0] for k in results["joint_noise"].keys()))
        alpha_a_vals = sorted(set(k[1] for k in results["joint_noise"].keys()))
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["alpha_s \\ alpha_a"] + [f"{a:.3f}" for a in alpha_a_vals])
            for alpha_s in alpha_s_vals:
                row = [f"{alpha_s:.3f}"]
                for alpha_a in alpha_a_vals:
                    sr = results["joint_noise"].get((alpha_s, alpha_a))
                    row.append(f"{sr:.4f}" if sr is not None else "")
                writer.writerow(row)
        saved.append(path)

    print(f"\nSaved results to:")
    for p in saved:
        print(f"  {p}")


def _print_summary(results: dict):
    """Print summary tables to console (only for modes that were run)."""
    if "state_noise_only" in results:
        print("\n" + "=" * 80)
        print("SUMMARY: STATE NOISE ONLY")
        print("=" * 80)
        print(f"{'alpha_s':>10}  {'success_rate':>15}")
        print("-" * 30)
        for alpha_s in sorted(results["state_noise_only"].keys()):
            print(f"{alpha_s:>10.3f}  {results['state_noise_only'][alpha_s]:>15.4f}")

    if "action_noise_only" in results:
        print("\n" + "=" * 80)
        print("SUMMARY: ACTION NOISE ONLY")
        print("=" * 80)
        print(f"{'alpha_a':>10}  {'success_rate':>15}")
        print("-" * 30)
        for alpha_a in sorted(results["action_noise_only"].keys()):
            print(f"{alpha_a:>10.3f}  {results['action_noise_only'][alpha_a]:>15.4f}")

    if "joint_noise" in results:
        print("\n" + "=" * 80)
        print("SUMMARY: JOINT STATE + ACTION NOISE")
        print("=" * 80)
        alpha_s_vals = sorted(set(k[0] for k in results["joint_noise"].keys()))
        alpha_a_vals = sorted(set(k[1] for k in results["joint_noise"].keys()))
        header = "alpha_s \\ alpha_a"
        print(f"{header:>17}" + "".join(f"  {a:>10.3f}" for a in alpha_a_vals))
        print("-" * (17 + 12 * len(alpha_a_vals)))
        for alpha_s in alpha_s_vals:
            row = f"{alpha_s:>17.3f}"
            for alpha_a in alpha_a_vals:
                sr = results["joint_noise"].get((alpha_s, alpha_a))
                row += f"  {sr:>10.4f}" if sr is not None else f"  {'—':>10}"
            print(row)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to diffusion policy checkpoint")
    parser.add_argument(
        "--env_ckpt",
        type=str,
        default=DEFAULT_ENV_CKPT,
        help="Path to BC RNN environment checkpoint",
    )
    parser.add_argument(
        "--alpha_s",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHA_S,
        help="State noise levels (default: {})".format(DEFAULT_ALPHA_S),
    )
    parser.add_argument(
        "--alpha_a",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHA_A,
        help="Action noise levels (default: {})".format(DEFAULT_ALPHA_A),
    )
    parser.add_argument("--n_rollouts", type=int, default=50, help="Rollouts per condition")
    parser.add_argument("--horizon", type=int, default=400, help="Episode horizon")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--t_start", type=int, default=None, help="Diffusion t_start parameter")
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Base path for output CSV files (suffix _state_only/_action_only/_joint appended per mode)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["state_only", "action_only", "joint", "all"],
        help="Which noise mode(s) to evaluate (default: all)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    checkpoint_path = (
        args.checkpoint
        if os.path.isabs(args.checkpoint)
        else os.path.join(PROJECT_ROOT, args.checkpoint)
    )
    if not os.path.isfile(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1

    env_ckpt = (
        args.env_ckpt
        if os.path.isabs(args.env_ckpt)
        else os.path.join(PROJECT_ROOT, args.env_ckpt)
    )
    if not os.path.isfile(env_ckpt):
        print(f"Error: Env checkpoint not found: {env_ckpt}")
        return 1

    print("=" * 80, flush=True)
    print("DIFFUSION POLICY NOISE MODES EVALUATION", flush=True)
    print("=" * 80, flush=True)
    print(f"Checkpoint:    {checkpoint_path}", flush=True)
    print(f"Env checkpoint: {env_ckpt}", flush=True)
    print(f"Mode:           {args.mode}", flush=True)
    print(f"State noise levels (alpha_s): {args.alpha_s}", flush=True)
    print(f"Action noise levels (alpha_a): {args.alpha_a}", flush=True)
    print(f"Rollouts per condition: {args.n_rollouts}", flush=True)
    print(f"Horizon: {args.horizon}", flush=True)
    print(f"Seed: {args.seed}", flush=True)

    configure_mujoco_env(verbose=False)

    # Run evaluation
    results = _run_evaluation(
        checkpoint_path=checkpoint_path,
        env_ckpt=env_ckpt,
        alpha_s_list=args.alpha_s,
        alpha_a_list=args.alpha_a,
        n_rollouts=args.n_rollouts,
        horizon=args.horizon,
        base_seed=args.seed,
        t_start=args.t_start,
        mode=args.mode,
    )

    # Print and save results
    _print_summary(results)
    _save_results_csv(results, args.output_csv)

    print("\n" + "=" * 80)
    print("Evaluation complete!")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
