#!/usr/bin/env python3
"""
Evaluate Square UNet diffusion policy under joint state-action noise,
with and without the joint denoiser.

Tests two conditions per (alpha_s, alpha_a) noise level:
  BASELINE  — diffusion policy only (noisy obs → policy → noisy action → env)
  +DENOISER — noisy (state, action) window denoised before execution

Usage:
    # Quick check (clean obs, one noise point)
    python evaluation/eval_diffusion_joint_denoiser_square.py \
        --diffusion_checkpoint checkpoints/square_diffusion_policy_unet/best_model.pt \
        --joint_ckpts diffusion_models/joint_a7_square.pt \
        --alpha_s 0.0 0.05 \
        --alpha_a 0.0 0.2 \
        --n_rollouts 25 \
        --output_csv results/square/diffusion_joint_denoiser/quick_eval.csv

    # Full noise grid
    python evaluation/eval_diffusion_joint_denoiser_square.py \
        --diffusion_checkpoint checkpoints/square_diffusion_policy_unet/best_model.pt \
        --joint_ckpts diffusion_models/joint_a7_square.pt \
        --n_rollouts 50 \
        --output_csv results/square/diffusion_joint_denoiser/full_eval.csv
"""

import argparse
import csv
import glob
import os
import sys
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (PROJECT_ROOT, EVAL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.mujoco import configure_mujoco_env
from diffusion.square_policy import DEFAULT_OBS_KEYS
from diffusion.square_policy_unet import load_unet_checkpoint, sample_action_sequence_x0


def _find_default_env_ckpt() -> str:
    search = os.path.join(PROJECT_ROOT, "checkpoints", "bc_rnn_square", "**", "models", "*.pth")
    matches = sorted(glob.glob(search, recursive=True))
    return matches[0] if matches else ""


DEFAULT_ENV_CKPT = _find_default_env_ckpt()
DEFAULT_DIFFUSION_CKPT = os.path.join(
    PROJECT_ROOT, "checkpoints", "square_diffusion_policy_unet", "best_model.pt"
)
DEFAULT_ALPHA_S = [0.0, 0.01, 0.02, 0.03]
DEFAULT_ALPHA_A = [0.0, 0.05, 0.1, 0.2]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_env(env_ckpt: str):
    import torch
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=env_ckpt, device=device, verbose=False)
    return EnvUtils.create_env_from_metadata(
        env_meta=ckpt_dict["env_metadata"],
        render=False, render_offscreen=False, use_image_obs=False,
    )


def _flatten_obs(obs: dict, obs_keys) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs[k]).reshape(-1) for k in obs_keys]
    ).astype(np.float32)


def _load_joint_model(ckpt_path: str, device):
    import torch
    from diffusion.joint_unet import JointUNet1D
    from diffusion.anchors import build_anchor
    from diffusion.model import make_beta_schedule

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = JointUNet1D(
        state_dim     = ckpt["state_dim"],
        action_dim    = ckpt["action_dim"],
        anchor_dim    = ckpt.get("anchor_dim", 128),
        time_emb_dim  = ckpt.get("time_emb_dim", 128),
        channel_sizes = ckpt.get("channel_sizes", (64, 128, 256)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    anchor = build_anchor(
        ckpt["anchor_id"],
        object_dim  = ckpt.get("object_dim", 14),
        proprio_dim = ckpt.get("proprio_dim", 14),
        gripper_k   = ckpt.get("gripper_k", 5),
    )
    anchor.load_state_dict(ckpt["anchor_state_dict"])
    anchor.eval().to(device)

    _, alphas, alphas_bar = make_beta_schedule(ckpt["diffusion_steps"])
    norm = {
        "state_mean":  _to_device(ckpt["state_mean"],  device),
        "state_std":   _to_device(ckpt["state_std"],   device),
        "action_mean": _to_device(ckpt["action_mean"], device),
        "action_std":  _to_device(ckpt["action_std"],  device),
    }
    joint_obs_keys = list(ckpt.get("obs_keys") or DEFAULT_OBS_KEYS)
    return (model, anchor,
            alphas.to(device), alphas_bar.to(device),
            norm, ckpt["horizon"],
            ckpt["state_dim"], ckpt["action_dim"],
            joint_obs_keys)


def _to_device(arr, device):
    import torch
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)


def _build_anchor_traj(obs, action, gripper_hist_buf, device):
    """Build the traj dict expected by anchor.compute()."""
    import torch
    object_pose_t0 = np.asarray(obs.get("object", np.zeros(14)), dtype=np.float32).flatten()
    gh_arr = np.array(
        [0.0] * (5 - len(gripper_hist_buf)) + list(gripper_hist_buf),
        dtype=np.float32,
    )
    proprio_arr = np.concatenate([
        np.asarray(obs.get("robot0_joint_pos", np.zeros(7)), dtype=np.float32).flatten(),
        np.asarray(obs.get("robot0_eef_pos",   np.zeros(3)), dtype=np.float32).flatten(),
        np.asarray(obs.get("robot0_eef_quat",  np.zeros(4)), dtype=np.float32).flatten(),
    ])
    _closed = float(action[6]) < 0.5
    _obj_z  = float(obs["object"][2]) if "object" in obs else 0.0
    _phase  = 2 if (_closed and _obj_z > 0.85) else (1 if _closed else 0)
    return {
        "object_pose_t0":  torch.from_numpy(object_pose_t0).float().unsqueeze(0).to(device),
        "gripper_history": torch.from_numpy(gh_arr).float().unsqueeze(0).to(device),
        "proprio":         torch.from_numpy(proprio_arr).float().unsqueeze(0).to(device),
        "phase":           torch.tensor([_phase], dtype=torch.long).to(device),
    }


# ---------------------------------------------------------------------------
# Rollout functions
# ---------------------------------------------------------------------------

def _run_rollout_baseline(
    diffusion_model, diffusion_ckpt,
    diffusion_alphas, diffusion_alphas_bar,
    env, horizon: int, seed: int, t_start: Optional[int],
    alpha_s: float, alpha_a: float,
) -> bool:
    import torch
    device     = next(diffusion_model.parameters()).device
    obs_keys   = list(diffusion_ckpt.get("obs_keys") or DEFAULT_OBS_KEYS)
    obs_mean   = np.asarray(diffusion_ckpt["obs_mean"],    dtype=np.float32)
    obs_std    = np.asarray(diffusion_ckpt["obs_std"],     dtype=np.float32)
    act_mean   = np.asarray(diffusion_ckpt["action_mean"], dtype=np.float32)
    act_std    = np.asarray(diffusion_ckpt["action_std"],  dtype=np.float32)
    obs_h      = int(diffusion_ckpt["obs_horizon"])
    act_h      = int(diffusion_ckpt["action_horizon"])
    diff_steps = int(diffusion_ckpt["diffusion_steps"])

    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = env.reset()

    history = deque([_flatten_obs(obs, obs_keys)] * obs_h, maxlen=obs_h)
    step = 0
    while step < horizon:
        if alpha_s > 0.0:
            noisy = dict(obs)
            for k in obs_keys:
                if k in noisy:
                    noisy[k] = np.asarray(noisy[k], dtype=np.float32) + rng.normal(0, alpha_s, np.asarray(noisy[k]).shape).astype(np.float32)
            history.append(_flatten_obs(noisy, obs_keys))
        else:
            history.append(_flatten_obs(obs, obs_keys))

        obs_hist = np.stack(history, axis=0)
        action_chunk = sample_action_sequence_x0(
            model=diffusion_model, obs_history=obs_hist,
            obs_mean=obs_mean, obs_std=obs_std,
            action_mean=act_mean, action_std=act_std,
            alphas=diffusion_alphas, alphas_bar=diffusion_alphas_bar,
            diffusion_steps=diff_steps, t_start=t_start, device=device,
        )
        for action in action_chunk[:act_h]:
            if step >= horizon:
                break
            if alpha_a > 0.0:
                action = action + rng.normal(0, alpha_a, action.shape).astype(np.float32)
            obs, _, done, _ = env.step(np.clip(action, -1.0, 1.0))
            step += 1
            if env.is_success()["task"]:
                return True
            if done:
                return False
    return False


def _run_rollout_with_denoiser(
    diffusion_model, diffusion_ckpt,
    diffusion_alphas, diffusion_alphas_bar,
    env, horizon: int, seed: int, t_start: Optional[int],
    joint_model, joint_anchor,
    joint_alphas, joint_alphas_bar,
    joint_norm: dict, joint_horizon: int,
    joint_state_dim: int, joint_action_dim: int,
    joint_obs_keys: Optional[list],
    alpha_s: float, alpha_a: float,
    joint_t_start: int = 10,
) -> bool:
    import torch
    from diffusion.joint_unet import joint_denoise

    device       = next(diffusion_model.parameters()).device
    obs_keys     = list(diffusion_ckpt.get("obs_keys") or DEFAULT_OBS_KEYS)
    state_keys   = joint_obs_keys if joint_obs_keys is not None else obs_keys
    obs_mean     = np.asarray(diffusion_ckpt["obs_mean"],    dtype=np.float32)
    obs_std      = np.asarray(diffusion_ckpt["obs_std"],     dtype=np.float32)
    act_mean     = np.asarray(diffusion_ckpt["action_mean"], dtype=np.float32)
    act_std      = np.asarray(diffusion_ckpt["action_std"],  dtype=np.float32)
    obs_h        = int(diffusion_ckpt["obs_horizon"])
    act_h        = int(diffusion_ckpt["action_horizon"])
    diff_steps   = int(diffusion_ckpt["diffusion_steps"])

    s_mean = joint_norm["state_mean"].cpu().numpy()
    s_std  = joint_norm["state_std"].cpu().numpy()
    a_mean = joint_norm["action_mean"].cpu().numpy()
    a_std  = joint_norm["action_std"].cpu().numpy()

    np.random.seed(seed); torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = env.reset()

    history         = deque([_flatten_obs(obs, obs_keys)] * obs_h, maxlen=obs_h)
    state_buf       = deque(maxlen=joint_horizon)
    action_buf      = deque(maxlen=joint_horizon)
    gripper_hist_buf = deque(maxlen=5)

    step = 0
    while step < horizon:
        # Noisy obs for diffusion policy query
        if alpha_s > 0.0:
            noisy = dict(obs)
            for k in obs_keys:
                if k in noisy:
                    noisy[k] = np.asarray(noisy[k], dtype=np.float32) + rng.normal(0, alpha_s, np.asarray(noisy[k]).shape).astype(np.float32)
            history.append(_flatten_obs(noisy, obs_keys))
        else:
            history.append(_flatten_obs(obs, obs_keys))

        obs_hist = np.stack(history, axis=0)
        action_chunk = sample_action_sequence_x0(
            model=diffusion_model, obs_history=obs_hist,
            obs_mean=obs_mean, obs_std=obs_std,
            action_mean=act_mean, action_std=act_std,
            alphas=diffusion_alphas, alphas_bar=diffusion_alphas_bar,
            diffusion_steps=diff_steps, t_start=t_start, device=device,
        )

        for action in action_chunk[:act_h]:
            if step >= horizon:
                break

            if alpha_a > 0.0:
                action = action + rng.normal(0, alpha_a, action.shape).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)

            # Corrupt obs for joint denoiser input
            noisy_j = dict(obs)
            if alpha_s > 0.0:
                for k in obs_keys:
                    if k in noisy_j:
                        noisy_j[k] = np.asarray(noisy_j[k], dtype=np.float32) + rng.normal(0, alpha_s, np.asarray(noisy_j[k]).shape).astype(np.float32)

            state_vec = _flatten_obs(noisy_j, state_keys)
            state_buf.append(state_vec)
            action_buf.append(action)
            gripper_hist_buf.append(float(action[6]))

            # Pad window to joint_horizon
            pad = joint_horizon - len(state_buf)
            s_win = np.stack([state_buf[0]] * pad + list(state_buf), axis=0)
            a_win = np.stack([action_buf[0]] * pad + list(action_buf), axis=0)

            s_t = torch.from_numpy((s_win - s_mean) / s_std).float().unsqueeze(0).to(device)
            a_t = torch.from_numpy((a_win - a_mean) / a_std).float().unsqueeze(0).to(device)

            traj = _build_anchor_traj(obs, action, gripper_hist_buf, device)

            with torch.no_grad():
                anchor_emb = joint_anchor.compute(traj)
                _, clean_a = joint_denoise(
                    joint_model, s_t, a_t, anchor_emb,
                    joint_alphas, joint_alphas_bar, t_start=joint_t_start,
                )

            clean_a_np = clean_a[0, -1].cpu().numpy() * a_std + a_mean
            clean_a_np = np.clip(clean_a_np, -1.0, 1.0)

            obs, _, done, _ = env.step(clean_a_np)
            step += 1
            if env.is_success()["task"]:
                return True
            if done:
                return False

    return False


# ---------------------------------------------------------------------------
# Eval loop
# ---------------------------------------------------------------------------

def _eval_condition(
    diffusion_model, diffusion_ckpt, diffusion_alphas, diffusion_alphas_bar,
    env, horizon, seed, n_rollouts, t_start,
    alpha_s, alpha_a,
    joint_model=None, joint_anchor=None,
    joint_alphas=None, joint_alphas_bar=None,
    joint_norm=None, joint_horizon=None,
    joint_state_dim=None, joint_action_dim=None,
    joint_obs_keys=None,
    joint_t_start: int = 10,
) -> float:
    successes = []
    for i in range(n_rollouts):
        if joint_model is None:
            ok = _run_rollout_baseline(
                diffusion_model, diffusion_ckpt, diffusion_alphas, diffusion_alphas_bar,
                env, horizon, seed + i, t_start, alpha_s, alpha_a,
            )
        else:
            ok = _run_rollout_with_denoiser(
                diffusion_model, diffusion_ckpt, diffusion_alphas, diffusion_alphas_bar,
                env, horizon, seed + i, t_start,
                joint_model, joint_anchor, joint_alphas, joint_alphas_bar,
                joint_norm, joint_horizon, joint_state_dim, joint_action_dim,
                joint_obs_keys, alpha_s, alpha_a, joint_t_start,
            )
        successes.append(ok)
    return float(np.mean(successes))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--diffusion_checkpoint", type=str, default=DEFAULT_DIFFUSION_CKPT,
                   help="Square UNet diffusion policy checkpoint")
    p.add_argument("--env_ckpt", type=str, default=DEFAULT_ENV_CKPT,
                   help="BC-RNN checkpoint (used only for env metadata)")
    p.add_argument("--joint_ckpts", type=str, nargs="*", default=None,
                   help="Joint denoiser checkpoint(s) to compare against baseline")
    p.add_argument("--alpha_s", type=float, nargs="+", default=DEFAULT_ALPHA_S,
                   help="State noise std levels (default: 0 0.01 0.02 0.03 0.04 0.05)")
    p.add_argument("--alpha_a", type=float, nargs="+", default=DEFAULT_ALPHA_A,
                   help="Action noise std levels (default: 0 0.05 0.1 0.2)")
    p.add_argument("--n_rollouts",   type=int,   default=25)
    p.add_argument("--horizon",      type=int,   default=500,
                   help="Max steps per episode (square default 500)")
    p.add_argument("--seed",         type=int,   default=0)
    p.add_argument("--t_start",      type=int,   default=None,
                   help="Diffusion denoising start step (None = full 100 steps)")
    p.add_argument("--joint_t_start", type=int,  default=10,
                   help="Joint denoiser reverse-diffusion start step (default 10)")
    p.add_argument("--output_csv",   type=str,   required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    configure_mujoco_env(verbose=False)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Validate paths
    for label, path in [("diffusion_checkpoint", args.diffusion_checkpoint),
                         ("env_ckpt", args.env_ckpt)]:
        if not os.path.isfile(path):
            print(f"ERROR: {label} not found: {path}")
            return 1

    print("=" * 90)
    print("SQUARE DIFFUSION POLICY + JOINT DENOISER EVALUATION")
    print("=" * 90)
    print(f"Diffusion checkpoint:  {args.diffusion_checkpoint}")
    print(f"Env checkpoint:        {args.env_ckpt}")
    print(f"Joint denoiser ckpts:  {args.joint_ckpts}")
    print(f"State noise (alpha_s): {args.alpha_s}")
    print(f"Action noise (alpha_a):{args.alpha_a}")
    print(f"Rollouts per cell:     {args.n_rollouts}")
    print(f"Episode horizon:       {args.horizon}")
    print(f"Device:                {device}")

    print("\nLoading Square UNet diffusion policy...")
    diffusion_model, diffusion_ckpt, diffusion_alphas, diffusion_alphas_bar = \
        load_unet_checkpoint(args.diffusion_checkpoint, device)
    print(f"  backbone={diffusion_ckpt.get('backbone')}  "
          f"prediction={diffusion_ckpt.get('prediction_type', 'x0')}")

    print("Loading environment...")
    env = _load_env(args.env_ckpt)

    # Collect joint denoiser models
    joint_models: Dict[str, str] = {}
    if args.joint_ckpts:
        for p in args.joint_ckpts:
            p_abs = p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)
            if os.path.isfile(p_abs):
                key = os.path.splitext(os.path.basename(p_abs))[0]
                joint_models[key] = p_abs
                print(f"Joint ckpt: {key} -> {os.path.basename(p_abs)}")
            else:
                print(f"WARNING: joint checkpoint not found: {p_abs}")

    col_names = ["BASELINE (diffusion only)"] + list(joint_models.keys())
    results: dict = {}

    print("\n" + "=" * 90)
    print("EVALUATION")
    print("=" * 90)

    total = len(args.alpha_s) * len(args.alpha_a)
    cell = 0
    for alpha_s in args.alpha_s:
        for alpha_a in args.alpha_a:
            cell += 1
            key = (f"{alpha_s:.3f}", f"{alpha_a:.3f}")
            results[key] = {}
            print(f"\n[{cell}/{total}] alpha_s={alpha_s:.3f}  alpha_a={alpha_a:.3f}")

            print("  BASELINE...", end=" ", flush=True)
            sr = _eval_condition(
                diffusion_model, diffusion_ckpt, diffusion_alphas, diffusion_alphas_bar,
                env, args.horizon, args.seed, args.n_rollouts, args.t_start,
                alpha_s, alpha_a,
                joint_t_start=args.joint_t_start,
            )
            results[key]["BASELINE (diffusion only)"] = sr
            print(f"success_rate={sr:.4f}")

            for model_key, model_path in sorted(joint_models.items()):
                print(f"  {model_key}...", end=" ", flush=True)
                try:
                    jm, ja, ja_al, ja_alb, ja_norm, ja_h, ja_sd, ja_ad, ja_ok = \
                        _load_joint_model(model_path, device)
                    sr = _eval_condition(
                        diffusion_model, diffusion_ckpt, diffusion_alphas, diffusion_alphas_bar,
                        env, args.horizon, args.seed, args.n_rollouts, args.t_start,
                        alpha_s, alpha_a,
                        joint_t_start=args.joint_t_start,
                        joint_model=jm, joint_anchor=ja,
                        joint_alphas=ja_al, joint_alphas_bar=ja_alb,
                        joint_norm=ja_norm, joint_horizon=ja_h,
                        joint_state_dim=ja_sd, joint_action_dim=ja_ad,
                        joint_obs_keys=ja_ok,
                    )
                    results[key][model_key] = sr
                    print(f"success_rate={sr:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    results[key][model_key] = None

    # Save CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha_s", "alpha_a"] + col_names)
        writer.writeheader()
        for (a_s, a_a), vals in sorted(results.items(), key=lambda x: (float(x[0][0]), float(x[0][1]))):
            row = {"alpha_s": a_s, "alpha_a": a_a}
            for col in col_names:
                v = vals.get(col)
                row[col] = f"{v:.4f}" if v is not None else "—"
            writer.writerow(row)

    # Print summary table
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'alpha_s':>10}  {'alpha_a':>10}" + "".join(f"  {c:>30}" for c in col_names))
    print("─" * (24 + 32 * len(col_names)))
    for (a_s, a_a), vals in sorted(results.items(), key=lambda x: (float(x[0][0]), float(x[0][1]))):
        row = f"{float(a_s):>10.3f}  {float(a_a):>10.3f}"
        for col in col_names:
            v = vals.get(col)
            row += f"  {v:>30.4f}" if v is not None else f"  {'—':>30}"
        print(row)

    print(f"\nResults saved to: {args.output_csv}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
