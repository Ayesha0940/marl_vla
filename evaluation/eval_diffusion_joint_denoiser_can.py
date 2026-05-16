#!/usr/bin/env python3
"""
Evaluate Can diffusion policy with joint denoiser assistance.

Tests baseline diffusion policy vs. diffusion policy + joint denoiser,
using trained joint models from diffusion_models/ablation_can/.

Usage:
    python evaluation/eval_diffusion_joint_denoiser_can.py \
        --diffusion_checkpoint checkpoints/can_diffusion_policy/best_model.pt \
        --variants baseline lambda01 all_three \
        --anchors A0 A2 A3 A7 \
        --alpha_s 0.0 0.02 0.05 \
        --alpha_a 0.0 0.1 0.2 \
        --n_rollouts 25 \
        --output_csv results/can/diffusion_joint_denoiser/quick_eval.csv
"""

import argparse
import csv
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
from diffusion.can_policy import DEFAULT_OBS_KEYS, load_can_checkpoint, sample_action_sequence
from diffusion.can_policy_unet import load_unet_checkpoint, sample_action_sequence_x0


def _load_diffusion_checkpoint(checkpoint_path: str, device):
    """Auto-detect MLP vs UNet and return (model, ckpt_dict, alphas, alphas_bar, sample_fn)."""
    import torch
    probe = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if probe.get("backbone") == "unet":
        model, ckpt, alphas, alphas_bar = load_unet_checkpoint(checkpoint_path, device)
        return model, ckpt, alphas, alphas_bar, sample_action_sequence_x0
    model, ckpt, alphas, alphas_bar = load_can_checkpoint(checkpoint_path, device)
    return model, ckpt, alphas, alphas_bar, sample_action_sequence


DEFAULT_ENV_CKPT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "bc_rnn_can",
    "bc_rnn_can",
    "20260405211805",
    "models",
    "model_epoch_600.pth",
)
DEFAULT_ABLATION_DIR = os.path.join(PROJECT_ROOT, "diffusion_models", "ablation_can")
DEFAULT_ALPHA_S = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
DEFAULT_ALPHA_A = [0.0, 0.05, 0.1, 0.2]


def _load_env(env_ckpt: str):
    import torch
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, ckpt_dict = FileUtils.policy_from_checkpoint(ckpt_path=env_ckpt, device=device, verbose=False)
    env_meta = ckpt_dict["env_metadata"]
    return EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )


def _flatten_obs(obs: dict, obs_keys) -> np.ndarray:
    return np.concatenate([np.asarray(obs[key]).reshape(-1) for key in obs_keys]).astype(np.float32)


def _prepare_history(obs_vec: np.ndarray, obs_horizon: int, history: Optional[deque]) -> deque:
    if history is None:
        history = deque([obs_vec.copy() for _ in range(obs_horizon)], maxlen=obs_horizon)
    else:
        history.append(obs_vec.copy())
    return history


def _load_joint_model(ckpt_path: str, device):
    import torch
    from diffusion.joint_unet import JointUNet1D
    from diffusion.anchors import build_anchor
    from diffusion.model import make_beta_schedule

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dim = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    horizon = ckpt["horizon"]
    T = ckpt["diffusion_steps"]
    anchor_id = ckpt["anchor_id"]

    model = JointUNet1D(
        state_dim=state_dim,
        action_dim=action_dim,
        anchor_dim=ckpt.get("anchor_dim", 128),
        time_emb_dim=ckpt.get("time_emb_dim", 128),
        channel_sizes=ckpt.get("channel_sizes", (64, 128, 256)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    anchor = build_anchor(
        anchor_id,
        object_dim=ckpt.get("object_dim", 14),
        proprio_dim=ckpt.get("proprio_dim", 14),
        gripper_k=ckpt.get("gripper_k", 5),
    )
    anchor.load_state_dict(ckpt["anchor_state_dict"])
    anchor.eval().to(device)

    _, alphas, alphas_bar = make_beta_schedule(T)
    alphas = alphas.to(device)
    alphas_bar = alphas_bar.to(device)

    norm = {
        "state_mean": torch.from_numpy(np.asarray(ckpt["state_mean"], dtype=np.float32)).to(device),
        "state_std": torch.from_numpy(np.asarray(ckpt["state_std"], dtype=np.float32)).to(device),
        "action_mean": torch.from_numpy(np.asarray(ckpt["action_mean"], dtype=np.float32)).to(device),
        "action_std": torch.from_numpy(np.asarray(ckpt["action_std"], dtype=np.float32)).to(device),
    }

    joint_obs_keys = list(ckpt.get("obs_keys") or DEFAULT_OBS_KEYS)

    return model, anchor, alphas, alphas_bar, norm, horizon, state_dim, action_dim, joint_obs_keys


def _find_ablation_models(ablation_dir: str, variants: List[str], anchors: List[str]) -> Dict[str, str]:
    models = {}

    if not os.path.isdir(ablation_dir):
        print(f"Warning: Ablation directory not found: {ablation_dir}")
        return models

    variant_map = {
        "baseline": "baseline",
        "lambda01": "lam01",
        "lambda025": "lam025",
        "all_three": "all_three",
        "asym_lam01": "asym_lam01",
        "asym_noise": "asym_noise",
        "no_warmstart": "no_warmstart",
    }

    for variant in variants:
        file_variant = variant_map.get(variant, variant)
        for anchor in anchors:
            anchor_num = anchor[-1].lower() if len(anchor) >= 2 else anchor.lower()
            pattern = os.path.join(ablation_dir, f"joint_{file_variant}_a{anchor_num}.pt")
            if os.path.isfile(pattern):
                key = f"{variant}_{anchor}"
                models[key] = pattern
                print(f"Found: {key} -> {os.path.basename(pattern)}")
            else:
                print(f"Warning: Model not found: {os.path.basename(pattern)}")

    return models


def _run_rollout_with_denoiser(
    diffusion_model,
    diffusion_checkpoint,
    diffusion_alphas,
    diffusion_alphas_bar,
    env,
    horizon: int,
    seed: int,
    t_start_diffusion: Optional[int],
    joint_model,
    joint_anchor,
    joint_alphas,
    joint_alphas_bar,
    joint_norm: dict,
    joint_horizon: int,
    joint_state_dim: int,
    joint_action_dim: int,
    joint_obs_keys: Optional[list] = None,
    alpha_s: float = 0.0,
    alpha_a: float = 0.0,
    exec_horizon: int = 2,
    sample_fn=None,
) -> bool:
    import torch
    from diffusion.joint_unet import joint_denoise

    if sample_fn is None:
        sample_fn = sample_action_sequence

    device = next(diffusion_model.parameters()).device
    obs_keys = list(diffusion_checkpoint.get("obs_keys") or DEFAULT_OBS_KEYS)
    state_obs_keys = joint_obs_keys if joint_obs_keys is not None else obs_keys
    obs_mean = np.asarray(diffusion_checkpoint["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(diffusion_checkpoint["obs_std"], dtype=np.float32)
    action_mean = np.asarray(diffusion_checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(diffusion_checkpoint["action_std"], dtype=np.float32)
    obs_horizon = int(diffusion_checkpoint["obs_horizon"])
    diffusion_steps = int(diffusion_checkpoint["diffusion_steps"])

    obs = env.reset()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_vec = _flatten_obs(obs, obs_keys)
    history = deque([obs_vec.copy()] * obs_horizon, maxlen=obs_horizon)

    state_buf = deque(maxlen=joint_horizon)
    action_buf = deque(maxlen=joint_horizon)
    gripper_hist_buf = deque(maxlen=5)

    step = 0
    while step < horizon:
        obs_hist = np.stack(history, axis=0)
        if alpha_s > 0.0:
            obs_hist = obs_hist.copy()
            obs_hist[-1] = obs_hist[-1] + rng.normal(0, alpha_s, size=obs_hist[-1].shape).astype(np.float32)
        noisy_obs = obs  # used for state_buf construction in inner loop

        action_chunk = sample_fn(
            model=diffusion_model,
            obs_history=obs_hist,
            obs_mean=obs_mean,
            obs_std=obs_std,
            action_mean=action_mean,
            action_std=action_std,
            alphas=diffusion_alphas,
            alphas_bar=diffusion_alphas_bar,
            diffusion_steps=diffusion_steps,
            t_start=t_start_diffusion,
            device=device,
        )

        for action in action_chunk[:exec_horizon]:
            if step >= horizon:
                break

            if alpha_a > 0.0:
                action = action + rng.normal(0, alpha_a, size=action.shape).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)

            noisy_obs_j = dict(obs)
            if alpha_s > 0.0:
                for key in obs_keys:
                    if key in noisy_obs_j:
                        noisy_obs_j[key] = (
                            np.asarray(noisy_obs_j[key], dtype=np.float32)
                            + rng.normal(0, alpha_s, size=np.asarray(noisy_obs_j[key]).shape).astype(np.float32)
                        )

            state_vec = _flatten_obs(noisy_obs_j, state_obs_keys)
            state_buf.append(state_vec)
            action_buf.append(action)
            gripper_hist_buf.append(float(action[6]))  # gripper command at index 6 for Can

            pad = joint_horizon - len(state_buf)
            states_win = np.stack([state_buf[0]] * pad + list(state_buf), axis=0)
            actions_win = np.stack([action_buf[0]] * pad + list(action_buf), axis=0)

            s_norm = (states_win - joint_norm["state_mean"].cpu().numpy()) / joint_norm["state_std"].cpu().numpy()
            a_norm = (actions_win - joint_norm["action_mean"].cpu().numpy()) / joint_norm["action_std"].cpu().numpy()

            s_t = torch.from_numpy(s_norm).float().unsqueeze(0).to(device)
            a_t = torch.from_numpy(a_norm).float().unsqueeze(0).to(device)

            _closed = float(action[6]) < 0.5
            _obj_z = float(obs["object"][2]) if "object" in obs else 0.0
            _phase = 2 if (_closed and _obj_z > 0.85) else (1 if _closed else 0)

            object_pose_t0 = np.asarray(obs["object"], dtype=np.float32).copy()
            gh_arr = np.array([0.0] * (5 - len(gripper_hist_buf)) + list(gripper_hist_buf), dtype=np.float32)
            proprio_parts = [
                np.asarray(obs.get("robot0_joint_pos", np.zeros(7)), dtype=np.float32).flatten(),
                np.asarray(obs.get("robot0_eef_pos", np.zeros(3)), dtype=np.float32).flatten(),
                np.asarray(obs.get("robot0_eef_quat", np.zeros(4)), dtype=np.float32).flatten(),
            ]
            proprio_arr = np.concatenate(proprio_parts).astype(np.float32)

            traj = {
                "object_pose_t0": torch.from_numpy(object_pose_t0).float().unsqueeze(0).to(device),
                "gripper_history": torch.from_numpy(gh_arr).float().unsqueeze(0).to(device),
                "proprio": torch.from_numpy(proprio_arr).float().unsqueeze(0).to(device),
                "phase": torch.tensor([_phase], dtype=torch.long).to(device),
            }

            with torch.no_grad():
                anchor_emb = joint_anchor.compute(traj)
                _, clean_a = joint_denoise(
                    joint_model,
                    s_t,
                    a_t,
                    anchor_emb,
                    joint_alphas,
                    joint_alphas_bar,
                    t_start=10,
                )

            clean_a_np = clean_a[0, -1].cpu().numpy()
            clean_a_np = clean_a_np * joint_norm["action_std"].cpu().numpy() + joint_norm["action_mean"].cpu().numpy()
            clean_a_np = np.clip(clean_a_np, -1.0, 1.0)

            obs, _, done, _ = env.step(clean_a_np)
            step += 1
            obs_vec = _flatten_obs(obs, obs_keys)
            history.append(obs_vec.copy())
            if env.is_success()["task"]:
                return True
            if done:
                return False

    return False


def _run_rollout_baseline(
    diffusion_model,
    diffusion_checkpoint,
    diffusion_alphas,
    diffusion_alphas_bar,
    env,
    horizon: int,
    seed: int,
    t_start_diffusion: Optional[int],
    alpha_s: float = 0.0,
    alpha_a: float = 0.0,
    exec_horizon: int = 2,
    sample_fn=None,
) -> bool:
    import torch

    if sample_fn is None:
        sample_fn = sample_action_sequence

    device = next(diffusion_model.parameters()).device
    obs_keys = list(diffusion_checkpoint.get("obs_keys") or DEFAULT_OBS_KEYS)
    obs_mean = np.asarray(diffusion_checkpoint["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(diffusion_checkpoint["obs_std"], dtype=np.float32)
    action_mean = np.asarray(diffusion_checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(diffusion_checkpoint["action_std"], dtype=np.float32)
    obs_horizon = int(diffusion_checkpoint["obs_horizon"])
    diffusion_steps = int(diffusion_checkpoint["diffusion_steps"])

    obs = env.reset()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_vec = _flatten_obs(obs, obs_keys)
    history = deque([obs_vec.copy()] * obs_horizon, maxlen=obs_horizon)

    step = 0
    while step < horizon:
        obs_hist = np.stack(history, axis=0)
        if alpha_s > 0.0:
            obs_hist = obs_hist.copy()
            obs_hist[-1] = obs_hist[-1] + rng.normal(0, alpha_s, size=obs_hist[-1].shape).astype(np.float32)

        action_chunk = sample_fn(
            model=diffusion_model,
            obs_history=obs_hist,
            obs_mean=obs_mean,
            obs_std=obs_std,
            action_mean=action_mean,
            action_std=action_std,
            alphas=diffusion_alphas,
            alphas_bar=diffusion_alphas_bar,
            diffusion_steps=diffusion_steps,
            t_start=t_start_diffusion,
            device=device,
        )

        for action in action_chunk[:exec_horizon]:
            if step >= horizon:
                break

            if alpha_a > 0.0:
                action = action + rng.normal(0, alpha_a, size=action.shape).astype(np.float32)

            obs, _, done, _ = env.step(np.clip(action, -1.0, 1.0))
            step += 1
            obs_vec = _flatten_obs(obs, obs_keys)
            history.append(obs_vec.copy())
            if env.is_success()["task"]:
                return True
            if done:
                return False

    return False


def _eval_condition(
    diffusion_model,
    diffusion_checkpoint,
    diffusion_alphas,
    diffusion_alphas_bar,
    env,
    horizon: int,
    base_seed: int,
    n_rollouts: int,
    t_start_diffusion: Optional[int],
    alpha_s: float,
    alpha_a: float,
    exec_horizon: int = 2,
    joint_model=None,
    joint_anchor=None,
    joint_alphas=None,
    joint_alphas_bar=None,
    joint_norm=None,
    joint_horizon=None,
    joint_state_dim=None,
    joint_action_dim=None,
    joint_obs_keys=None,
    sample_fn=None,
) -> float:
    successes = []
    for index in range(n_rollouts):
        if joint_model is None:
            success = _run_rollout_baseline(
                diffusion_model, diffusion_checkpoint, diffusion_alphas, diffusion_alphas_bar,
                env, horizon, base_seed + index, t_start_diffusion,
                alpha_s=alpha_s, alpha_a=alpha_a, exec_horizon=exec_horizon,
                sample_fn=sample_fn,
            )
        else:
            success = _run_rollout_with_denoiser(
                diffusion_model, diffusion_checkpoint, diffusion_alphas, diffusion_alphas_bar,
                env, horizon, base_seed + index, t_start_diffusion,
                joint_model, joint_anchor, joint_alphas, joint_alphas_bar,
                joint_norm, joint_horizon, joint_state_dim, joint_action_dim,
                joint_obs_keys=joint_obs_keys,
                alpha_s=alpha_s, alpha_a=alpha_a, exec_horizon=exec_horizon,
                sample_fn=sample_fn,
            )
        successes.append(success)

    return float(np.mean(successes))


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--diffusion_checkpoint", type=str,
                        default=os.path.join(PROJECT_ROOT, "checkpoints", "can_diffusion_policy", "best_model.pt"),
                        help="Path to Can diffusion policy checkpoint")
    parser.add_argument("--env_ckpt", type=str, default=DEFAULT_ENV_CKPT,
                        help="Path to BC RNN Can environment checkpoint")
    parser.add_argument("--ablation_dir", type=str, default=DEFAULT_ABLATION_DIR,
                        help="Path to ablation_can directory with joint denoiser checkpoints")
    parser.add_argument("--joint_ckpts", type=str, nargs="*", default=None,
                        help="Explicit joint denoiser checkpoint paths (overrides ablation discovery)")
    parser.add_argument("--variants", type=str, nargs="+",
                        default=["baseline", "lambda01", "all_three"],
                        help="Variant patterns to evaluate")
    parser.add_argument("--anchors", type=str, nargs="+",
                        default=["A0", "A2", "A3", "A7"],
                        help="Anchor IDs to test")
    parser.add_argument("--alpha_s", type=float, nargs="+", default=DEFAULT_ALPHA_S,
                        help="State noise levels")
    parser.add_argument("--alpha_a", type=float, nargs="+", default=DEFAULT_ALPHA_A,
                        help="Action noise levels")
    parser.add_argument("--exec_horizon", type=int, default=2,
                        help="Actions to execute per chunk before replanning (default 2 for Can)")
    parser.add_argument("--n_rollouts", type=int, default=25, help="Rollouts per condition")
    parser.add_argument("--horizon", type=int, default=400, help="Episode horizon")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--t_start", type=int, default=None, help="Diffusion t_start parameter")
    parser.add_argument("--output_csv", type=str, required=True, help="Output CSV path")
    return parser.parse_args()


def main():
    args = parse_args()

    diffusion_checkpoint = (
        args.diffusion_checkpoint if os.path.isabs(args.diffusion_checkpoint)
        else os.path.join(PROJECT_ROOT, args.diffusion_checkpoint)
    )
    if not os.path.isfile(diffusion_checkpoint):
        print(f"Error: Diffusion checkpoint not found: {diffusion_checkpoint}")
        return 1

    env_ckpt = (
        args.env_ckpt if os.path.isabs(args.env_ckpt)
        else os.path.join(PROJECT_ROOT, args.env_ckpt)
    )
    if not os.path.isfile(env_ckpt):
        print(f"Error: Env checkpoint not found: {env_ckpt}")
        return 1

    print("=" * 90)
    print("CAN DIFFUSION POLICY + JOINT DENOISER EVALUATION")
    print("=" * 90)
    print(f"Diffusion checkpoint: {diffusion_checkpoint}")
    print(f"Env checkpoint:       {env_ckpt}")
    print(f"Ablation dir:         {args.ablation_dir}")
    print(f"Variants:             {args.variants}")
    print(f"Anchors:              {args.anchors}")
    print(f"exec_horizon:         {args.exec_horizon}")
    print(f"State noise (alpha_s): {args.alpha_s}")
    print(f"Action noise (alpha_a): {args.alpha_a}")
    print(f"Rollouts per condition: {args.n_rollouts}")

    configure_mujoco_env(verbose=False)

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print("Loading Can diffusion policy...")
    diffusion_model, diffusion_checkpoint_dict, diffusion_alphas, diffusion_alphas_bar, sample_fn = \
        _load_diffusion_checkpoint(diffusion_checkpoint, device)
    backbone = diffusion_checkpoint_dict.get("backbone", "mlp")
    pred_type = diffusion_checkpoint_dict.get("prediction_type", "epsilon")
    print(f"  -> backbone={backbone}  prediction={pred_type}")

    print("Loading environment...")
    env = _load_env(env_ckpt)

    print("Searching for joint denoiser models...")
    if args.joint_ckpts:
        joint_models = {}
        for p in args.joint_ckpts:
            p_abs = p if os.path.isabs(p) else os.path.join(PROJECT_ROOT, p)
            if os.path.isfile(p_abs):
                key = os.path.splitext(os.path.basename(p_abs))[0]
                joint_models[key] = p_abs
                print(f"Using explicit joint ckpt: {key} -> {os.path.basename(p_abs)}")
            else:
                print(f"Warning: joint checkpoint not found: {p_abs}")
    else:
        joint_models = _find_ablation_models(args.ablation_dir, args.variants, args.anchors)

    if not joint_models:
        print(f"Warning: No joint denoiser models found — running baseline only.")

    results = {}
    col_names = ["BASELINE (diffusion only)"] + list(joint_models.keys())

    print("\n" + "=" * 90)
    print("EVALUATION")
    print("=" * 90)

    total_cells = len(args.alpha_s) * len(args.alpha_a)
    cell_idx = 0

    for alpha_s in args.alpha_s:
        for alpha_a in args.alpha_a:
            cell_idx += 1
            key = (f"{alpha_s:.3f}", f"{alpha_a:.3f}")
            results[key] = {}

            print(f"\n[Cell {cell_idx}/{total_cells}] alpha_s={alpha_s:.3f}, alpha_a={alpha_a:.3f}")

            print("  BASELINE...", end=" ", flush=True)
            sr_baseline = _eval_condition(
                diffusion_model, diffusion_checkpoint_dict,
                diffusion_alphas, diffusion_alphas_bar,
                env, args.horizon, args.seed, args.n_rollouts, args.t_start,
                alpha_s, alpha_a, exec_horizon=args.exec_horizon,
                sample_fn=sample_fn,
            )
            results[key]["BASELINE (diffusion only)"] = sr_baseline
            print(f"success_rate={sr_baseline:.4f}")

            for model_key, model_path in sorted(joint_models.items()):
                print(f"  {model_key}...", end=" ", flush=True)
                try:
                    jm, ja, ja_alphas, ja_alphas_bar, ja_norm, ja_h, ja_s_dim, ja_a_dim, ja_obs_keys = \
                        _load_joint_model(model_path, device)

                    sr = _eval_condition(
                        diffusion_model, diffusion_checkpoint_dict,
                        diffusion_alphas, diffusion_alphas_bar,
                        env, args.horizon, args.seed, args.n_rollouts, args.t_start,
                        alpha_s, alpha_a, exec_horizon=args.exec_horizon,
                        joint_model=jm, joint_anchor=ja,
                        joint_alphas=ja_alphas, joint_alphas_bar=ja_alphas_bar,
                        joint_norm=ja_norm, joint_horizon=ja_h,
                        joint_state_dim=ja_s_dim, joint_action_dim=ja_a_dim,
                        joint_obs_keys=ja_obs_keys,
                        sample_fn=sample_fn,
                    )
                    results[key][model_key] = sr
                    print(f"success_rate={sr:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    results[key][model_key] = None

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

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'alpha_s':>10}  {'alpha_a':>10}" + "".join(f"  {c:>15}" for c in col_names))
    print("─" * (24 + 17 * len(col_names)))
    for (a_s, a_a), vals in sorted(results.items(), key=lambda x: (float(x[0][0]), float(x[0][1]))):
        row_str = f"{float(a_s):>10.3f}  {float(a_a):>10.3f}"
        for col in col_names:
            v = vals.get(col)
            row_str += f"  {v:>15.4f}" if v is not None else f"  {'—':>15}"
        print(row_str)

    print(f"\nResults saved to: {args.output_csv}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
