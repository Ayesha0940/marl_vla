#!/usr/bin/env python3
"""
Evaluate lift diffusion policy with joint denoiser assistance.

Tests baseline diffusion policy vs. diffusion policy + joint denoiser,
using existing trained joint models from diffusion_models/ablation/.

The joint denoiser takes noisy state-action pairs and denoises them,
improving robustness under sensor/actuation noise.

Usage:
    python evaluation/eval_diffusion_joint_denoiser.py \
        --diffusion_checkpoint checkpoints/lift_diffusion_policy/model_epoch_3500.pt \
        --ablation_dir diffusion_models/ablation \
        --variants baseline lambda01 \
        --anchors A0 A7 \
        --alpha_s 0.01 0.02 0.05 \
        --alpha_a 0.05 0.1 0.2 \
        --n_rollouts 50 \
        --output_csv results/lift/diffusion_joint_denoiser/ablation_results.csv
"""

import argparse
import csv
import glob
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (PROJECT_ROOT, EVAL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.mujoco import configure_mujoco_env
from diffusion.lift_policy import DEFAULT_OBS_KEYS, load_lift_checkpoint, sample_action_sequence

try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "training", "lift"))
    from train_unet_diffusion_lift import sample_action_sequence_x0
except ImportError:
    sample_action_sequence_x0 = None

DEFAULT_ENV_CKPT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "bc_rnn_lift",
    "bc_rnn_lift",
    "20260418190845",
    "models",
    "model_epoch_600.pth",
)

DEFAULT_ALPHA_S = [0.0]
DEFAULT_ALPHA_A = [0.0]


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


def _prepare_history(obs_vec: np.ndarray, obs_horizon: int, history: Optional[deque]) -> deque:
    if history is None:
        history = deque([obs_vec.copy() for _ in range(obs_horizon)], maxlen=obs_horizon)
    else:
        history.append(obs_vec.copy())
    return history


def _load_joint_model(ckpt_path: str, device):
    """Load joint denoiser model from checkpoint."""
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
    """
    Find joint denoiser models in ablation directory.
    
    Args:
        ablation_dir: Path to diffusion_models/ablation/
        variants: List of variant patterns (e.g., ['baseline', 'lambda01', 'all_three'])
        anchors: List of anchor IDs (e.g., ['A0', 'A7'])
    
    Returns:
        Dict mapping (variant, anchor) -> checkpoint_path
    """
    models = {}
    
    if not os.path.isdir(ablation_dir):
        print(f"Warning: Ablation directory not found: {ablation_dir}")
        return models
    
    # Map user-friendly variant names to actual file patterns
    variant_map = {
        'baseline': 'baseline',
        'lambda01': 'lam01',
        'lambda025': 'lam025',
        'all_three': 'all_three',
        'asym_lam01': 'asym_lam01',
        'asym_noise': 'asym_noise',
        'no_warmstart': 'no_warmstart',
    }
    
    # Pattern: joint_<variant>_a<N>.pt (lowercase a, no _lift suffix)
    for variant in variants:
        file_variant = variant_map.get(variant, variant)
        for anchor in anchors:
            # Extract anchor number: "A0" -> "0", "a0" -> "0"
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
    prediction_type: str = "epsilon",
) -> bool:
    """
    Run rollout with diffusion policy + joint denoiser.

    Workflow:
    1. Get action from diffusion policy
    2. Add action noise
    3. Get observation with state noise
    4. Apply joint denoiser to denoise state-action pair
    5. Use denoised action
    """
    import torch
    from diffusion.joint_unet import joint_denoise

    device = next(diffusion_model.parameters()).device
    obs_keys = list(diffusion_checkpoint.get("obs_keys") or DEFAULT_OBS_KEYS)
    state_obs_keys = joint_obs_keys if joint_obs_keys is not None else obs_keys
    obs_mean = np.asarray(diffusion_checkpoint["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(diffusion_checkpoint["obs_std"], dtype=np.float32)
    action_mean = np.asarray(diffusion_checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(diffusion_checkpoint["action_std"], dtype=np.float32)
    obs_horizon = int(diffusion_checkpoint["obs_horizon"])
    action_horizon = int(diffusion_checkpoint["action_horizon"])
    diffusion_steps = int(diffusion_checkpoint["diffusion_steps"])

    obs = env.reset()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_vec = _flatten_obs(obs, obs_keys)
    history = deque([obs_vec.copy()] * obs_horizon, maxlen=obs_horizon)

    # Joint denoiser state/action history buffers
    state_buf = deque(maxlen=joint_horizon)
    action_buf = deque(maxlen=joint_horizon)
    gripper_hist_buf = deque(maxlen=5)

    step = 0
    while step < horizon:
        # Apply state noise before diffusion policy (clean obs kept for env stepping)
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
            noisy_obs = obs
            obs_vec = _flatten_obs(obs, obs_keys)
        history.append(obs_vec.copy())
        obs_hist = np.stack(history, axis=0)

        # Get action from diffusion policy
        _sampler_kwargs = dict(
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
        if prediction_type == "x0":
            if sample_action_sequence_x0 is None:
                raise RuntimeError("sample_action_sequence_x0 not available; check training/lift/train_unet_diffusion_lift.py")
            action_chunk = sample_action_sequence_x0(**_sampler_kwargs)
        else:
            action_chunk = sample_action_sequence(**_sampler_kwargs)

        for action in action_chunk[:action_horizon]:
            if step >= horizon:
                break

            # Add action noise
            if alpha_a > 0.0:
                action = action + rng.normal(0, alpha_a, size=action.shape).astype(np.float32)
            action = np.clip(action, -1.0, 1.0)

            # Corrupt observation for joint denoiser input
            noisy_obs_j = dict(obs)
            if alpha_s > 0.0:
                for key in obs_keys:
                    if key in noisy_obs_j:
                        noisy_obs_j[key] = (
                            np.asarray(noisy_obs_j[key], dtype=np.float32)
                            + rng.normal(0, alpha_s, size=np.asarray(noisy_obs_j[key]).shape).astype(np.float32)
                        )

            # Build joint denoiser input using the joint denoiser's own obs_keys ordering
            state_vec = _flatten_obs(noisy_obs_j, state_obs_keys)
            state_buf.append(state_vec)
            action_buf.append(action)
            gripper_hist_buf.append(float(action[6]))  # gripper command at index 6 for Lift

            # Pad window
            pad = joint_horizon - len(state_buf)
            states_win = np.stack([state_buf[0]] * pad + list(state_buf), axis=0)
            actions_win = np.stack([action_buf[0]] * pad + list(action_buf), axis=0)

            # Normalize
            s_norm = (states_win - joint_norm["state_mean"].cpu().numpy()) / joint_norm["state_std"].cpu().numpy()
            a_norm = (actions_win - joint_norm["action_mean"].cpu().numpy()) / joint_norm["action_std"].cpu().numpy()

            s_t = torch.from_numpy(s_norm).float().unsqueeze(0).to(device)
            a_t = torch.from_numpy(a_norm).float().unsqueeze(0).to(device)

            # Build anchor dict for A0 or A7
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

                # Apply joint denoiser
                _, clean_a = joint_denoise(
                    joint_model,
                    s_t,
                    a_t,
                    anchor_emb,
                    joint_alphas,
                    joint_alphas_bar,
                    t_start=10,
                )

            # Denormalize and use denoised action
            clean_a_np = clean_a[0, -1].cpu().numpy()
            clean_a_np = clean_a_np * joint_norm["action_std"].cpu().numpy() + joint_norm["action_mean"].cpu().numpy()
            clean_a_np = np.clip(clean_a_np, -1.0, 1.0)

            obs, _, done, _ = env.step(clean_a_np)
            step += 1
            # Update rolling history after every step
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
    prediction_type: str = "epsilon",
) -> bool:
    """Run baseline: diffusion policy only (no joint denoiser)."""
    import torch

    device = next(diffusion_model.parameters()).device
    obs_keys = list(diffusion_checkpoint.get("obs_keys") or DEFAULT_OBS_KEYS)
    obs_mean = np.asarray(diffusion_checkpoint["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(diffusion_checkpoint["obs_std"], dtype=np.float32)
    action_mean = np.asarray(diffusion_checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(diffusion_checkpoint["action_std"], dtype=np.float32)
    obs_horizon = int(diffusion_checkpoint["obs_horizon"])
    action_horizon = int(diffusion_checkpoint["action_horizon"])
    diffusion_steps = int(diffusion_checkpoint["diffusion_steps"])

    obs = env.reset()
    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_vec = _flatten_obs(obs, obs_keys)
    history = deque([obs_vec.copy()] * obs_horizon, maxlen=obs_horizon)

    step = 0
    while step < horizon:
        # Build noisy obs for the policy query (keep clean obs for env stepping)
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

        _sampler_kwargs = dict(
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
        if prediction_type == "x0":
            if sample_action_sequence_x0 is None:
                raise RuntimeError("sample_action_sequence_x0 not available; check training/lift/train_unet_diffusion_lift.py")
            action_chunk = sample_action_sequence_x0(**_sampler_kwargs)
        else:
            action_chunk = sample_action_sequence(**_sampler_kwargs)

        for action in action_chunk[:action_horizon]:
            if step >= horizon:
                break

            # Add action noise
            if alpha_a > 0.0:
                action = action + rng.normal(0, alpha_a, size=action.shape).astype(np.float32)

            obs, _, done, _ = env.step(np.clip(action, -1.0, 1.0))
            step += 1
            # Update rolling history after every step
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
    prediction_type: str = "epsilon",
    joint_model=None,
    joint_anchor=None,
    joint_alphas=None,
    joint_alphas_bar=None,
    joint_norm=None,
    joint_horizon=None,
    joint_state_dim=None,
    joint_action_dim=None,
    joint_obs_keys=None,
) -> float:
    """Evaluate success rate for a condition."""
    successes = []
    for index in range(n_rollouts):
        if joint_model is None:
            # Baseline: diffusion only
            success = _run_rollout_baseline(
                diffusion_model,
                diffusion_checkpoint,
                diffusion_alphas,
                diffusion_alphas_bar,
                env,
                horizon,
                base_seed + index,
                t_start_diffusion,
                alpha_s=alpha_s,
                alpha_a=alpha_a,
                prediction_type=prediction_type,
            )
        else:
            # With joint denoiser
            success = _run_rollout_with_denoiser(
                diffusion_model,
                diffusion_checkpoint,
                diffusion_alphas,
                diffusion_alphas_bar,
                env,
                horizon,
                base_seed + index,
                t_start_diffusion,
                joint_model,
                joint_anchor,
                joint_alphas,
                joint_alphas_bar,
                joint_norm,
                joint_horizon,
                joint_state_dim,
                joint_action_dim,
                joint_obs_keys=joint_obs_keys,
                alpha_s=alpha_s,
                alpha_a=alpha_a,
                prediction_type=prediction_type,
            )
        successes.append(success)

    return float(np.mean(successes))


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--diffusion_checkpoint", type=str, required=True, help="Path to lift diffusion policy checkpoint")
    parser.add_argument("--env_ckpt", type=str, default=DEFAULT_ENV_CKPT, help="Path to BC RNN environment checkpoint")
    parser.add_argument(
        "--ablation_dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "diffusion_models", "ablation"),
        help="Path to ablation directory with joint denoiser checkpoints",
    )
    parser.add_argument(
        "--joint_ckpts",
        type=str,
        nargs="*",
        default=None,
        help="Explicit joint denoiser checkpoint paths to use instead of ablation discovery",
    )
    parser.add_argument(
        "--baseline_only",
        action="store_true",
        help="Skip denoiser evaluation; run baseline diffusion policy only",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=["baseline", "lambda01", "all_three"],
        help="Variant patterns (e.g., baseline lambda01 all_three)",
    )
    parser.add_argument(
        "--anchors",
        type=str,
        nargs="+",
        default=["A0", "A7"],
        help="Anchor IDs to test (e.g., A0 A7)",
    )
    parser.add_argument(
        "--alpha_s",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHA_S,
        help="State noise levels",
    )
    parser.add_argument(
        "--alpha_a",
        type=float,
        nargs="+",
        default=DEFAULT_ALPHA_A,
        help="Action noise levels",
    )
    parser.add_argument("--n_rollouts", type=int, default=25, help="Rollouts per condition")
    parser.add_argument("--horizon", type=int, default=400, help="Episode horizon")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--t_start", type=int, default=None, help="Diffusion t_start parameter")
    parser.add_argument(
        "--output_csv",
        type=str,
        required=True,
        help="Output CSV path",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve paths
    diffusion_checkpoint = (
        args.diffusion_checkpoint
        if os.path.isabs(args.diffusion_checkpoint)
        else os.path.join(PROJECT_ROOT, args.diffusion_checkpoint)
    )
    if not os.path.isfile(diffusion_checkpoint):
        print(f"Error: Diffusion checkpoint not found: {diffusion_checkpoint}")
        return 1

    env_ckpt = (
        args.env_ckpt
        if os.path.isabs(args.env_ckpt)
        else os.path.join(PROJECT_ROOT, args.env_ckpt)
    )
    if not os.path.isfile(env_ckpt):
        print(f"Error: Env checkpoint not found: {env_ckpt}")
        return 1

    print("=" * 90)
    print("DIFFUSION POLICY + JOINT DENOISER EVALUATION")
    print("=" * 90)
    print(f"Diffusion checkpoint: {diffusion_checkpoint}")
    print(f"Env checkpoint: {env_ckpt}")
    print(f"Ablation dir: {args.ablation_dir}")
    print(f"Variants: {args.variants}")
    print(f"Anchors: {args.anchors}")
    print(f"State noise levels (alpha_s): {args.alpha_s}")
    print(f"Action noise levels (alpha_a): {args.alpha_a}")
    print(f"Rollouts per condition: {args.n_rollouts}")

    configure_mujoco_env(verbose=False)

    # Load diffusion policy
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print("Loading diffusion policy...")
    _peek = torch.load(diffusion_checkpoint, map_location="cpu", weights_only=False)

    # Robust backbone detection: prefer explicit "backbone" field, otherwise
    # inspect keys inside the saved state_dict to guess UNet vs MLP.
    sd_keys = []
    if isinstance(_peek, dict):
        if "model_state_dict" in _peek and isinstance(_peek["model_state_dict"], dict):
            sd_keys = list(_peek["model_state_dict"].keys())
        else:
            sd_keys = list(_peek.keys())

    def _looks_like_unet(keys):
        for k in keys:
            lk = k.lower()
            if lk.startswith("down_blocks") or lk.startswith("up_blocks") or lk.startswith("final_conv") or "cond_encoder" in lk:
                return True
        return False

    is_unet = (_peek.get("backbone") == "unet") or _looks_like_unet(sd_keys)

    if is_unet:
        from diffusion.lift_policy_unet import load_unet_checkpoint
        diffusion_model, diffusion_checkpoint_dict, diffusion_alphas, diffusion_alphas_bar = load_unet_checkpoint(
            diffusion_checkpoint, device
        )
        print("  -> UNet backbone detected")
    else:
        diffusion_model, diffusion_checkpoint_dict, diffusion_alphas, diffusion_alphas_bar = load_lift_checkpoint(
            diffusion_checkpoint, device
        )
        print("  -> MLP backbone detected")

    prediction_type = diffusion_checkpoint_dict.get("prediction_type", "epsilon")
    print(f"  -> prediction_type: {prediction_type}")

    # Load environment
    print("Loading environment...")
    env = _load_env(env_ckpt)

    # Find and load joint denoiser models (or use explicit list if provided)
    joint_models = {}
    if args.baseline_only:
        print("Baseline only mode: skipping denoiser models")
    else:
        print("Searching for joint denoiser models...")
        if args.joint_ckpts:
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
            print(f"Warning: No joint denoiser models found in {args.ablation_dir}")
            joint_models = {}

    # Prepare results structure
    results = {}
    col_names = ["BASELINE (diffusion only)"] + list(joint_models.keys())

    # Evaluation loop
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

            # Baseline: diffusion only
            print(f"  BASELINE...", end=" ", flush=True)
            sr_baseline = _eval_condition(
                diffusion_model,
                diffusion_checkpoint_dict,
                diffusion_alphas,
                diffusion_alphas_bar,
                env,
                args.horizon,
                args.seed,
                args.n_rollouts,
                args.t_start,
                alpha_s,
                alpha_a,
                prediction_type=prediction_type,
            )
            results[key]["BASELINE (diffusion only)"] = sr_baseline
            print(f"success_rate={sr_baseline:.4f}")

            # With joint denoisers
            for model_key, model_path in sorted(joint_models.items()):
                print(f"  {model_key}...", end=" ", flush=True)
                try:
                    jm, ja, ja_alphas, ja_alphas_bar, ja_norm, ja_h, ja_s_dim, ja_a_dim, ja_obs_keys = _load_joint_model(model_path, device)

                    sr = _eval_condition(
                        diffusion_model,
                        diffusion_checkpoint_dict,
                        diffusion_alphas,
                        diffusion_alphas_bar,
                        env,
                        args.horizon,
                        args.seed,
                        args.n_rollouts,
                        args.t_start,
                        alpha_s,
                        alpha_a,
                        prediction_type=prediction_type,
                        joint_model=jm,
                        joint_anchor=ja,
                        joint_alphas=ja_alphas,
                        joint_alphas_bar=ja_alphas_bar,
                        joint_norm=ja_norm,
                        joint_horizon=ja_h,
                        joint_state_dim=ja_s_dim,
                        joint_action_dim=ja_a_dim,
                        joint_obs_keys=ja_obs_keys,
                    )
                    results[key][model_key] = sr
                    print(f"success_rate={sr:.4f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    results[key][model_key] = None

    # Save results
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

    # Print summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"{'alpha_s':>10}  {'alpha_a':>10}" + "".join(f"  {c:>15}" for c in col_names))
    print("─" * (24 + 17 * len(col_names)))
    for (a_s, a_a), vals in sorted(results.items(), key=lambda x: (float(x[0][0]), float(x[0][1]))):
        row = f"{float(a_s):>10.3f}  {float(a_a):>10.3f}"
        for col in col_names:
            v = vals.get(col)
            row += f"  {v:>15.4f}" if v is not None else f"  {'—':>15}"
        print(row)

    print(f"\nResults saved to: {args.output_csv}")
    print("=" * 90)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
