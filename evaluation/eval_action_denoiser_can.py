#!/usr/bin/env python3
"""
Evaluate Can diffusion policy with action-only denoiser assistance.

Tests baseline (diffusion policy only) vs. action denoiser vs. optional joint denoiser,
under Gaussian perturbation applied to both state and action at deployment.

The action denoiser takes noisy (state, action) windows and recovers a clean action
conditioned on the (noisy) state, without touching the state itself.

Usage:
    python evaluation/eval_action_denoiser_can.py \\
        --diffusion_checkpoint checkpoints/can_diffusion_policy/best_model.pt \\
        --action_denoiser diffusion_models/action_a7_can.pt \\
        --joint_denoiser  diffusion_models/joint_cross_attn_can_a7.pt \\
        --alpha_s 0.0 0.01 0.02 0.03 0.04 0.05 \\
        --alpha_a 0.0 0.05 0.1 0.2 \\
        --t_start 10 20 \\
        --n_rollouts 50 \\
        --output_csv results/can/action_denoiser/results.csv
"""

import argparse
import csv
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
for _p in (PROJECT_ROOT, EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from common.mujoco import configure_mujoco_env
from diffusion.can_policy import DEFAULT_OBS_KEYS, load_can_checkpoint, sample_action_sequence
from diffusion.can_policy_unet import load_unet_checkpoint, sample_action_sequence_x0

DEFAULT_ENV_CKPT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "bc_rnn_can",
    "bc_rnn_can",
    "20260405211805",
    "models",
    "model_epoch_600.pth",
)

DEFAULT_ALPHA_S  = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
DEFAULT_ALPHA_A  = [0.0, 0.05, 0.1, 0.2]
DEFAULT_T_STARTS = [10, 20]


# ── Env and obs helpers ────────────────────────────────────────────────────────

def _load_env(env_ckpt: str):
    import torch
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=env_ckpt, device=device, verbose=False,
    )
    return EnvUtils.create_env_from_metadata(
        env_meta=ckpt_dict["env_metadata"],
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )


def _flatten_obs(obs: dict, obs_keys) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs[k]).reshape(-1) for k in obs_keys]
    ).astype(np.float32)


# ── Checkpoint loaders ─────────────────────────────────────────────────────────

def _load_action_model(ckpt_path: str, device):
    """
    Load an action denoiser checkpoint (arch='action_unet').
    Returns (model, anchor, alphas, alphas_bar, norm, horizon, state_dim, action_dim, obs_keys, anchor_id).
    """
    import torch
    from diffusion.anchors import build_anchor
    from diffusion.model import make_beta_schedule

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert ckpt.get("arch") == "action_unet", (
        f"Expected arch='action_unet', got {ckpt.get('arch')!r}. "
        "Use --joint_denoiser for joint denoiser checkpoints."
    )

    from diffusion.action_denoiser_unet import ActionDenoisingUNet1D
    model = ActionDenoisingUNet1D(
        state_dim     = ckpt["state_dim"],
        action_dim    = ckpt["action_dim"],
        anchor_dim    = ckpt.get("anchor_dim", 128),
        time_emb_dim  = ckpt.get("time_emb_dim", 128),
        channel_sizes = tuple(ckpt.get("channel_sizes", (64, 128, 256))),
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
    alphas     = alphas.to(device)
    alphas_bar = alphas_bar.to(device)

    norm = {
        "state_mean":  _to_device(ckpt["state_mean"],  device),
        "state_std":   _to_device(ckpt["state_std"],   device),
        "action_mean": _to_device(ckpt["action_mean"], device),
        "action_std":  _to_device(ckpt["action_std"],  device),
    }

    return (
        model, anchor, alphas, alphas_bar, norm,
        ckpt["horizon"], ckpt["state_dim"], ckpt["action_dim"],
        list(ckpt.get("obs_keys") or DEFAULT_OBS_KEYS),
        ckpt["anchor_id"],
    )


def _load_joint_model(ckpt_path: str, device):
    """
    Load a joint denoiser checkpoint (arch='cross_attn' or unet).
    Returns (model, anchor, alphas, alphas_bar, norm, horizon, state_dim, action_dim, obs_keys, anchor_id).
    """
    import torch
    from diffusion.anchors import build_anchor
    from diffusion.model import make_beta_schedule

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if ckpt.get("arch") == "cross_attn":
        from diffusion.joint_cross_attn import CrossAttnJointDenoiser
        model = CrossAttnJointDenoiser(
            state_dim  = ckpt["state_dim"],
            action_dim = ckpt["action_dim"],
            anchor_dim = ckpt.get("anchor_dim", 128),
            d_model    = ckpt.get("d_model", 192),
            n_heads    = ckpt.get("n_heads", 6),
            n_layers   = ckpt.get("n_layers", 4),
        )
    else:
        from diffusion.joint_unet import JointUNet1D
        model = JointUNet1D(
            state_dim     = ckpt["state_dim"],
            action_dim    = ckpt["action_dim"],
            anchor_dim    = ckpt.get("anchor_dim", 128),
            time_emb_dim  = ckpt.get("time_emb_dim", 128),
            channel_sizes = tuple(ckpt.get("channel_sizes", (64, 128, 256))),
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
    alphas     = alphas.to(device)
    alphas_bar = alphas_bar.to(device)

    norm = {
        "state_mean":  _to_device(ckpt["state_mean"],  device),
        "state_std":   _to_device(ckpt["state_std"],   device),
        "action_mean": _to_device(ckpt["action_mean"], device),
        "action_std":  _to_device(ckpt["action_std"],  device),
    }

    return (
        model, anchor, alphas, alphas_bar, norm,
        ckpt["horizon"], ckpt["state_dim"], ckpt["action_dim"],
        list(ckpt.get("obs_keys") or DEFAULT_OBS_KEYS),
        ckpt["anchor_id"],
    )


def _to_device(arr, device):
    import torch
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)


# ── Anchor dict builder (shared across rollout types) ─────────────────────────

def _build_anchor_dict(obs, action, gripper_hist_buf, device):
    """Build the traj dict expected by Anchor.compute from current obs/action."""
    import torch

    _closed  = float(action[6]) < 0.5
    _obj_z   = float(obs["object"][2]) if "object" in obs else 0.0
    _phase   = 2 if (_closed and _obj_z > 0.85) else (1 if _closed else 0)

    object_pose_t0 = np.asarray(obs.get("object", np.zeros(14)), dtype=np.float32)
    gh_arr = np.array(
        [0.0] * (5 - len(gripper_hist_buf)) + list(gripper_hist_buf), dtype=np.float32,
    )
    proprio_arr = np.concatenate([
        np.asarray(obs.get("robot0_joint_pos", np.zeros(7)),  dtype=np.float32).flatten(),
        np.asarray(obs.get("robot0_eef_pos",   np.zeros(3)),  dtype=np.float32).flatten(),
        np.asarray(obs.get("robot0_eef_quat",  np.zeros(4)),  dtype=np.float32).flatten(),
    ]).astype(np.float32)

    return {
        "object_pose_t0":  torch.from_numpy(object_pose_t0).float().unsqueeze(0).to(device),
        "gripper_history": torch.from_numpy(gh_arr).float().unsqueeze(0).to(device),
        "proprio":         torch.from_numpy(proprio_arr).float().unsqueeze(0).to(device),
        "phase":           torch.tensor([_phase], dtype=torch.long).to(device),
    }


# ── Rollout implementations ────────────────────────────────────────────────────

def _run_rollout_baseline(
    diffusion_model,
    dp: dict,
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
    import torch

    device     = next(diffusion_model.parameters()).device
    obs_keys   = dp["obs_keys"]
    obs_mean   = dp["obs_mean"]
    obs_std    = dp["obs_std"]
    act_mean   = dp["act_mean"]
    act_std    = dp["act_std"]
    obs_hor    = dp["obs_hor"]
    act_hor    = dp["act_hor"]
    diff_steps = dp["diff_steps"]

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = env.reset()

    history = deque([_flatten_obs(obs, obs_keys)] * obs_hor, maxlen=obs_hor)

    step = 0
    while step < horizon:
        if alpha_s > 0.0:
            noisy_obs = {
                k: np.asarray(obs[k], dtype=np.float32) + rng.normal(0, alpha_s, size=np.asarray(obs[k]).shape).astype(np.float32)
                if k in obs else obs[k]
                for k in obs
            }
            history.append(_flatten_obs(noisy_obs, obs_keys))
        else:
            history.append(_flatten_obs(obs, obs_keys))

        obs_hist = np.stack(history, axis=0)
        kwargs   = dict(
            model=diffusion_model, obs_history=obs_hist,
            obs_mean=obs_mean, obs_std=obs_std,
            action_mean=act_mean, action_std=act_std,
            alphas=diffusion_alphas, alphas_bar=diffusion_alphas_bar,
            diffusion_steps=diff_steps, t_start=t_start_diffusion, device=device,
        )
        action_chunk = (
            sample_action_sequence_x0(**kwargs)
            if prediction_type == "x0" and sample_action_sequence_x0 is not None
            else sample_action_sequence(**kwargs)
        )

        for action in action_chunk[:act_hor]:
            if step >= horizon:
                break
            if alpha_a > 0.0:
                action = action + rng.normal(0, alpha_a, size=action.shape).astype(np.float32)
            obs, _, done, _ = env.step(np.clip(action, -1.0, 1.0))
            step += 1
            history.append(_flatten_obs(obs, obs_keys))
            if env.is_success()["task"]:
                return True
            if done:
                return False

    return False


def _run_rollout_action_denoiser(
    diffusion_model,
    dp: dict,
    diffusion_alphas,
    diffusion_alphas_bar,
    env,
    horizon: int,
    seed: int,
    t_start_diffusion: Optional[int],
    action_model,
    action_anchor,
    action_alphas,
    action_alphas_bar,
    action_norm: dict,
    action_horizon: int,
    action_state_dim: int,
    action_action_dim: int,
    action_obs_keys: List[str],
    alpha_s: float = 0.0,
    alpha_a: float = 0.0,
    denoiser_t_start: int = 10,
    prediction_type: str = "epsilon",
) -> bool:
    """
    Rollout with diffusion policy + action-only denoiser (chunk-level).

    Workflow per chunk:
    1. Query diffusion policy for action_chunk (act_hor actions)
    2. Corrupt the full chunk with action noise
    3. Reuse the noisy obs (already built for policy input) as denoiser state context
    4. Run action_denoise ONCE on the full chunk — not once per step
    5. Execute each denoised action, updating rolling buffers
    """
    import torch
    from diffusion.action_denoiser_unet import action_denoise

    device        = next(diffusion_model.parameters()).device
    obs_keys      = dp["obs_keys"]
    obs_mean      = dp["obs_mean"]
    obs_std       = dp["obs_std"]
    act_mean      = dp["act_mean"]
    act_std       = dp["act_std"]
    obs_hor       = dp["obs_hor"]
    act_hor       = dp["act_hor"]
    diff_steps    = dp["diff_steps"]
    state_mean_np = action_norm["state_mean"].cpu().numpy()
    state_std_np  = action_norm["state_std"].cpu().numpy()
    act_mean_np   = action_norm["action_mean"].cpu().numpy()
    act_std_np    = action_norm["action_std"].cpu().numpy()

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = env.reset()

    history          = deque([_flatten_obs(obs, obs_keys)] * obs_hor, maxlen=obs_hor)
    state_buf        = deque(maxlen=action_horizon)
    gripper_hist_buf = deque(maxlen=5)

    step = 0
    while step < horizon:
        # Build noisy obs once — reused for both the policy history and the denoiser state
        if alpha_s > 0.0:
            noisy_obs_dict = {
                k: np.asarray(obs[k], dtype=np.float32) + rng.normal(0, alpha_s, size=np.asarray(obs[k]).shape).astype(np.float32)
                if k in obs else obs[k]
                for k in obs
            }
        else:
            noisy_obs_dict = obs
        history.append(_flatten_obs(noisy_obs_dict, obs_keys))

        obs_hist = np.stack(history, axis=0)
        kwargs   = dict(
            model=diffusion_model, obs_history=obs_hist,
            obs_mean=obs_mean, obs_std=obs_std,
            action_mean=act_mean, action_std=act_std,
            alphas=diffusion_alphas, alphas_bar=diffusion_alphas_bar,
            diffusion_steps=diff_steps, t_start=t_start_diffusion, device=device,
        )
        action_chunk = (
            sample_action_sequence_x0(**kwargs)
            if prediction_type == "x0" and sample_action_sequence_x0 is not None
            else sample_action_sequence(**kwargs)
        )
        chunk = action_chunk[:act_hor]  # (act_hor, Da)

        # Corrupt the whole chunk at once
        if alpha_a > 0.0:
            noisy_chunk = chunk + rng.normal(0, alpha_a, chunk.shape).astype(np.float32)
        else:
            noisy_chunk = chunk.copy()
        noisy_chunk = np.clip(noisy_chunk, -1.0, 1.0)

        # Build state window from rolling buffer + current obs
        state_vec   = _flatten_obs(noisy_obs_dict, action_obs_keys)
        state_list  = list(state_buf) + [state_vec]
        state_list  = state_list[-action_horizon:]
        pad         = action_horizon - len(state_list)
        if pad > 0:
            state_list = [state_list[0]] * pad + state_list
        states_win  = np.stack(state_list, axis=0)   # (H, Ds)

        # Build action window: pad front with first noisy action if chunk < horizon
        if act_hor < action_horizon:
            a_pad       = action_horizon - act_hor
            actions_win = np.concatenate([np.stack([noisy_chunk[0]] * a_pad), noisy_chunk], axis=0)
        else:
            actions_win = noisy_chunk                 # (H, Da)

        # Normalize and run denoiser ONCE for the whole chunk
        s_t = torch.from_numpy((states_win  - state_mean_np) / state_std_np).float().unsqueeze(0).to(device)
        a_t = torch.from_numpy((actions_win - act_mean_np)   / act_std_np).float().unsqueeze(0).to(device)

        traj = _build_anchor_dict(obs, noisy_chunk[0], gripper_hist_buf, device)
        with torch.no_grad():
            anchor_emb = action_anchor.compute(traj)
            denoised   = action_denoise(
                action_model, a_t, s_t, anchor_emb,
                action_alphas, action_alphas_bar,
                t_start=denoiser_t_start,
            )  # (1, H, Da)

        # Denormalize — take last act_hor timesteps from the denoised window
        denoised_np = denoised[0, -act_hor:].cpu().numpy() * act_std_np + act_mean_np
        denoised_np = np.clip(denoised_np, -1.0, 1.0)

        # Execute all denoised actions in the chunk
        for denoised_action in denoised_np:
            if step >= horizon:
                break
            obs, _, done, _ = env.step(denoised_action)
            step += 1
            state_buf.append(_flatten_obs(obs, action_obs_keys))
            gripper_hist_buf.append(float(denoised_action[6]))
            history.append(_flatten_obs(obs, obs_keys))
            if env.is_success()["task"]:
                return True
            if done:
                return False

    return False


def _run_rollout_joint_denoiser(
    diffusion_model,
    dp: dict,
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
    joint_obs_keys: List[str],
    alpha_s: float = 0.0,
    alpha_a: float = 0.0,
    denoiser_t_start: int = 10,
    prediction_type: str = "epsilon",
) -> bool:
    """Rollout with diffusion policy + joint (state+action) denoiser (chunk-level)."""
    import torch
    from diffusion.joint_unet import joint_denoise
    from diffusion.joint_cross_attn import cross_attn_joint_denoise, CrossAttnJointDenoiser

    device        = next(diffusion_model.parameters()).device
    obs_keys      = dp["obs_keys"]
    obs_mean      = dp["obs_mean"]
    obs_std       = dp["obs_std"]
    act_mean      = dp["act_mean"]
    act_std       = dp["act_std"]
    obs_hor       = dp["obs_hor"]
    act_hor       = dp["act_hor"]
    diff_steps    = dp["diff_steps"]
    state_mean_np = joint_norm["state_mean"].cpu().numpy()
    state_std_np  = joint_norm["state_std"].cpu().numpy()
    act_mean_np   = joint_norm["action_mean"].cpu().numpy()
    act_std_np    = joint_norm["action_std"].cpu().numpy()

    np.random.seed(seed)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    obs = env.reset()

    history          = deque([_flatten_obs(obs, obs_keys)] * obs_hor, maxlen=obs_hor)
    state_buf        = deque(maxlen=joint_horizon)
    gripper_hist_buf = deque(maxlen=5)

    _denoise_fn = (
        cross_attn_joint_denoise
        if isinstance(joint_model, CrossAttnJointDenoiser)
        else joint_denoise
    )

    step = 0
    while step < horizon:
        # Build noisy obs once — reused for policy input and denoiser state context
        if alpha_s > 0.0:
            noisy_obs_dict = {
                k: np.asarray(obs[k], dtype=np.float32) + rng.normal(0, alpha_s, size=np.asarray(obs[k]).shape).astype(np.float32)
                if k in obs else obs[k]
                for k in obs
            }
        else:
            noisy_obs_dict = obs
        history.append(_flatten_obs(noisy_obs_dict, obs_keys))

        obs_hist = np.stack(history, axis=0)
        kwargs   = dict(
            model=diffusion_model, obs_history=obs_hist,
            obs_mean=obs_mean, obs_std=obs_std,
            action_mean=act_mean, action_std=act_std,
            alphas=diffusion_alphas, alphas_bar=diffusion_alphas_bar,
            diffusion_steps=diff_steps, t_start=t_start_diffusion, device=device,
        )
        action_chunk = (
            sample_action_sequence_x0(**kwargs)
            if prediction_type == "x0" and sample_action_sequence_x0 is not None
            else sample_action_sequence(**kwargs)
        )
        chunk = action_chunk[:act_hor]

        # Corrupt the whole chunk at once
        if alpha_a > 0.0:
            noisy_chunk = chunk + rng.normal(0, alpha_a, chunk.shape).astype(np.float32)
        else:
            noisy_chunk = chunk.copy()
        noisy_chunk = np.clip(noisy_chunk, -1.0, 1.0)

        # Build state window from rolling buffer + current obs
        state_vec  = _flatten_obs(noisy_obs_dict, joint_obs_keys)
        state_list = list(state_buf) + [state_vec]
        state_list = state_list[-joint_horizon:]
        pad        = joint_horizon - len(state_list)
        if pad > 0:
            state_list = [state_list[0]] * pad + state_list
        states_win = np.stack(state_list, axis=0)   # (H, Ds)

        if act_hor < joint_horizon:
            a_pad       = joint_horizon - act_hor
            actions_win = np.concatenate([np.stack([noisy_chunk[0]] * a_pad), noisy_chunk], axis=0)
        else:
            actions_win = noisy_chunk

        s_t = torch.from_numpy((states_win  - state_mean_np) / state_std_np).float().unsqueeze(0).to(device)
        a_t = torch.from_numpy((actions_win - act_mean_np)   / act_std_np).float().unsqueeze(0).to(device)

        traj = _build_anchor_dict(obs, noisy_chunk[0], gripper_hist_buf, device)
        with torch.no_grad():
            anchor_emb = joint_anchor.compute(traj)
            _, clean_a = _denoise_fn(
                joint_model, s_t, a_t, anchor_emb,
                joint_alphas, joint_alphas_bar,
                t_start=denoiser_t_start,
            )

        denoised_np = clean_a[0, -act_hor:].cpu().numpy() * act_std_np + act_mean_np
        denoised_np = np.clip(denoised_np, -1.0, 1.0)

        for denoised_action in denoised_np:
            if step >= horizon:
                break
            obs, _, done, _ = env.step(denoised_action)
            step += 1
            state_buf.append(_flatten_obs(obs, joint_obs_keys))
            gripper_hist_buf.append(float(denoised_action[6]))
            history.append(_flatten_obs(obs, obs_keys))
            if env.is_success()["task"]:
                return True
            if done:
                return False

    return False


# ── Grid evaluation ────────────────────────────────────────────────────────────

def _eval_grid(
    diffusion_model,
    diffusion_checkpoint,
    diffusion_alphas,
    diffusion_alphas_bar,
    env,
    horizon: int,
    base_seed: int,
    n_rollouts: int,
    t_start_diffusion: Optional[int],
    prediction_type: str,
    alpha_s_list: List[float],
    alpha_a_list: List[float],
    t_start_list: List[int],
    action_models: List[dict],
    joint_models: List[dict],
) -> List[dict]:
    """Run the full evaluation grid; return list of result dicts."""
    # Pre-compute diffusion policy constants once — not per rollout
    dp = {
        "obs_keys":   list(diffusion_checkpoint.get("obs_keys") or DEFAULT_OBS_KEYS),
        "obs_mean":   np.asarray(diffusion_checkpoint["obs_mean"],    dtype=np.float32),
        "obs_std":    np.asarray(diffusion_checkpoint["obs_std"],     dtype=np.float32),
        "act_mean":   np.asarray(diffusion_checkpoint["action_mean"], dtype=np.float32),
        "act_std":    np.asarray(diffusion_checkpoint["action_std"],  dtype=np.float32),
        "obs_hor":    int(diffusion_checkpoint["obs_horizon"]),
        "act_hor":    int(diffusion_checkpoint["action_horizon"]),
        "diff_steps": int(diffusion_checkpoint["diffusion_steps"]),
    }

    results = []

    for alpha_s in alpha_s_list:
        for alpha_a in alpha_a_list:
            # Baseline (no denoiser) — t_start doesn't apply
            print(f"  baseline  alpha_s={alpha_s:.3f} alpha_a={alpha_a:.3f} ...", end=" ", flush=True)
            t0 = time.time()
            successes = [
                _run_rollout_baseline(
                    diffusion_model, dp,
                    diffusion_alphas, diffusion_alphas_bar,
                    env, horizon, base_seed + i, t_start_diffusion,
                    alpha_s=alpha_s, alpha_a=alpha_a,
                    prediction_type=prediction_type,
                )
                for i in range(n_rollouts)
            ]
            sr = float(np.mean(successes))
            print(f"sr={sr:.3f}  ({time.time()-t0:.0f}s)")
            results.append({
                "model_type": "baseline", "anchor_id": "none",
                "alpha_s": alpha_s, "alpha_a": alpha_a,
                "t_start": "N/A", "n_rollouts": n_rollouts, "success_rate": sr,
            })

            # Action denoiser variants
            for am in action_models:
                for t_start in t_start_list:
                    label = am["label"]
                    print(f"  action_denoiser ({label})  alpha_s={alpha_s:.3f} alpha_a={alpha_a:.3f} t_start={t_start} ...", end=" ", flush=True)
                    t0 = time.time()
                    successes = [
                        _run_rollout_action_denoiser(
                            diffusion_model, dp,
                            diffusion_alphas, diffusion_alphas_bar,
                            env, horizon, base_seed + i, t_start_diffusion,
                            am["model"], am["anchor"], am["alphas"], am["alphas_bar"],
                            am["norm"], am["horizon"], am["state_dim"], am["action_dim"],
                            am["obs_keys"],
                            alpha_s=alpha_s, alpha_a=alpha_a,
                            denoiser_t_start=t_start,
                            prediction_type=prediction_type,
                        )
                        for i in range(n_rollouts)
                    ]
                    sr = float(np.mean(successes))
                    print(f"sr={sr:.3f}  ({time.time()-t0:.0f}s)")
                    results.append({
                        "model_type": f"action_denoiser_{label}",
                        "anchor_id": am["anchor_id"],
                        "alpha_s": alpha_s, "alpha_a": alpha_a,
                        "t_start": t_start, "n_rollouts": n_rollouts, "success_rate": sr,
                    })

            # Joint denoiser variants (comparison)
            for jm in joint_models:
                for t_start in t_start_list:
                    label = jm["label"]
                    print(f"  joint_denoiser ({label})  alpha_s={alpha_s:.3f} alpha_a={alpha_a:.3f} t_start={t_start} ...", end=" ", flush=True)
                    t0 = time.time()
                    successes = [
                        _run_rollout_joint_denoiser(
                            diffusion_model, dp,
                            diffusion_alphas, diffusion_alphas_bar,
                            env, horizon, base_seed + i, t_start_diffusion,
                            jm["model"], jm["anchor"], jm["alphas"], jm["alphas_bar"],
                            jm["norm"], jm["horizon"], jm["state_dim"], jm["action_dim"],
                            jm["obs_keys"],
                            alpha_s=alpha_s, alpha_a=alpha_a,
                            denoiser_t_start=t_start,
                            prediction_type=prediction_type,
                        )
                        for i in range(n_rollouts)
                    ]
                    sr = float(np.mean(successes))
                    print(f"sr={sr:.3f}  ({time.time()-t0:.0f}s)")
                    results.append({
                        "model_type": f"joint_denoiser_{label}",
                        "anchor_id": jm["anchor_id"],
                        "alpha_s": alpha_s, "alpha_a": alpha_a,
                        "t_start": t_start, "n_rollouts": n_rollouts, "success_rate": sr,
                    })

    return results


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--diffusion_checkpoint", type=str, required=True,
                   help="Can diffusion policy checkpoint (.pt)")
    p.add_argument("--env_ckpt", type=str, default=DEFAULT_ENV_CKPT,
                   help="BC-RNN checkpoint for env creation")
    p.add_argument("--action_denoiser", type=str, nargs="+", default=[],
                   help="Action denoiser checkpoint(s) (arch=action_unet)")
    p.add_argument("--joint_denoiser", type=str, nargs="*", default=[],
                   help="Joint denoiser checkpoint(s) for comparison (optional)")
    p.add_argument("--alpha_s", type=float, nargs="+", default=DEFAULT_ALPHA_S,
                   help="State noise std levels to sweep")
    p.add_argument("--alpha_a", type=float, nargs="+", default=DEFAULT_ALPHA_A,
                   help="Action noise std levels to sweep")
    p.add_argument("--t_start", type=int, nargs="+", default=DEFAULT_T_STARTS,
                   help="Denoiser reverse-diffusion start steps to sweep")
    p.add_argument("--n_rollouts", type=int, default=50)
    p.add_argument("--horizon", type=int, default=400,
                   help="Max steps per episode")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--t_start_diffusion", type=int, default=None,
                   help="Diffusion policy sampling t_start (None = full)")
    p.add_argument("--output_csv", type=str,
                   default="results/can/action_denoiser/results.csv")
    return p.parse_args()


def main():
    import torch

    args = parse_args()
    configure_mujoco_env()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load diffusion policy — auto-detect UNet vs MLP backbone
    print(f"\nLoading diffusion policy: {args.diffusion_checkpoint}")
    _peek = torch.load(args.diffusion_checkpoint, map_location="cpu", weights_only=False)
    if _peek.get("backbone") == "unet":
        diffusion_model, diffusion_checkpoint, diffusion_alphas, diffusion_alphas_bar = \
            load_unet_checkpoint(args.diffusion_checkpoint, device)
        print("  -> UNet backbone")
    else:
        diffusion_model, diffusion_checkpoint, diffusion_alphas, diffusion_alphas_bar = \
            load_can_checkpoint(args.diffusion_checkpoint, device)
        print("  -> MLP backbone")

    diffusion_alphas     = diffusion_alphas.to(device)
    diffusion_alphas_bar = diffusion_alphas_bar.to(device)
    prediction_type      = diffusion_checkpoint.get("prediction_type", "epsilon")
    print(f"  prediction_type: {prediction_type}")

    # Load action denoiser(s)
    action_models = []
    for i, ckpt_path in enumerate(args.action_denoiser):
        print(f"\nLoading action denoiser: {ckpt_path}")
        result = _load_action_model(ckpt_path, device)
        model, anchor, alphas, alphas_bar, norm, hor, sdim, adim, obs_keys, anchor_id = result
        label = f"{anchor_id}_{i}" if len(args.action_denoiser) > 1 else anchor_id
        action_models.append({
            "model": model, "anchor": anchor,
            "alphas": alphas, "alphas_bar": alphas_bar,
            "norm": norm, "horizon": hor,
            "state_dim": sdim, "action_dim": adim,
            "obs_keys": obs_keys, "anchor_id": anchor_id, "label": label,
        })
        print(f"  anchor_id: {anchor_id}  state_dim: {sdim}  action_dim: {adim}  horizon: {hor}")

    # Load joint denoiser(s)
    joint_models = []
    for i, ckpt_path in enumerate(args.joint_denoiser or []):
        print(f"\nLoading joint denoiser: {ckpt_path}")
        result = _load_joint_model(ckpt_path, device)
        model, anchor, alphas, alphas_bar, norm, hor, sdim, adim, obs_keys, anchor_id = result
        label = f"{anchor_id}_{i}" if len(args.joint_denoiser) > 1 else anchor_id
        joint_models.append({
            "model": model, "anchor": anchor,
            "alphas": alphas, "alphas_bar": alphas_bar,
            "norm": norm, "horizon": hor,
            "state_dim": sdim, "action_dim": adim,
            "obs_keys": obs_keys, "anchor_id": anchor_id, "label": label,
        })
        print(f"  anchor_id: {anchor_id}  state_dim: {sdim}  action_dim: {adim}  horizon: {hor}")

    if not action_models and not joint_models:
        print("\nWarning: no denoiser checkpoints provided — running baseline only.")

    # Load environment
    print(f"\nLoading environment from: {args.env_ckpt}")
    env = _load_env(args.env_ckpt)

    print(f"\nEvaluation grid")
    print(f"  alpha_s:   {args.alpha_s}")
    print(f"  alpha_a:   {args.alpha_a}")
    print(f"  t_start:   {args.t_start}")
    print(f"  n_rollouts:{args.n_rollouts}")
    print(f"  horizon:   {args.horizon}")
    print(f"  output:    {args.output_csv}\n")

    results = _eval_grid(
        diffusion_model, diffusion_checkpoint,
        diffusion_alphas, diffusion_alphas_bar,
        env, args.horizon, args.base_seed, args.n_rollouts,
        args.t_start_diffusion, prediction_type,
        args.alpha_s, args.alpha_a, args.t_start,
        action_models, joint_models,
    )

    # Save CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    fieldnames = ["model_type", "anchor_id", "alpha_s", "alpha_a",
                  "t_start", "n_rollouts", "success_rate"]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {args.output_csv}")
    print(f"  {len(results)} rows written")

    # Summary table
    print("\n── Summary ────────────────────────────────────────────────────")
    print(f"{'model_type':<35} {'anchor':>6} {'alpha_s':>8} {'alpha_a':>8} {'t_start':>7} {'sr':>6}")
    print("-" * 76)
    for r in results:
        print(
            f"{r['model_type']:<35} {r['anchor_id']:>6} {r['alpha_s']:>8.3f} "
            f"{r['alpha_a']:>8.3f} {str(r['t_start']):>7} {r['success_rate']:>6.3f}"
        )


if __name__ == "__main__":
    main()
