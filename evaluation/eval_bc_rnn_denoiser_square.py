#!/usr/bin/env python3
"""
Evaluate BC RNN policy on the Square task with optional action and joint denoisers.

Tests three modes:
  baseline         — pure BC RNN, no denoiser
  action_denoiser  — BC RNN actions refined by ActionDenoisingUNet1D on a sliding window
  joint_denoiser   — BC RNN actions refined by JointUNet1D / CrossAttnJointDenoiser

Gaussian noise square be injected into observations (alpha_s) and actions (alpha_a)
to simulate deployment perturbations.

Usage:
    python evaluation/eval_bc_rnn_denoiser_square.py \\
        --bc_rnn_checkpoint checkpoints/bc_rnn_square/.../model_epoch_600.pth \\
        --action_denoiser   diffusion_models/action_a0_square.pt \\
        --joint_denoiser    diffusion_models/joint_a0_square.pt \\
        --alpha_s 0.0 0.02 0.05 \\
        --alpha_a 0.0 0.3 \\
        --t_start 5 10 \\
        --n_rollouts 30 \\
        --output_csv results/square/bc_rnn_denoiser/bc_rnn_a0_results.csv
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

DEFAULT_ALPHA_S  = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05]
DEFAULT_ALPHA_A  = [0.0, 0.05, 0.1, 0.2]
DEFAULT_T_STARTS = [10, 20]
DEFAULT_OBS_KEYS = [
    "robot0_joint_pos", "robot0_joint_vel",
    "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos",
    "object",
]

_PROPRIO_KEYS      = ["robot0_joint_pos", "robot0_eef_pos", "robot0_eef_quat"]
_GRIPPER_ACTION_DIM = 6   # index of gripper command in 7-dim Square action
_SQUARE_Z_THRESH      = 0.85


# ── BC RNN + env loading ───────────────────────────────────────────────────────

def _load_policy_and_env(bc_ckpt_path: str, device_str: str = "auto"):
    import torch
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    try:
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=bc_ckpt_path, device=device, verbose=False
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device.type == "cuda":
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            policy, ckpt_dict = FileUtils.policy_from_checkpoint(
                ckpt_path=bc_ckpt_path, device=device, verbose=False
            )
        else:
            raise

    env_meta  = ckpt_dict["env_metadata"]
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False
    )

    obs_keys = list(ckpt_dict["shape_metadata"]["all_obs_keys"])
    return policy, env, obs_keys, ckpt_dict


# ── Checkpoint loaders (denoiser code untouched) ──────────────────────────────

def _to_device(arr, device):
    import torch
    return torch.from_numpy(np.asarray(arr, dtype=np.float32)).to(device)


def _load_action_model(ckpt_path: str, device):
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
        ckpt.get("gripper_k", 5),
    )


def _load_joint_model(ckpt_path: str, device):
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
        ckpt.get("gripper_k", 5),
    )


# ── Observation helpers ────────────────────────────────────────────────────────

def _flatten_obs(obs_dict: dict, obs_keys: list) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs_dict[k]).flatten() for k in obs_keys]
    ).astype(np.float32)


def _corrupt_obs(obs_dict: dict, obs_keys: list, alpha_s: float, rng: np.random.Generator) -> dict:
    if alpha_s == 0.0:
        return obs_dict
    noisy = dict(obs_dict)
    for key in obs_keys:
        arr = np.asarray(obs_dict[key], dtype=np.float32)
        noisy[key] = arr + rng.normal(0.0, alpha_s, size=arr.shape).astype(np.float32)
    return noisy


def _build_anchor_dict(obs, action, object_pose_t0, gripper_hist_buf, gripper_k, device):
    import torch

    _closed = float(action[_GRIPPER_ACTION_DIM]) < 0.5
    _obj_z  = float(obs["object"][2]) if "object" in obs else 0.0
    _phase  = 2 if (_closed and _obj_z > _SQUARE_Z_THRESH) else (1 if _closed else 0)

    hist   = list(gripper_hist_buf)
    gh_arr = np.array(
        [0.0] * (gripper_k - len(hist)) + hist, dtype=np.float32
    )
    proprio_parts = [
        np.asarray(obs[k], dtype=np.float32).flatten()
        for k in _PROPRIO_KEYS if k in obs
    ]
    proprio_arr = (
        np.concatenate(proprio_parts) if proprio_parts
        else np.zeros(14, dtype=np.float32)
    )

    return {
        "object_pose_t0":  torch.from_numpy(object_pose_t0).float().unsqueeze(0).to(device),
        "gripper_history": torch.from_numpy(gh_arr).float().unsqueeze(0).to(device),
        "proprio":         torch.from_numpy(proprio_arr).float().unsqueeze(0).to(device),
        "phase":           torch.tensor([_phase], dtype=torch.long).to(device),
    }


# ── Rollout implementations ────────────────────────────────────────────────────

def _run_rollout_baseline(
    policy,
    env,
    obs_keys: list,
    alpha_s: float,
    alpha_a: float,
    episode_horizon: int,
    seed: int,
) -> bool:
    import torch

    rng = np.random.default_rng(seed)
    obs = env.reset()
    policy.start_episode()

    for _ in range(episode_horizon):
        noisy_obs = _corrupt_obs(obs, obs_keys, alpha_s, rng)
        with torch.no_grad():
            ac = policy(noisy_obs)
            if isinstance(ac, torch.Tensor):
                ac = ac.cpu().numpy()
            ac = np.asarray(ac).flatten().astype(np.float32)
        if alpha_a > 0.0:
            ac = ac + rng.normal(0.0, alpha_a, size=ac.shape).astype(np.float32)
        ac = np.clip(ac, -1.0, 1.0)
        obs, _, done, _ = env.step(ac)
        if env.is_success()["task"]:
            return True
        if done:
            return False

    return False


def _run_rollout_action_denoiser(
    policy,
    env,
    obs_keys: list,
    model,
    anchor,
    alphas,
    alphas_bar,
    norm: dict,
    horizon_H: int,
    state_dim: int,
    action_dim: int,
    alpha_s: float,
    alpha_a: float,
    t_start: int,
    episode_horizon: int,
    seed: int,
    gripper_k: int = 5,
) -> bool:
    """BC RNN step-by-step + action denoiser on sliding window of size horizon_H."""
    import torch
    from diffusion.action_denoiser_unet import action_denoise

    dev = alphas.device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    obs = env.reset()
    policy.start_episode()

    object_pose_t0   = np.asarray(obs["object"], dtype=np.float32).copy()
    state_buf        = deque(maxlen=horizon_H)
    action_buf       = deque(maxlen=horizon_H)
    gripper_hist_buf = deque(maxlen=gripper_k)

    state_mean_np = norm["state_mean"].cpu().numpy()
    state_std_np  = norm["state_std"].cpu().numpy()
    act_mean_np   = norm["action_mean"].cpu().numpy()
    act_std_np    = norm["action_std"].cpu().numpy()

    for _ in range(episode_horizon):
        noisy_obs = _corrupt_obs(obs, obs_keys, alpha_s, rng)

        with torch.no_grad():
            ac = policy(noisy_obs)
            if isinstance(ac, torch.Tensor):
                ac = ac.cpu().numpy()
            ac = np.asarray(ac).flatten().astype(np.float32)

        if alpha_a > 0.0:
            ac = ac + rng.normal(0.0, alpha_a, size=ac.shape).astype(np.float32)

        gripper_hist_buf.append(float(ac[_GRIPPER_ACTION_DIM]))

        state_vec = _flatten_obs(noisy_obs, obs_keys)
        state_buf.append(state_vec)
        action_buf.append(ac)

        # Pad window to H if not enough history yet
        pad         = horizon_H - len(state_buf)
        states_win  = np.stack([state_buf[0]] * pad + list(state_buf), axis=0)   # (H, Ds)
        actions_win = np.stack([action_buf[0]] * pad + list(action_buf), axis=0) # (H, Da)

        s_norm = (states_win  - state_mean_np) / state_std_np
        a_norm = (actions_win - act_mean_np)   / act_std_np

        s_t = torch.from_numpy(s_norm).float().unsqueeze(0).to(dev)  # (1, H, Ds)
        a_t = torch.from_numpy(a_norm).float().unsqueeze(0).to(dev)  # (1, H, Da)

        traj = _build_anchor_dict(obs, ac, object_pose_t0, gripper_hist_buf, gripper_k, dev)
        with torch.no_grad():
            anchor_emb = anchor.compute(traj)
            clean_a = action_denoise(
                model, a_t, s_t, anchor_emb, alphas, alphas_bar, t_start=t_start,
            )  # (1, H, Da)

        # Use the denoised action at the last position in the window (current step)
        clean_a_np = clean_a[0, -1].cpu().numpy()
        clean_a_np = clean_a_np * act_std_np + act_mean_np
        clean_a_np = np.clip(clean_a_np, -1.0, 1.0)

        obs, _, done, _ = env.step(clean_a_np)
        if env.is_success()["task"]:
            return True
        if done:
            return False

    return False


def _run_rollout_joint_denoiser(
    policy,
    env,
    obs_keys: list,
    model,
    anchor,
    alphas,
    alphas_bar,
    norm: dict,
    horizon_H: int,
    state_dim: int,
    action_dim: int,
    alpha_s: float,
    alpha_a: float,
    t_start: int,
    episode_horizon: int,
    seed: int,
    gripper_k: int = 5,
) -> bool:
    """BC RNN step-by-step + joint denoiser on sliding window of size horizon_H."""
    import torch
    from diffusion.joint_unet import joint_denoise

    try:
        from diffusion.joint_cross_attn import cross_attn_joint_denoise, CrossAttnJointDenoiser
        _has_cross_attn = True
    except ImportError:
        _has_cross_attn = False

    use_cross_attn = _has_cross_attn and isinstance(
        model, CrossAttnJointDenoiser if _has_cross_attn else type(None)
    )

    dev = alphas.device
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    obs = env.reset()
    policy.start_episode()

    object_pose_t0   = np.asarray(obs["object"], dtype=np.float32).copy()
    state_buf        = deque(maxlen=horizon_H)
    action_buf       = deque(maxlen=horizon_H)
    gripper_hist_buf = deque(maxlen=gripper_k)

    state_mean_np = norm["state_mean"].cpu().numpy()
    state_std_np  = norm["state_std"].cpu().numpy()
    act_mean_np   = norm["action_mean"].cpu().numpy()
    act_std_np    = norm["action_std"].cpu().numpy()

    for _ in range(episode_horizon):
        noisy_obs = _corrupt_obs(obs, obs_keys, alpha_s, rng)

        with torch.no_grad():
            ac = policy(noisy_obs)
            if isinstance(ac, torch.Tensor):
                ac = ac.cpu().numpy()
            ac = np.asarray(ac).flatten().astype(np.float32)

        if alpha_a > 0.0:
            ac = ac + rng.normal(0.0, alpha_a, size=ac.shape).astype(np.float32)

        gripper_hist_buf.append(float(ac[_GRIPPER_ACTION_DIM]))

        state_vec = _flatten_obs(noisy_obs, obs_keys)
        state_buf.append(state_vec)
        action_buf.append(ac)

        pad         = horizon_H - len(state_buf)
        states_win  = np.stack([state_buf[0]] * pad + list(state_buf), axis=0)
        actions_win = np.stack([action_buf[0]] * pad + list(action_buf), axis=0)

        s_norm = (states_win  - state_mean_np) / state_std_np
        a_norm = (actions_win - act_mean_np)   / act_std_np

        s_t = torch.from_numpy(s_norm).float().unsqueeze(0).to(dev)
        a_t = torch.from_numpy(a_norm).float().unsqueeze(0).to(dev)

        traj = _build_anchor_dict(obs, ac, object_pose_t0, gripper_hist_buf, gripper_k, dev)
        with torch.no_grad():
            anchor_emb = anchor.compute(traj)
            if use_cross_attn:
                _, clean_a = cross_attn_joint_denoise(
                    model, s_t, a_t, anchor_emb, alphas, alphas_bar, t_start=t_start,
                )
            else:
                _, clean_a = joint_denoise(
                    model, s_t, a_t, anchor_emb, alphas, alphas_bar, t_start=t_start,
                )

        clean_a_np = clean_a[0, -1].cpu().numpy()
        clean_a_np = clean_a_np * act_std_np + act_mean_np
        clean_a_np = np.clip(clean_a_np, -1.0, 1.0)

        obs, _, done, _ = env.step(clean_a_np)
        if env.is_success()["task"]:
            return True
        if done:
            return False

    return False


# ── Grid evaluation ────────────────────────────────────────────────────────────

def _eval_grid(
    policy,
    env,
    obs_keys: list,
    alpha_s_list: List[float],
    alpha_a_list: List[float],
    t_start_list: List[int],
    n_rollouts: int,
    episode_horizon: int,
    base_seed: int,
    action_models: List[dict],
    joint_models: List[dict],
) -> List[dict]:
    results = []

    for alpha_s in alpha_s_list:
        for alpha_a in alpha_a_list:
            # Baseline
            print(f"  baseline  alpha_s={alpha_s:.3f} alpha_a={alpha_a:.3f} ...", end=" ", flush=True)
            t0 = time.time()
            successes = [
                _run_rollout_baseline(
                    policy, env, obs_keys,
                    alpha_s=alpha_s, alpha_a=alpha_a,
                    episode_horizon=episode_horizon,
                    seed=base_seed + i,
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
                    print(
                        f"  action_denoiser ({label})  alpha_s={alpha_s:.3f} "
                        f"alpha_a={alpha_a:.3f} t_start={t_start} ...",
                        end=" ", flush=True,
                    )
                    t0 = time.time()
                    successes = [
                        _run_rollout_action_denoiser(
                            policy, env, obs_keys,
                            am["model"], am["anchor"], am["alphas"], am["alphas_bar"],
                            am["norm"], am["horizon"], am["state_dim"], am["action_dim"],
                            alpha_s=alpha_s, alpha_a=alpha_a, t_start=t_start,
                            episode_horizon=episode_horizon,
                            seed=base_seed + i,
                            gripper_k=am["gripper_k"],
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

            # Joint denoiser variants
            for jm in joint_models:
                for t_start in t_start_list:
                    label = jm["label"]
                    print(
                        f"  joint_denoiser ({label})  alpha_s={alpha_s:.3f} "
                        f"alpha_a={alpha_a:.3f} t_start={t_start} ...",
                        end=" ", flush=True,
                    )
                    t0 = time.time()
                    successes = [
                        _run_rollout_joint_denoiser(
                            policy, env, obs_keys,
                            jm["model"], jm["anchor"], jm["alphas"], jm["alphas_bar"],
                            jm["norm"], jm["horizon"], jm["state_dim"], jm["action_dim"],
                            alpha_s=alpha_s, alpha_a=alpha_a, t_start=t_start,
                            episode_horizon=episode_horizon,
                            seed=base_seed + i,
                            gripper_k=jm["gripper_k"],
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
    p.add_argument("--bc_rnn_checkpoint", type=str, required=True,
                   help="BC-RNN checkpoint (.pth) used as base policy")
    p.add_argument("--action_denoiser", type=str, nargs="+", default=[],
                   help="Action denoiser checkpoint(s) (arch=action_unet)")
    p.add_argument("--joint_denoiser", type=str, nargs="*", default=[],
                   help="Joint denoiser checkpoint(s) for comparison")
    p.add_argument("--alpha_s", type=float, nargs="+", default=DEFAULT_ALPHA_S,
                   help="State noise std levels to sweep")
    p.add_argument("--alpha_a", type=float, nargs="+", default=DEFAULT_ALPHA_A,
                   help="Action noise std levels to sweep")
    p.add_argument("--t_start", type=int, nargs="+", default=DEFAULT_T_STARTS,
                   help="Denoiser reverse-diffusion start steps to sweep")
    p.add_argument("--n_rollouts", type=int, default=30,
                   help="Rollouts per configuration")
    p.add_argument("--episode_horizon", type=int, default=400,
                   help="Max env steps per episode")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--output_csv", type=str,
                   default="results/square/bc_rnn_denoiser/results.csv")
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


def main():
    import torch

    args = parse_args()
    configure_mujoco_env()
    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device: {device}")

    # Load BC RNN policy and environment
    print(f"\nLoading BC RNN policy: {args.bc_rnn_checkpoint}")
    policy, env, obs_keys, _ = _load_policy_and_env(args.bc_rnn_checkpoint, args.device)
    print(f"  obs_keys: {obs_keys}")

    # Load action denoiser(s)
    action_models = []
    for i, ckpt_path in enumerate(args.action_denoiser):
        print(f"\nLoading action denoiser: {ckpt_path}")
        model, anchor, alphas, alphas_bar, norm, hor, sdim, adim, ckpt_obs_keys, anchor_id, gk = \
            _load_action_model(ckpt_path, device)
        label = f"{anchor_id}_{i}" if len(args.action_denoiser) > 1 else anchor_id
        action_models.append({
            "model": model, "anchor": anchor,
            "alphas": alphas, "alphas_bar": alphas_bar,
            "norm": norm, "horizon": hor,
            "state_dim": sdim, "action_dim": adim,
            "obs_keys": ckpt_obs_keys, "anchor_id": anchor_id,
            "label": label, "gripper_k": gk,
        })
        print(f"  anchor={anchor_id}  state_dim={sdim}  action_dim={adim}  H={hor}  gripper_k={gk}")

    # Load joint denoiser(s)
    joint_models = []
    for i, ckpt_path in enumerate(args.joint_denoiser or []):
        print(f"\nLoading joint denoiser: {ckpt_path}")
        model, anchor, alphas, alphas_bar, norm, hor, sdim, adim, ckpt_obs_keys, anchor_id, gk = \
            _load_joint_model(ckpt_path, device)
        label = f"{anchor_id}_{i}" if len(args.joint_denoiser) > 1 else anchor_id
        joint_models.append({
            "model": model, "anchor": anchor,
            "alphas": alphas, "alphas_bar": alphas_bar,
            "norm": norm, "horizon": hor,
            "state_dim": sdim, "action_dim": adim,
            "obs_keys": ckpt_obs_keys, "anchor_id": anchor_id,
            "label": label, "gripper_k": gk,
        })
        print(f"  anchor={anchor_id}  state_dim={sdim}  action_dim={adim}  H={hor}  gripper_k={gk}")

    print(f"\nEval grid: alpha_s={args.alpha_s}  alpha_a={args.alpha_a}  t_start={args.t_start}")
    print(f"           n_rollouts={args.n_rollouts}  episode_horizon={args.episode_horizon}\n")

    results = _eval_grid(
        policy=policy,
        env=env,
        obs_keys=obs_keys,
        alpha_s_list=args.alpha_s,
        alpha_a_list=args.alpha_a,
        t_start_list=args.t_start,
        n_rollouts=args.n_rollouts,
        episode_horizon=args.episode_horizon,
        base_seed=args.base_seed,
        action_models=action_models,
        joint_models=joint_models,
    )

    # Write CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    fieldnames = ["model_type", "anchor_id", "alpha_s", "alpha_a", "t_start", "n_rollouts", "success_rate"]
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Summary table
    print(f"\n{'='*72}")
    print(f"{'model_type':<35} {'alpha_s':>7} {'alpha_a':>7} {'t_start':>7} {'sr':>6}")
    print(f"{'='*72}")
    for r in results:
        print(
            f"{r['model_type']:<35} {r['alpha_s']:>7.3f} {r['alpha_a']:>7.3f} "
            f"{str(r['t_start']):>7} {r['success_rate']:>6.3f}"
        )
    print(f"{'='*72}")
    print(f"\nResults saved to: {args.output_csv}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
