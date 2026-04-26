#!/usr/bin/env python3
"""
Evaluation harness for the joint (state, action) denoiser on Robomimic Lift.

Sweeps a 2-D noise grid (alpha_s × alpha_a) and reports per-cell success rates,
with three baselines and the joint denoiser conditioned on a selected anchor.

Baselines included automatically:
    BASE-clean : no noise, no denoiser  (ceiling)
    BASE-noisy : noise injected, no denoiser  (floor)
    JOINT-Ax   : joint denoiser + chosen anchor

Usage:
    python evaluation/eval_joint_denoiser.py \
        --joint_ckpt diffusion_models/joint_a1_lift.pt \
        --anchor A1 \
        --n_rollouts 50 \
        --t_start 20

Checkpoint format (save this from your training script):
    torch.save({
        'model_state_dict':  model.state_dict(),
        'anchor_state_dict': anchor.state_dict(),
        'state_dim':         int,
        'action_dim':        int,
        'horizon':           int,   # H
        'diffusion_steps':   int,   # T
        'state_mean':        np.ndarray (D_s,),
        'state_std':         np.ndarray (D_s,),
        'action_mean':       np.ndarray (D_a,),
        'action_std':        np.ndarray (D_a,),
        'anchor_id':         str,   # 'A1' ... 'A8'
        'object_dim':        int,   # required for A1/A6/A7/A8
        'proprio_dim':       int,   # required for A3/A7/A8
        'gripper_k':         int,   # required for A2/A6/A8
        # optional arch overrides:
        'channel_sizes':     tuple, # default (64, 128, 256)
        'anchor_dim':        int,   # default 128
        'time_emb_dim':      int,   # default 128
    }, path)
"""

import argparse
import os
import sys
import time
from collections import deque
from datetime import datetime
from typing import Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR     = os.path.dirname(os.path.abspath(__file__))
for _p in (PROJECT_ROOT, EVAL_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ── Default paths ──────────────────────────────────────────────────────────────

DEFAULT_BC_CKPT = os.path.join(
    PROJECT_ROOT,
    "checkpoints", "bc_rnn_lift", "bc_rnn_lift",
    "20260405174006", "models", "model_epoch_600.pth",
)

DEFAULT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "results", "lift", "joint_denoiser")

# Noise grid from the spec
DEFAULT_ALPHA_A = [0.05, 0.1, 0.2]
DEFAULT_ALPHA_S = [0.01, 0.02, 0.03, 0.04, 0.05]


# ── Environment / policy loading ───────────────────────────────────────────────

def _load_policy_and_env(checkpoint_path: str, device_str: str = "auto"):
    import torch
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils

    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    try:
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=checkpoint_path, device=device, verbose=False
        )
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower() and device.type == "cuda":
            import torch
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            policy, ckpt_dict = FileUtils.policy_from_checkpoint(
                ckpt_path=checkpoint_path, device=device, verbose=False
            )
        else:
            raise

    env_meta = ckpt_dict["env_metadata"]
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta, render=False, render_offscreen=False, use_image_obs=False
    )
    return policy, env, ckpt_dict


# ── Joint denoiser loading ─────────────────────────────────────────────────────

def _load_joint_model(ckpt_path: str, device):
    import torch
    from diffusion.joint_unet import JointUNet1D
    from diffusion.anchors import build_anchor
    from diffusion.model import make_beta_schedule

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    state_dim  = ckpt["state_dim"]
    action_dim = ckpt["action_dim"]
    horizon    = ckpt["horizon"]
    T          = ckpt["diffusion_steps"]
    anchor_id  = ckpt["anchor_id"]

    model = JointUNet1D(
        state_dim    = state_dim,
        action_dim   = action_dim,
        anchor_dim   = ckpt.get("anchor_dim", 128),
        time_emb_dim = ckpt.get("time_emb_dim", 128),
        channel_sizes= ckpt.get("channel_sizes", (64, 128, 256)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(device)

    anchor = build_anchor(
        anchor_id,
        object_dim  = ckpt.get("object_dim",  14),
        proprio_dim = ckpt.get("proprio_dim", 14),
        gripper_k   = ckpt.get("gripper_k",   5),
    )
    anchor.load_state_dict(ckpt["anchor_state_dict"])
    anchor.eval().to(device)

    _, alphas, alphas_bar = make_beta_schedule(T)
    alphas     = alphas.to(device)
    alphas_bar = alphas_bar.to(device)

    norm = {
        "state_mean":  torch.from_numpy(np.asarray(ckpt["state_mean"],  dtype=np.float32)).to(device),
        "state_std":   torch.from_numpy(np.asarray(ckpt["state_std"],   dtype=np.float32)).to(device),
        "action_mean": torch.from_numpy(np.asarray(ckpt["action_mean"], dtype=np.float32)).to(device),
        "action_std":  torch.from_numpy(np.asarray(ckpt["action_std"],  dtype=np.float32)).to(device),
    }

    gripper_k = ckpt.get("gripper_k", 5)
    print(
        f"[JointDenoiser] loaded | anchor={anchor_id} | "
        f"state_dim={state_dim} | action_dim={action_dim} | H={horizon} | T={T}"
    )
    return model, anchor, alphas, alphas_bar, norm, horizon, state_dim, action_dim, gripper_k


# ── Observation utilities ──────────────────────────────────────────────────────

def _flatten_obs(obs_dict: dict, obs_keys: list) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs_dict[k]).flatten() for k in obs_keys]
    ).astype(np.float32)


def _corrupt_obs(obs_dict: dict, obs_keys: list, alpha_s: float, rng: np.random.Generator) -> dict:
    """Add Gaussian noise to each obs key independently."""
    if alpha_s == 0.0:
        return obs_dict
    noisy = dict(obs_dict)
    for key in obs_keys:
        arr = np.asarray(obs_dict[key], dtype=np.float32)
        noisy[key] = arr + rng.normal(0.0, alpha_s, size=arr.shape).astype(np.float32)
    return noisy


# ── Single rollout ─────────────────────────────────────────────────────────────

_PROPRIO_KEYS = ["robot0_joint_pos", "robot0_eef_pos", "robot0_eef_quat"]
_GRIPPER_ACTION_DIM = 6   # index of gripper command in the 7-dim Lift action
_LIFT_Z_THRESH = 0.85


def _run_rollout_joint(
    policy,
    env,
    obs_keys: list,
    model,
    anchor,
    alphas,
    alphas_bar,
    norm: dict,
    horizon_H: int,          # U-Net temporal horizon
    state_dim: int,
    action_dim: int,
    alpha_s: float,          # state noise std
    alpha_a: float,          # action noise std
    t_start: int,
    episode_horizon: int,    # max env steps
    seed: int,
    gripper_k: int = 5,
) -> tuple:
    """
    Run one episode with joint denoising.

    Returns:
        (episode_reward, success, mean_latency_ms)
    """
    import torch
    from diffusion.joint_unet import joint_denoise

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    obs     = env.reset()
    policy.start_episode()

    # A1 anchor: capture clean object pose at episode start
    object_pose_t0 = np.asarray(obs["object"], dtype=np.float32).copy()

    # Sliding window buffers (H most recent steps)
    state_buf       = deque(maxlen=horizon_H)
    action_buf      = deque(maxlen=horizon_H)
    gripper_hist_buf = deque(maxlen=gripper_k)   # A2 anchor

    ep_reward = 0.0
    success   = False
    latencies = []

    for _ in range(episode_horizon):
        # 1. Corrupt observation (policy sees noisy obs)
        noisy_obs = _corrupt_obs(obs, obs_keys, alpha_s, rng)

        # 2. Policy action from corrupted obs
        with torch.no_grad():
            ac = policy(noisy_obs)
            if isinstance(ac, torch.Tensor):
                ac = ac.cpu().numpy()
            ac = np.asarray(ac).flatten().astype(np.float32)

        # 3. Add action noise
        if alpha_a > 0.0:
            ac = ac + rng.normal(0.0, alpha_a, size=ac.shape).astype(np.float32)

        # Update gripper history with pre-noise command (for A2 anchor)
        gripper_hist_buf.append(float(ac[_GRIPPER_ACTION_DIM]))

        # 4. Accumulate into sliding window
        state_vec = _flatten_obs(noisy_obs, obs_keys)  # (D_s,)
        state_buf.append(state_vec)
        action_buf.append(ac)

        # 5. Pad window to H if not enough history yet
        pad = horizon_H - len(state_buf)
        states_win  = np.stack(
            [state_buf[0]] * pad + list(state_buf), axis=0
        )   # (H, D_s)
        actions_win = np.stack(
            [action_buf[0]] * pad + list(action_buf), axis=0
        )   # (H, D_a)

        # 6. Normalize
        s_norm = (states_win  - norm["state_mean"].cpu().numpy())  / norm["state_std"].cpu().numpy()
        a_norm = (actions_win - norm["action_mean"].cpu().numpy()) / norm["action_std"].cpu().numpy()

        s_t = torch.from_numpy(s_norm).float().unsqueeze(0).to(alphas.device)   # (1, H, D_s)
        a_t = torch.from_numpy(a_norm).float().unsqueeze(0).to(alphas.device)   # (1, H, D_a)

        # 7. Build anchor dict (all keys populated so any A0–A8 anchor works)
        # A2: gripper history — K commands before this step, zero-padded at start
        hist = list(gripper_hist_buf)
        gh_arr = np.array(
            [0.0] * (gripper_k - len(hist)) + hist, dtype=np.float32
        )
        # A3: clean proprio from unnoised obs (robot0_joint_pos + eef_pos + eef_quat)
        proprio_parts = [
            np.asarray(obs[k], dtype=np.float32).flatten()
            for k in _PROPRIO_KEYS if k in obs
        ]
        proprio_arr = (
            np.concatenate(proprio_parts) if proprio_parts
            else np.zeros(14, dtype=np.float32)
        )
        # A4: rule-based phase from clean object z + gripper command
        _closed = float(ac[_GRIPPER_ACTION_DIM]) < 0.5
        _obj_z  = float(obs["object"][2]) if "object" in obs else 0.0
        _phase  = 2 if (_closed and _obj_z > _LIFT_Z_THRESH) else (1 if _closed else 0)

        dev = alphas.device
        traj = {
            "object_pose_t0":  torch.from_numpy(object_pose_t0).float().unsqueeze(0).to(dev),
            "gripper_history": torch.from_numpy(gh_arr).float().unsqueeze(0).to(dev),
            "proprio":         torch.from_numpy(proprio_arr).float().unsqueeze(0).to(dev),
            "phase":           torch.tensor([_phase], dtype=torch.long).to(dev),
        }
        with torch.no_grad():
            anchor_emb = anchor.compute(traj)   # (1, D_c)

        # 8. Joint denoising
        t0 = time.perf_counter()
        with torch.no_grad():
            _, clean_a = joint_denoise(
                model, s_t, a_t, anchor_emb, alphas, alphas_bar, t_start=t_start
            )
        latencies.append((time.perf_counter() - t0) * 1000.0)

        # 9. Denormalize most-recent action (index H-1 = last in window)
        clean_a_np = clean_a[0, -1].cpu().numpy()   # (D_a,)
        clean_a_np = (
            clean_a_np * norm["action_std"].cpu().numpy()
            + norm["action_mean"].cpu().numpy()
        )
        clean_a_np = np.clip(clean_a_np, -1.0, 1.0)

        # 10. Step environment
        obs, reward, done, _ = env.step(clean_a_np)
        ep_reward += reward

        if env.is_success()["task"]:
            success = True
            break
        if done:
            break

    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    return ep_reward, success, mean_lat


def _run_rollout_noisy(
    policy,
    env,
    obs_keys: list,
    alpha_s: float,
    alpha_a: float,
    episode_horizon: int,
    seed: int,
) -> tuple:
    """Baseline: noisy policy, no denoiser."""
    import torch

    rng = np.random.default_rng(seed)

    obs = env.reset()
    policy.start_episode()
    ep_reward = 0.0
    success   = False

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
        obs, reward, done, _ = env.step(ac)
        ep_reward += reward
        if env.is_success()["task"]:
            success = True
            break
        if done:
            break

    return ep_reward, success


def _run_rollout_clean(
    policy,
    env,
    episode_horizon: int,
    seed: int,
) -> tuple:
    """Baseline: clean policy, no noise, no denoiser."""
    import torch

    np.random.seed(seed)

    obs = env.reset()
    policy.start_episode()
    ep_reward = 0.0
    success   = False

    for _ in range(episode_horizon):
        with torch.no_grad():
            ac = policy(obs)
            if isinstance(ac, torch.Tensor):
                ac = ac.cpu().numpy()
            ac = np.asarray(ac).flatten().astype(np.float32)
        ac = np.clip(ac, -1.0, 1.0)
        obs, reward, done, _ = env.step(ac)
        ep_reward += reward
        if env.is_success()["task"]:
            success = True
            break
        if done:
            break

    return ep_reward, success


# ── Noise grid evaluation ──────────────────────────────────────────────────────

def _eval_grid(
    policy,
    env,
    obs_keys: list,
    joint_ckpts: list,        # list of checkpoint paths, one per anchor
    t_starts: list,           # list of int
    alpha_s_list: list,
    alpha_a_list: list,
    n_rollouts: int,
    episode_horizon: int,
    base_seed: int,
    device,
) -> dict:
    """
    Sweep (alpha_s × alpha_a) grid across all (checkpoint × t_start) combos.

    Returns a dict:
        results[(alpha_s, alpha_a)][col_name] = success_rate
    where col_name is e.g. "A0(t=10)".
    """
    # Load all checkpoints upfront
    models = []
    for ckpt_path in joint_ckpts:
        m, anc, alps, alps_bar, norm, H, Ds, Da, gk = _load_joint_model(ckpt_path, device)
        import torch
        anchor_id = str(torch.load(ckpt_path, map_location="cpu", weights_only=False).get("anchor_id", "?"))
        models.append((m, anc, alps, alps_bar, norm, H, Ds, Da, gk, anchor_id))

    col_names = [f"{anchor_id}(t={t})" for (_, _, _, _, _, _, _, _, _, anchor_id) in models
                 for t in t_starts]
    results = {}

    total_cells = len(alpha_s_list) * len(alpha_a_list)
    cell_idx = 0

    for alpha_s in alpha_s_list:
        for alpha_a in alpha_a_list:
            cell_idx += 1
            key = (f"{alpha_s:.3f}", f"{alpha_a:.3f}")
            results[key] = {}
            print(f"\n[Cell {cell_idx}/{total_cells}] alpha_s={alpha_s:.3f}  alpha_a={alpha_a:.3f}")

            for (model, anchor, alphas, alphas_bar, norm,
                 horizon_H, state_dim, action_dim, gripper_k, anchor_id) in models:
                for t_start in t_starts:
                    col = f"{anchor_id}(t={t_start})"
                    joint_successes, joint_lats = [], []
                    for i in range(n_rollouts):
                        seed = base_seed + int(alpha_s * 1000) + int(alpha_a * 1000) * 1000 + i
                        env.reset()
                        _, s, lat = _run_rollout_joint(
                            policy, env, obs_keys,
                            model, anchor, alphas, alphas_bar, norm,
                            horizon_H, state_dim, action_dim,
                            alpha_s, alpha_a, t_start, episode_horizon, seed,
                            gripper_k=gripper_k,
                        )
                        joint_successes.append(s)
                        joint_lats.append(lat)
                    sr  = float(np.mean(joint_successes))
                    lat = float(np.mean(joint_lats))
                    results[key][col] = sr
                    print(f"  {col:15s}  success_rate={sr:.4f}  latency={lat:.1f}ms")

    return results, col_names


# ── Results I/O ────────────────────────────────────────────────────────────────

def _save_results(results: dict, col_names: list, output_csv: str):
    """Write pivot CSV: rows = (alpha_s, alpha_a), cols = anchor(t=X) combos."""
    import csv
    os.makedirs(os.path.dirname(os.path.abspath(output_csv)), exist_ok=True)
    header = ["alpha_s", "alpha_a"] + col_names
    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for (a_s, a_a), vals in sorted(results.items(), key=lambda x: (float(x[0][0]), float(x[0][1]))):
            row = {"alpha_s": a_s, "alpha_a": a_a}
            for col in col_names:
                v = vals.get(col)
                row[col] = f"{v:.4f}" if v is not None else ""
            writer.writerow(row)
    print(f"\nSaved: {output_csv}  ({len(results)} rows × {len(col_names)} anchor columns)")


def _print_summary(results: dict, col_names: list):
    print(f"\n{'alpha_s':>8}  {'alpha_a':>8}" + "".join(f"  {c:>14}" for c in col_names))
    print("─" * (20 + 16 * len(col_names)))
    for (a_s, a_a), vals in sorted(results.items(), key=lambda x: (float(x[0][0]), float(x[0][1]))):
        row = f"{float(a_s):>8.3f}  {float(a_a):>8.3f}"
        for col in col_names:
            v = vals.get(col)
            row += f"  {v:>14.4f}" if v is not None else f"  {'—':>14}"
        print(row)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bc_rnn_ckpt",  default=DEFAULT_BC_CKPT)
    p.add_argument("--joint_ckpts",  nargs="+", required=True,
                   help="One checkpoint per anchor, e.g.: ckpt_a0.pt ckpt_a2.pt ckpt_a7.pt")
    p.add_argument("--t_starts",     nargs="+", type=int, default=[10],
                   help="Reverse diffusion start steps. Default: 10")
    p.add_argument("--alpha_s",      default=",".join(str(v) for v in DEFAULT_ALPHA_S))
    p.add_argument("--alpha_a",      default=",".join(str(v) for v in DEFAULT_ALPHA_A))
    p.add_argument("--n_rollouts",   type=int, default=25)
    p.add_argument("--episode_horizon", type=int, default=400)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--output_csv",   required=True,
                   help="Output CSV path, e.g. results/lift/joint_denoiser/baseline_results.csv")
    p.add_argument("--device",       default="auto")
    return p.parse_args()


def _configure_mujoco():
    paths = [
        os.path.expanduser("~/.mujoco/mujoco210/bin"),
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu/nvidia",
    ]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    for p in paths:
        if os.path.exists(p) and p not in current.split(":"):
            current = f"{p}:{current}" if current else p
    os.environ["LD_LIBRARY_PATH"] = current


def main():
    args = parse_args()
    _configure_mujoco()

    alpha_s_list = [float(x) for x in args.alpha_s.split(",")]
    alpha_a_list = [float(x) for x in args.alpha_a.split(",")]

    import torch
    device_str = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if device_str == "auto" else torch.device(device_str)

    print(f"Device:       {device}")
    print(f"Checkpoints:  {args.joint_ckpts}")
    print(f"t_starts:     {args.t_starts}")
    print(f"α_s grid:     {alpha_s_list}")
    print(f"α_a grid:     {alpha_a_list}")
    print(f"Rollouts/cell:{args.n_rollouts}")

    policy, env, ckpt_dict = _load_policy_and_env(args.bc_rnn_ckpt, device_str)
    obs_keys = list(ckpt_dict["shape_metadata"]["all_obs_keys"])

    results, col_names = _eval_grid(
        policy         = policy,
        env            = env,
        obs_keys       = obs_keys,
        joint_ckpts    = args.joint_ckpts,
        t_starts       = args.t_starts,
        alpha_s_list   = alpha_s_list,
        alpha_a_list   = alpha_a_list,
        n_rollouts     = args.n_rollouts,
        episode_horizon= args.episode_horizon,
        base_seed      = args.seed,
        device         = device,
    )

    _print_summary(results, col_names)
    _save_results(results, col_names, args.output_csv)


if __name__ == "__main__":
    main()
