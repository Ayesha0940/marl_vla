#!/usr/bin/env python3
"""Evaluate the custom Lift diffusion policy."""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys
from collections import deque
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

import numpy as np
import torch


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (PROJECT_ROOT, EVAL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.mujoco import configure_mujoco_env
from common.results import save_results_json
from diffusion.lift_policy import DEFAULT_OBS_KEYS, load_lift_checkpoint, sample_action_sequence


#!/usr/bin/env python3
"""
Evaluate the custom Lift diffusion policy.
Supports both epsilon prediction (v1/v2) and x0 prediction (v3).
Auto-detects prediction_type from checkpoint.
"""



# Import x0 sampler from the training script that defines the x0 sampler.
try:
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "training", "lift"))
    from train_unet_diffusion_lift import sample_action_sequence_x0
except ImportError:
    sample_action_sequence_x0 = None


def find_default_env_ckpt():
    """Try to auto-discover a reasonable bc_rnn_lift env checkpoint.

    Looks for any `models/*.pth` under `checkpoints/bc_rnn_lift/**` and
    returns the most recent match. Falls back to the original hardcoded
    path for compatibility.
    """
    search = os.path.join(PROJECT_ROOT, "checkpoints", "bc_rnn_lift", "**", "models", "*.pth")
    matches = sorted(glob.glob(search, recursive=True))
    if matches:
        return matches[-1]
    return os.path.join(
        PROJECT_ROOT,
        "checkpoints", "bc_rnn_lift", "bc_rnn_lift",
        "20260405174006", "models", "model_epoch_600.pth",
    )


DEFAULT_ENV_CKPT = find_default_env_ckpt()


def _load_checkpoint(checkpoint_path: str, device: torch.device):
    ckpt     = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backbone = ckpt.get("backbone", "mlp")
    if backbone == "unet":
        from diffusion.lift_policy_unet import load_unet_checkpoint
        return load_unet_checkpoint(checkpoint_path, device)
    elif backbone == "transformer":
        from diffusion.lift_policy_transformer import load_transformer_checkpoint
        return load_transformer_checkpoint(checkpoint_path, device)
    else:
        from diffusion.lift_policy import load_lift_checkpoint
        return load_lift_checkpoint(checkpoint_path, device)


def _load_env(env_ckpt: str):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=env_ckpt, device=device, verbose=False)
    return EnvUtils.create_env_from_metadata(
        env_meta=ckpt_dict["env_metadata"],
        render=False, render_offscreen=False, use_image_obs=False,
    )


def _flatten_obs(obs: dict, obs_keys: Iterable[str]) -> np.ndarray:
    return np.concatenate(
        [np.asarray(obs[k]).reshape(-1) for k in obs_keys]
    ).astype(np.float32)


def _run_rollout(
    model, checkpoint, alphas, alphas_bar,
    env, horizon: int, seed: int, t_start: Optional[int],
    prediction_type: str, exec_horizon: Optional[int] = None,
) -> bool:
    device          = next(model.parameters()).device
    obs_keys        = list(checkpoint.get("obs_keys") or DEFAULT_OBS_KEYS)
    obs_mean        = np.asarray(checkpoint["obs_mean"],    dtype=np.float32)
    obs_std         = np.asarray(checkpoint["obs_std"],     dtype=np.float32)
    action_mean     = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std      = np.asarray(checkpoint["action_std"],  dtype=np.float32)
    obs_horizon     = int(checkpoint["obs_horizon"])
    action_horizon  = int(checkpoint["action_horizon"])
    diffusion_steps = int(checkpoint["diffusion_steps"])

    obs = env.reset()
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_vec = _flatten_obs(obs, obs_keys)
    history = deque([obs_vec.copy()] * obs_horizon, maxlen=obs_horizon)

    step = 0
    while step < horizon:
        obs_hist = np.stack(history, axis=0)

        # Route to correct sampler based on prediction_type
        if prediction_type == "x0":
            if sample_action_sequence_x0 is None:
                raise RuntimeError(
                    "sample_action_sequence_x0 not available. "
                    "Make sure training/lift/train_unet_diffusion_lift.py is on the path."
                )
            action_chunk = sample_action_sequence_x0(
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
        else:
            action_chunk = sample_action_sequence(
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

        n_exec = exec_horizon if exec_horizon is not None else action_horizon
        for action in action_chunk[:n_exec]:
            if step >= horizon:
                break
            obs, _, done, _ = env.step(np.clip(action, -1.0, 1.0))
            step += 1
            obs_vec = _flatten_obs(obs, obs_keys)
            history.append(obs_vec.copy())
            if env.is_success()["task"]:
                return True
            if done:
                return False

    return False


def _eval_checkpoint(
    checkpoint_path: str, env_ckpt: str,
    n_rollouts: int, horizon: int, seed: int, t_start: Optional[int],
    exec_horizon: Optional[int] = None,
) -> dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, alphas, alphas_bar = _load_checkpoint(checkpoint_path, device)
    env = _load_env(env_ckpt)

    # Auto-detect prediction type
    prediction_type = checkpoint.get("prediction_type", "epsilon")
    print(f"  prediction_type: {prediction_type}")
    if exec_horizon is not None:
        print(f"  exec_horizon: {exec_horizon}")

    successes = []
    for idx in range(n_rollouts):
        success = _run_rollout(
            model, checkpoint, alphas, alphas_bar,
            env, horizon, seed + idx, t_start, prediction_type, exec_horizon,
        )
        successes.append(success)
        print(f"  Rollout {idx+1:3d}/{n_rollouts}: {'SUCCESS' if success else 'FAILURE'}")

    rate = float(np.mean(successes))
    print(f"Success Rate: {rate:.3f} ({sum(successes)}/{n_rollouts})")
    return {"success_rate": rate, "n_success": int(sum(successes)), "n_rollouts": n_rollouts}


def _find_checkpoints(path: str) -> List[str]:
    if os.path.isdir(path):
        matches = sorted(glob.glob(os.path.join(path, "*.pt")))
        return matches or sorted(glob.glob(os.path.join(path, "*.pth")))
    return [path]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--agent",  type=str)
    g.add_argument("--sweep",  type=str)
    p.add_argument("--env_ckpt",   type=str,  default=DEFAULT_ENV_CKPT)
    p.add_argument("--n_rollouts", type=int,  default=50)
    p.add_argument("--horizon",    type=int,  default=400)
    p.add_argument("--seed",       type=int,  default=0)
    p.add_argument("--t_start",      type=int,  default=None)
    p.add_argument("--exec_horizon", type=int,  default=None,
                   help="Actions executed per policy query (default: full action_horizon). "
                        "Chi et al. 2023 uses 8 with pred_horizon=16.")
    return p.parse_args()


def main() -> int:
    configure_mujoco_env(verbose=False)
    args = parse_args()

    checkpoints = (_find_checkpoints(args.sweep) if args.sweep
                   else [args.agent])

    rows = []
    for ckpt_path in checkpoints:
        print(f"\n--- {os.path.basename(ckpt_path)} ---")
        stats = _eval_checkpoint(
            ckpt_path, args.env_ckpt,
            args.n_rollouts, args.horizon, args.seed, args.t_start,
            args.exec_horizon,
        )
        rows.append({"checkpoint": ckpt_path, **stats})

    results_dir = os.path.join(PROJECT_ROOT, "results", "lift", "custom_diffusion_policy")
    os.makedirs(results_dir, exist_ok=True)
    ts       = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(results_dir, f"eval_{ts}.json")
    save_results_json(rows, vars(args), out_path)
    print(f"\nResults saved to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
