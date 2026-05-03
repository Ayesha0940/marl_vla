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


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (PROJECT_ROOT, EVAL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.mujoco import configure_mujoco_env
from common.results import save_results_json
from diffusion.lift_policy import DEFAULT_OBS_KEYS, load_lift_checkpoint, sample_action_sequence


DEFAULT_ENV_CKPT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "bc_rnn_lift",
    "bc_rnn_lift",
    "20260405174006",
    "models",
    "model_epoch_600.pth",
)


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


def _flatten_obs(obs: dict, obs_keys: Iterable[str]) -> np.ndarray:
    return np.concatenate([np.asarray(obs[key]).reshape(-1) for key in obs_keys]).astype(np.float32)


def _prepare_history(obs_vec: np.ndarray, obs_horizon: int, history: Optional[deque]) -> deque:
    if history is None:
        history = deque([obs_vec.copy() for _ in range(obs_horizon)], maxlen=obs_horizon)
    else:
        history.append(obs_vec.copy())
    return history


def _find_checkpoints(path: str) -> List[str]:
    if os.path.isdir(path):
        matches = sorted(glob.glob(os.path.join(path, "*.pt")))
        if matches:
            return matches
        matches = sorted(glob.glob(os.path.join(path, "*.pth")))
        return matches
    return [path]


def _run_rollout(model, checkpoint, alphas, alphas_bar, env, horizon: int, seed: int, t_start: Optional[int]) -> bool:
    import torch

    device = next(model.parameters()).device
    obs_keys = list(checkpoint.get("obs_keys") or DEFAULT_OBS_KEYS)
    obs_mean = np.asarray(checkpoint["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(checkpoint["obs_std"], dtype=np.float32)
    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(checkpoint["action_std"], dtype=np.float32)
    obs_horizon = int(checkpoint["obs_horizon"])
    action_horizon = int(checkpoint["action_horizon"])
    diffusion_steps = int(checkpoint["diffusion_steps"])

    obs = env.reset()
    history = None
    np.random.seed(seed)
    torch.manual_seed(seed)

    step = 0
    while step < horizon:
        obs_vec = _flatten_obs(obs, obs_keys)
        history = _prepare_history(obs_vec, obs_horizon, history)
        obs_hist = np.stack(history, axis=0)

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

        for action in action_chunk[:action_horizon]:
            if step >= horizon:
                break
            obs, _, done, _ = env.step(np.clip(action, -1.0, 1.0))
            step += 1
            if env.is_success()["task"]:
                return True
            if done:
                return False

    return False


def _eval_checkpoint(checkpoint_path: str, env_ckpt: str, n_rollouts: int, horizon: int, seed: int, t_start: Optional[int]) -> dict:
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint, alphas, alphas_bar = load_lift_checkpoint(checkpoint_path, device)
    env = _load_env(env_ckpt)

    successes = []
    for index in range(n_rollouts):
        success = _run_rollout(model, checkpoint, alphas, alphas_bar, env, horizon, seed + index, t_start)
        successes.append(success)
        print(f"  Rollout {index + 1:3d}/{n_rollouts}: {'SUCCESS' if success else 'FAILURE'}")

    success_rate = float(np.mean(successes))
    print(f"Success Rate: {success_rate:.3f} ({sum(successes)}/{n_rollouts})")
    return {"success_rate": success_rate, "n_success": int(sum(successes)), "n_rollouts": n_rollouts}


def _run_single(args: argparse.Namespace) -> int:
    checkpoint_path = args.agent if os.path.isabs(args.agent) else os.path.join(PROJECT_ROOT, args.agent)
    if not os.path.isfile(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        return 1

    env_ckpt = args.env_ckpt if os.path.isabs(args.env_ckpt) else os.path.join(PROJECT_ROOT, args.env_ckpt)
    if not os.path.isfile(env_ckpt):
        print(f"Env checkpoint not found: {env_ckpt}")
        return 1

    print(f"Evaluating: {checkpoint_path}")
    print(f"Env checkpoint: {env_ckpt}")
    stats = _eval_checkpoint(checkpoint_path, env_ckpt, args.n_rollouts, args.horizon, args.seed, args.t_start)

    results_dir = os.path.join(PROJECT_ROOT, "results", "lift", "custom_diffusion_policy")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f"eval_{timestamp}.json")
    save_results_json(
        [{"checkpoint": checkpoint_path, **stats}],
        {
            "checkpoint": checkpoint_path,
            "env_ckpt": env_ckpt,
            "n_rollouts": args.n_rollouts,
            "horizon": args.horizon,
            "seed": args.seed,
            "t_start": args.t_start,
        },
        json_path,
    )
    return 0


def _run_sweep(args: argparse.Namespace) -> int:
    checkpoint_dir = args.sweep if os.path.isabs(args.sweep) else os.path.join(PROJECT_ROOT, args.sweep)
    if not os.path.isdir(checkpoint_dir):
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return 1

    env_ckpt = args.env_ckpt if os.path.isabs(args.env_ckpt) else os.path.join(PROJECT_ROOT, args.env_ckpt)
    if not os.path.isfile(env_ckpt):
        print(f"Env checkpoint not found: {env_ckpt}")
        return 1

    checkpoints = _find_checkpoints(checkpoint_dir)
    if not checkpoints:
        print(f"No checkpoints found in {checkpoint_dir}")
        return 1

    rows = []
    for checkpoint_path in checkpoints:
        print(f"\n--- {os.path.basename(checkpoint_path)} ---")
        stats = _eval_checkpoint(checkpoint_path, env_ckpt, args.n_rollouts, args.horizon, args.seed, args.t_start)
        rows.append({"checkpoint": checkpoint_path, **stats})

    results_dir = os.path.join(PROJECT_ROOT, "results", "lift", "custom_diffusion_policy")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f"sweep_{timestamp}.json")
    save_results_json(
        rows,
        {
            "checkpoint_dir": checkpoint_dir,
            "env_ckpt": env_ckpt,
            "n_rollouts": args.n_rollouts,
            "horizon": args.horizon,
            "seed": args.seed,
            "t_start": args.t_start,
        },
        json_path,
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the custom Lift diffusion policy")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--agent", type=str, help="Path to a checkpoint .pt file")
    group.add_argument("--sweep", type=str, help="Path to a directory of checkpoints")
    parser.add_argument("--env_ckpt", type=str, default=DEFAULT_ENV_CKPT, help="Robomimic Lift checkpoint used to build the env")
    parser.add_argument("--n_rollouts", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t_start", type=int, default=None, help="Reverse diffusion start step; defaults to the full schedule")
    return parser.parse_args()


def main() -> int:
    configure_mujoco_env(verbose=False)
    args = parse_args()
    if args.agent:
        return _run_single(args)
    return _run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
