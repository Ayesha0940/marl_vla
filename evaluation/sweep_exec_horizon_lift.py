#!/usr/bin/env python3
"""Sweep exec_horizon values for the Lift diffusion policy and save results for analysis."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import deque
from datetime import datetime
from typing import Iterable, List, Optional

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_DIR = os.path.dirname(os.path.abspath(__file__))
for path in (PROJECT_ROOT, EVAL_DIR):
    if path not in sys.path:
        sys.path.insert(0, path)

from common.mujoco import configure_mujoco_env
from common.results import save_results_json
from diffusion.lift_policy import DEFAULT_OBS_KEYS, load_lift_checkpoint, sample_action_sequence


DEFAULT_AGENT = os.path.join(
    PROJECT_ROOT, "checkpoints", "lift_diffusion_policy", "model_epoch_4000.pt"
)
DEFAULT_ENV_CKPT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "bc_rnn_lift",
    "bc_rnn_lift",
    "20260405174006",
    "models",
    "model_epoch_600.pth",
)
DEFAULT_EXEC_HORIZONS = [1, 2, 4, 8]


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


def _flatten_obs(obs: dict, obs_keys: Iterable[str]) -> np.ndarray:
    return np.concatenate([np.asarray(obs[key]).reshape(-1) for key in obs_keys]).astype(np.float32)


def _prepare_history(obs_vec: np.ndarray, obs_horizon: int, history: Optional[deque]) -> deque:
    if history is None:
        history = deque([obs_vec.copy() for _ in range(obs_horizon)], maxlen=obs_horizon)
    else:
        history.append(obs_vec.copy())
    return history


def _run_rollout(model, checkpoint, alphas, alphas_bar, env, horizon: int, seed: int,
                 t_start: Optional[int], exec_horizon: int) -> bool:
    import torch

    device = next(model.parameters()).device
    obs_keys = list(checkpoint.get("obs_keys") or DEFAULT_OBS_KEYS)
    obs_mean = np.asarray(checkpoint["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(checkpoint["obs_std"], dtype=np.float32)
    action_mean = np.asarray(checkpoint["action_mean"], dtype=np.float32)
    action_std = np.asarray(checkpoint["action_std"], dtype=np.float32)
    obs_horizon = int(checkpoint["obs_horizon"])
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

        for action in action_chunk[:exec_horizon]:
            if step >= horizon:
                break
            obs, _, done, _ = env.step(np.clip(action, -1.0, 1.0))
            step += 1
            if env.is_success()["task"]:
                return True
            if done:
                return False

    return False


def _eval_exec_horizon(model, checkpoint, alphas, alphas_bar, env,
                       exec_horizon: int, n_rollouts: int, horizon: int,
                       seed: int, t_start: Optional[int]) -> dict:
    successes = []
    for idx in range(n_rollouts):
        success = _run_rollout(
            model, checkpoint, alphas, alphas_bar, env,
            horizon, seed + idx, t_start, exec_horizon,
        )
        successes.append(success)
        status = "SUCCESS" if success else "FAILURE"
        print(f"    Rollout {idx + 1:3d}/{n_rollouts}: {status}")

    success_rate = float(np.mean(successes))
    print(f"  exec_horizon={exec_horizon}: {success_rate:.3f} ({sum(successes)}/{n_rollouts})")
    return {
        "exec_horizon": exec_horizon,
        "success_rate": success_rate,
        "n_success": int(sum(successes)),
        "n_rollouts": n_rollouts,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep exec_horizon for Lift diffusion policy")
    parser.add_argument("--agent", type=str, default=DEFAULT_AGENT)
    parser.add_argument("--env_ckpt", type=str, default=DEFAULT_ENV_CKPT)
    parser.add_argument("--exec_horizons", type=int, nargs="+", default=DEFAULT_EXEC_HORIZONS,
                        help="exec_horizon values to sweep (default: 1 2 4 8)")
    parser.add_argument("--n_rollouts", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--t_start", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    configure_mujoco_env(verbose=False)
    args = parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agent_path = args.agent if os.path.isabs(args.agent) else os.path.join(PROJECT_ROOT, args.agent)
    env_ckpt = args.env_ckpt if os.path.isabs(args.env_ckpt) else os.path.join(PROJECT_ROOT, args.env_ckpt)

    if not os.path.isfile(agent_path):
        print(f"Checkpoint not found: {agent_path}")
        return 1
    if not os.path.isfile(env_ckpt):
        print(f"Env checkpoint not found: {env_ckpt}")
        return 1

    print(f"Checkpoint: {agent_path}")
    print(f"Env:        {env_ckpt}")
    print(f"Sweeping exec_horizons: {args.exec_horizons}")
    print(f"Rollouts per value:     {args.n_rollouts}")

    model, checkpoint, alphas, alphas_bar = load_lift_checkpoint(agent_path, device)
    env = _load_env(env_ckpt)

    results = []
    for exec_horizon in args.exec_horizons:
        print(f"\n--- exec_horizon={exec_horizon} ---")
        row = _eval_exec_horizon(
            model, checkpoint, alphas, alphas_bar, env,
            exec_horizon, args.n_rollouts, args.horizon, args.seed, args.t_start,
        )
        results.append(row)

    print("\n=== Summary ===")
    print(f"{'exec_horizon':>14} {'success_rate':>13} {'n_success':>10}")
    for row in results:
        print(f"{row['exec_horizon']:>14} {row['success_rate']:>13.3f} {row['n_success']:>10}/{args.n_rollouts}")

    results_dir = os.path.join(PROJECT_ROOT, "results", "lift", "custom_diffusion_policy")
    os.makedirs(results_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = os.path.join(results_dir, f"exec_horizon_sweep_{timestamp}.json")
    save_results_json(
        results,
        {
            "checkpoint": agent_path,
            "env_ckpt": env_ckpt,
            "exec_horizons": args.exec_horizons,
            "n_rollouts": args.n_rollouts,
            "horizon": args.horizon,
            "seed": args.seed,
            "t_start": args.t_start,
        },
        json_path,
    )
    print(f"\nResults saved to: {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
