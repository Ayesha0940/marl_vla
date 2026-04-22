#!/usr/bin/env python3
"""
Unified evaluation launcher.

This script is the single entrypoint for evaluation workflows.

It now includes internal implementations for the most-used suites, and
falls back to legacy script dispatch for suites not migrated yet.

Examples:
  python evaluation/eval.py --task can --suite standard
  python evaluation/eval.py --task lift --suite standard --epoch 600 --n_rollouts 100
  python evaluation/eval.py --task square --suite robustness-diffusion \
      --diffusion_model diffusion_models/square_diffusion_model.pt --t_start 10 20 40
  python evaluation/eval.py --task transport --suite sweep --epochs 200,400,600
"""

import argparse
import glob
import os
import re
import subprocess
import sys
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from common.launcher import (
    SCRIPT_MAP,
    all_tasks,
    print_supported_pairs,
    validate_task_suite,
)
from common.checkpoints import find_latest_checkpoint_for_epoch
from common.mujoco import configure_mujoco_env
from common.results import print_robustness_summary, save_results_csv, save_results_json

EVAL_DIR = os.path.join(PROJECT_ROOT, "evaluation")
TRANSPORT_CHECKPOINT_ROOT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "bc_rnn",
    "bc_rnn_transport_tuned",
    "bc_rnn_transport_tuned",
)

ROBUSTNESS_CONFIGS: Dict[str, Dict] = {
    "can": {
        "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints", "bc_rnn_can", "bc_rnn_can"),
        "results_dir": os.path.join(PROJECT_ROOT, "results", "can"),
        "noise_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.25, 1.5, 1.75, 2.0],
        "default_epoch": 600,
        "default_horizon": 400,
        "csv_prefix": "robustness_eval_can",
        "manual_methods": ["none"],
    },
    "lift": {
        "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints", "bc_rnn_lift", "bc_rnn_lift"),
        "results_dir": os.path.join(PROJECT_ROOT, "results", "lift"),
        "noise_levels": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
        "default_epoch": 600,
        "default_horizon": 400,
        "csv_prefix": "robustness_eval_lift",
        "manual_methods": ["none"],
    },
    "square": {
        "checkpoint_dir": os.path.join(PROJECT_ROOT, "checkpoints", "bc_rnn_square", "bc_rnn_square_v3"),
        "results_dir": os.path.join(PROJECT_ROOT, "results", "square"),
        "noise_levels": [0.0, 0.04, 0.08, 0.12, 0.16, 0.2],
        "default_epoch": 2000,
        "default_horizon": 500,
        "csv_prefix": "robustness_eval_square",
        "manual_methods": ["none", "kalman", "ema", "median"],
    },
}

VIDEO_NOISE_LEVELS: Dict[str, List[float]] = {
    "can": [0.0],
    "lift": [0.1, 0.2, 0.5, 1.0],
    "square": [0.0],
}


class EMAFilter:
    def __init__(self, dim: int, alpha: float = 0.2):
        self.alpha = alpha
        self.x = np.zeros(dim)

    def update(self, a):
        self.x = self.alpha * a + (1 - self.alpha) * self.x
        return self.x


class MedianFilter:
    def __init__(self, dim: int, k: int = 5):
        self.buffer = deque(maxlen=k)

    def update(self, a):
        self.buffer.append(a)
        return np.median(self.buffer, axis=0)


class KalmanFilter:
    def __init__(self, dim: int, Q: float = 1e-3, R: float = 1e-2):
        self.dim = dim
        self.x = np.zeros(dim)
        self.P = np.eye(dim)
        self.Q = Q * np.eye(dim)
        self.R = R * np.eye(dim)

    def update(self, z):
        x_pred = self.x
        p_pred = self.P + self.Q
        k_gain = p_pred @ np.linalg.inv(p_pred + self.R)
        self.x = x_pred + k_gain @ (z - x_pred)
        self.P = (np.eye(self.dim) - k_gain) @ p_pred
        return self.x


def _create_filter(method: str, action_dim: int):
    if method == "kalman":
        return KalmanFilter(dim=action_dim)
    if method == "ema":
        return EMAFilter(dim=action_dim)
    if method == "median":
        return MedianFilter(dim=action_dim)
    return None


def _task_single_checkpoint_dir(task: str) -> str:
    """Return single-root checkpoint dirs for can/lift tasks.

    Square and transport use different checkpoint discovery logic and are
    handled by dedicated helpers.
    """
    if task == "can":
        return os.path.join(PROJECT_ROOT, "checkpoints", "bc_rnn_can", "bc_rnn_can")
    if task == "lift":
        return os.path.join(PROJECT_ROOT, "checkpoints", "bc_rnn_lift", "bc_rnn_lift")
    raise ValueError(f"Unsupported single-root task: {task}")


def _resolve_checkpoint(checkpoint_dir: str, epoch: int) -> Optional[str]:
    ckpt = find_latest_checkpoint_for_epoch(checkpoint_dir, epoch)
    if ckpt:
        return ckpt
    print(f"Checkpoint not found for epoch {epoch}: {checkpoint_dir}")
    return None


def _list_transport_run_dirs(root_dir: str) -> List[str]:
    if not os.path.isdir(root_dir):
        return []
    runs = []
    for name in os.listdir(root_dir):
        run_dir = os.path.join(root_dir, name)
        if os.path.isdir(os.path.join(run_dir, "models")):
            runs.append(run_dir)
    runs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return runs


def _parse_epoch_from_path(path: str) -> Optional[int]:
    m = re.search(r"model_epoch_(\d+)\.pth$", os.path.basename(path))
    return int(m.group(1)) if m else None


def _find_transport_checkpoint(run_dir: str, epoch: Optional[int] = None) -> Optional[str]:
    models_dir = os.path.join(run_dir, "models")
    if not os.path.isdir(models_dir):
        return None

    if epoch is not None:
        target = os.path.join(models_dir, f"model_epoch_{epoch}.pth")
        return target if os.path.exists(target) else None

    all_ckpts = glob.glob(os.path.join(models_dir, "model_epoch_*.pth"))
    all_ckpts = [p for p in all_ckpts if _parse_epoch_from_path(p) is not None]
    if not all_ckpts:
        return None
    all_ckpts.sort(key=lambda p: _parse_epoch_from_path(p), reverse=True)
    return all_ckpts[0]


def _resolve_default(value: Optional[int], default: int) -> int:
    return default if value is None else value


def _run_standard_eval(task: str, args: argparse.Namespace) -> int:
    configure_mujoco_env()

    epoch = _resolve_default(args.epoch, 600)
    n_rollouts = _resolve_default(args.n_rollouts, 50)
    horizon = _resolve_default(args.horizon, 400)
    seed = _resolve_default(args.seed, 0)

    ckpt = _resolve_checkpoint(_task_single_checkpoint_dir(task), epoch)
    if not ckpt:
        return 0 if args.dry_run else 1

    cmd = [
        sys.executable,
        "-m",
        "robomimic.scripts.run_trained_agent",
        "--agent",
        ckpt,
        "--n_rollouts",
        str(n_rollouts),
        "--horizon",
        str(horizon),
        "--seed",
        str(seed),
    ]

    print(f"Using checkpoint: {ckpt}")
    print(f"Command: {' '.join(cmd)}")
    if args.dry_run:
        return 0
    return subprocess.run(cmd, cwd=PROJECT_ROOT).returncode


def _run_transport_standard(args: argparse.Namespace) -> int:
    configure_mujoco_env()

    n_rollouts = _resolve_default(args.n_rollouts, 50)
    horizon = _resolve_default(args.horizon, 700)
    seed = _resolve_default(args.seed, 0)

    if args.run_dir:
        run_dirs = [args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)]
    else:
        root = args.checkpoint_root if args.checkpoint_root else TRANSPORT_CHECKPOINT_ROOT
        root = root if os.path.isabs(root) else os.path.join(PROJECT_ROOT, root)
        run_dirs = _list_transport_run_dirs(root)

    if not run_dirs:
        print("No transport runs found.")
        return 0 if args.dry_run else 1

    if not args.all_runs:
        run_dirs = [run_dirs[0]]

    failures = 0
    for run_dir in run_dirs:
        ckpt = _find_transport_checkpoint(run_dir, epoch=args.epoch)
        if not ckpt:
            print(f"Skipping run without matching checkpoint: {run_dir}")
            failures += 1
            continue

        cmd = [
            sys.executable,
            "-m",
            "robomimic.scripts.run_trained_agent",
            "--agent",
            ckpt,
            "--n_rollouts",
            str(n_rollouts),
            "--horizon",
            str(horizon),
            "--seed",
            str(seed),
        ]
        print(f"Run: {run_dir}")
        print(f"Checkpoint: {ckpt}")
        print(f"Command: {' '.join(cmd)}")
        if args.dry_run:
            continue
        rc = subprocess.run(cmd, cwd=PROJECT_ROOT).returncode
        if rc != 0:
            failures += 1

    return 0 if failures == 0 else 1


def _parse_epochs_csv(epochs_csv: str) -> List[int]:
    if not epochs_csv.strip():
        return [100, 200, 300, 400, 500, 650, 800, 900, 1000]
    return [int(x.strip()) for x in epochs_csv.split(",") if x.strip()]


def _extract_stats_from_output(text: str):
    s = re.search(r'"Success_Rate"\s*:\s*([0-9.]+)', text)
    r = re.search(r'"Return"\s*:\s*([0-9.]+)', text)
    h = re.search(r'"Horizon"\s*:\s*([0-9.]+)', text)
    return (
        float(s.group(1)) if s else None,
        float(r.group(1)) if r else None,
        float(h.group(1)) if h else None,
    )


def _run_transport_sweep(args: argparse.Namespace) -> int:
    configure_mujoco_env()

    n_rollouts = _resolve_default(args.n_rollouts, 100)
    horizon = _resolve_default(args.horizon, 700)
    seed = _resolve_default(args.seed, 0)

    if args.run_dir:
        run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
    else:
        root = args.checkpoint_root if args.checkpoint_root else TRANSPORT_CHECKPOINT_ROOT
        root = root if os.path.isabs(root) else os.path.join(PROJECT_ROOT, root)
        runs = _list_transport_run_dirs(root)
        if not runs:
            print("No transport runs found.")
            return 0 if args.dry_run else 1
        run_dir = runs[0]

    models_dir = os.path.join(run_dir, "models")
    if not os.path.isdir(models_dir):
        print(f"models directory not found: {models_dir}")
        return 1

    candidates = _parse_epochs_csv(args.epochs)
    available = glob.glob(os.path.join(models_dir, "model_epoch_*.pth"))
    by_epoch = {}
    for p in available:
        ep = _parse_epoch_from_path(p)
        if ep is not None:
            by_epoch[ep] = p
    checkpoints = [(ep, by_epoch[ep]) for ep in candidates if ep in by_epoch]

    if not checkpoints:
        print("No checkpoints found for requested epochs.")
        return 0 if args.dry_run else 1

    rows = []
    for epoch, ckpt in checkpoints:
        cmd = [
            sys.executable,
            "-m",
            "robomimic.scripts.run_trained_agent",
            "--agent",
            ckpt,
            "--n_rollouts",
            str(n_rollouts),
            "--horizon",
            str(horizon),
            "--seed",
            str(seed),
        ]
        print(f"Evaluating epoch {epoch}: {' '.join(cmd)}")
        if args.dry_run:
            rows.append((epoch, None, None, None, 0))
            continue
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, capture_output=True, text=True)
        out = (result.stdout or "") + "\n" + (result.stderr or "")
        sr, ret, hor = _extract_stats_from_output(out)
        rows.append((epoch, sr, ret, hor, result.returncode))
        print(f"epoch={epoch} success={sr} return={ret} horizon={hor} rc={result.returncode}")

    rows_sorted = sorted(rows, key=lambda x: (-1 if x[1] is None else -x[1], x[0]))
    print("\nSweep summary (best first):")
    for epoch, sr, ret, hor, rc in rows_sorted:
        print(f"epoch={epoch:4d} success={sr} return={ret} horizon={hor} rc={rc}")

    return 0


def _candidate_render_gpu_ids(preferred_id: Optional[int] = None) -> List[int]:
    if preferred_id is not None:
        return [preferred_id]
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = []
        for tok in visible.split(","):
            tok = tok.strip()
            if tok.isdigit():
                ids.append(int(tok))
        return ids or [0]
    return list(range(8))


def _ensure_image_obs(obs, env):
    if "agentview_image" in obs or any(k.endswith("_image") for k in obs.keys()):
        return obs

    frame = None
    if hasattr(env, "env") and hasattr(env.env, "sim"):
        frame = env.env.sim.render(width=84, height=84, camera_name="agentview")
    elif hasattr(env, "sim"):
        frame = env.sim.render(width=84, height=84, camera_name="agentview")

    if frame is None:
        raise RuntimeError("No image observation available for vision conditioning.")

    obs = dict(obs)
    obs["agentview_image"] = frame.astype(np.uint8)
    return obs


def _load_policy_and_environment(checkpoint_path: str, render_gpu_id: Optional[int] = None):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=checkpoint_path,
            device=device,
            verbose=False,
        )
    except Exception as exc:
        err = str(exc).lower()
        is_cuda_oom = (
            torch.cuda.is_available()
            and device.type == "cuda"
            and ("out of memory" in err or "cudaerrormemoryallocation" in err)
        )
        if not is_cuda_oom:
            raise
        torch.cuda.empty_cache()
        device = torch.device("cpu")
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=checkpoint_path,
            device=device,
            verbose=False,
        )

    from diffusion.model import DIFFUSION_CONSTS

    cond_mode = DIFFUSION_CONSTS.get("cond_mode", "state")
    needs_vision = cond_mode in ("vision", "state+vision")

    env_meta = ckpt_dict["env_metadata"]
    if needs_vision:
        env_meta["env_kwargs"]["camera_names"] = ["agentview"]
        env_meta["env_kwargs"]["camera_heights"] = 84
        env_meta["env_kwargs"]["camera_widths"] = 84
        env_meta["env_kwargs"]["use_camera_obs"] = True
        env_meta["env_kwargs"]["has_offscreen_renderer"] = True

    if not needs_vision:
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        return policy, env, ckpt_dict

    last_exc = None
    for gpu_id in _candidate_render_gpu_ids(render_gpu_id):
        try:
            env_meta_try = dict(env_meta)
            env_kwargs_try = dict(env_meta_try.get("env_kwargs", {}))
            env_kwargs_try["render_gpu_device_id"] = int(gpu_id)
            env_meta_try["env_kwargs"] = env_kwargs_try
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta_try,
                render=False,
                render_offscreen=True,
                use_image_obs=True,
            )
            return policy, env, ckpt_dict
        except Exception as exc:
            last_exc = exc
            err = str(exc).lower()
            if "framebuffer is not complete" in err or "egl" in err:
                continue
            raise

    raise RuntimeError("Failed to create offscreen renderer on candidate GPUs.") from last_exc


def _load_policy_env_basic(
    checkpoint_path: str,
    *,
    render_offscreen: bool = False,
    use_image_obs: bool = False,
    camera_size: int = 84,
):
    import robomimic.utils.env_utils as EnvUtils
    import robomimic.utils.file_utils as FileUtils
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=checkpoint_path,
        device=device,
        verbose=False,
    )

    env_meta = ckpt_dict["env_metadata"]
    if render_offscreen:
        env_meta["env_kwargs"]["camera_names"] = ["agentview"]
        env_meta["env_kwargs"]["camera_heights"] = camera_size
        env_meta["env_kwargs"]["camera_widths"] = camera_size

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=render_offscreen,
        use_image_obs=use_image_obs,
    )
    return policy, env, ckpt_dict


def _get_action_dimension(env, policy) -> int:
    try:
        return int(env.action_spec[0].shape[0])
    except Exception:
        pass
    try:
        return int(env.action_dimension)
    except Exception:
        return len(policy(env.reset()))


def _run_single_rollout(policy, env, noise_std: float, method: str, filt, horizon: int, obs_keys, t_start: int):
    import torch

    obs = env.reset()
    policy.start_episode()
    ep_reward = 0.0
    success = False

    if method == "diffusion":
        from diffusion.model import DIFFUSION_CONSTS, build_cond_vec, diffusion_denoise_action, diffusion_denoise_action_window
        use_window = DIFFUSION_CONSTS.get("H", 1) > 1
    else:
        DIFFUSION_CONSTS = None
        build_cond_vec = None
        diffusion_denoise_action = None
        diffusion_denoise_action_window = None
        use_window = False

    for _ in range(horizon):
        with torch.no_grad():
            ac = policy(obs)
            if isinstance(ac, torch.Tensor):
                ac = ac.cpu().numpy()
            elif isinstance(ac, (list, tuple)):
                ac = np.asarray(ac[0]) if len(ac) > 0 else np.array([])

        noisy_ac = ac + np.random.normal(0, noise_std, size=ac.shape) if noise_std > 0 else ac

        if method == "diffusion":
            cond_mode = DIFFUSION_CONSTS.get("cond_mode", "state")
            if cond_mode in ("vision", "state+vision"):
                obs = _ensure_image_obs(obs, env)
            cond_vec = build_cond_vec(obs, obs_keys, cond_mode)
            if use_window:
                ac = diffusion_denoise_action_window(noisy_action_vec=noisy_ac, cond_vec=cond_vec, t_start=t_start)
            else:
                ac = diffusion_denoise_action(noisy_action_vec=noisy_ac, cond_vec=cond_vec, t_start=t_start)
        elif filt is not None:
            ac = filt.update(noisy_ac)
        else:
            ac = noisy_ac

        ac = np.clip(ac, -1.0, 1.0)
        obs, reward, done, _ = env.step(ac)
        ep_reward += reward
        if env.is_success()["task"]:
            success = True
            break
        if done:
            break

    return ep_reward, success


def _find_best_square_checkpoint(checkpoint_dir: str) -> Optional[str]:
    pattern = os.path.join(checkpoint_dir, "*/models/model_epoch_*.pth")
    candidates = sorted(glob.glob(pattern))
    best_ckpt = None
    best_success = -1.0
    success_pattern = re.compile(r"_success_([0-9]+(?:\.[0-9]+)?)\.pth$")
    for ckpt in candidates:
        m = success_pattern.search(ckpt)
        if not m:
            continue
        success = float(m.group(1))
        if success > best_success:
            best_success = success
            best_ckpt = ckpt
    return best_ckpt


def _find_square_checkpoint_recursive(epoch: int, checkpoint_name: Optional[str] = None) -> Optional[str]:
    root = os.path.join(PROJECT_ROOT, "checkpoints", "bc_rnn_square")
    if checkpoint_name:
        pattern = os.path.join(root, "**", "models", checkpoint_name)
        matches = sorted(glob.glob(pattern, recursive=True))
        return matches[-1] if matches else None
    pattern = os.path.join(root, "**", "models", f"model_epoch_{epoch}.pth")
    matches = sorted(glob.glob(pattern, recursive=True))
    return matches[-1] if matches else None


def _run_simple_robustness(
    *,
    task: str,
    checkpoint_dir: str,
    default_epoch: int,
    default_horizon: int,
    noise_levels: Sequence[float],
    methods: Sequence[str],
    csv_prefix: str,
    args: argparse.Namespace,
    support_best: bool = False,
):
    results_dir = os.path.join(PROJECT_ROOT, "results", task)
    os.makedirs(results_dir, exist_ok=True)
    configure_mujoco_env(force_gl_egl=(task == "can"))
    np.random.seed(_resolve_default(args.seed, 0))

    if args.checkpoint_path:
        ckpt = args.checkpoint_path if os.path.isabs(args.checkpoint_path) else os.path.join(PROJECT_ROOT, args.checkpoint_path)
        if not os.path.isfile(ckpt):
            print(f"Checkpoint file not found: {ckpt}")
            return 0 if args.dry_run else 1
    elif support_best and args.best:
        ckpt = _find_best_square_checkpoint(checkpoint_dir)
        if not ckpt:
            print("No best-success checkpoint found.")
            return 1
    else:
        ckpt = _resolve_checkpoint(checkpoint_dir, _resolve_default(args.epoch, default_epoch))
        if not ckpt:
            return 0 if args.dry_run else 1

    n_rollouts = _resolve_default(args.n_rollouts, 50)
    horizon = _resolve_default(args.horizon, default_horizon)
    seed = _resolve_default(args.seed, 0)

    if args.dry_run:
        print(f"[dry_run] task={task}")
        print(f"[dry_run] checkpoint={ckpt}")
        print(f"[dry_run] methods={list(methods)}")
        print(f"[dry_run] noise_levels={list(noise_levels)}")
        return 0

    policy, env, _ = _load_policy_env_basic(ckpt)
    action_dim = _get_action_dimension(env, policy)

    all_results = []
    for noise_std in noise_levels:
        for method in methods:
            rewards = []
            successes = []
            for _ in range(n_rollouts):
                filt = _create_filter(method, action_dim)
                ep_reward, success = _run_single_rollout(
                    policy=policy,
                    env=env,
                    noise_std=noise_std,
                    method=method,
                    filt=filt,
                    horizon=horizon,
                    obs_keys=None,
                    t_start=40,
                )
                rewards.append(ep_reward)
                successes.append(int(success))

            all_results.append(
                {
                    "noise_std": noise_std,
                    "mean_reward": float(np.mean(rewards)),
                    "success_rate": float(np.mean(successes)),
                    "n_rollouts": n_rollouts,
                    "seed": seed,
                    "method": method,
                }
            )

    print_robustness_summary(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        "task": task,
        "checkpoint_path": ckpt,
        "epoch": args.epoch,
        "noise_levels": list(noise_levels),
        "methods": list(methods),
        "n_rollouts": n_rollouts,
        "horizon": horizon,
        "seed": seed,
    }
    json_path = os.path.join(results_dir, f"robustness_eval_{timestamp}.json")
    csv_path = os.path.join(results_dir, f"{csv_prefix}_{timestamp}.csv")
    save_results_json(all_results, config, json_path)
    save_results_csv(
        all_results,
        csv_path,
        fieldnames=["method", "noise_std", "mean_reward", "success_rate", "n_rollouts", "seed"],
    )
    return 0


def _run_filter_sweep(task: str, args: argparse.Namespace) -> int:
    checkpoint_dir = _task_single_checkpoint_dir(task)
    results_dir = os.path.join(PROJECT_ROOT, "results", task)
    os.makedirs(results_dir, exist_ok=True)
    configure_mujoco_env(force_gl_egl=(task == "can"))

    epoch = _resolve_default(args.epoch, 600)
    ckpt = _resolve_checkpoint(checkpoint_dir, epoch)
    if not ckpt:
        return 0 if args.dry_run else 1

    n_rollouts = _resolve_default(args.n_rollouts, 50)
    horizon = _resolve_default(args.horizon, 400)
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0]
    methods = ["none", "ema", "median", "kalman"]
    ema_alphas = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8]
    median_ks = [3, 5, 7, 9]
    kalman_qs = [1e-5, 1e-4, 1e-3, 1e-2]
    kalman_rs = [1e-3, 1e-2, 1e-1]

    if args.dry_run:
        print(f"[dry_run] task={task} suite=filters")
        print(f"[dry_run] checkpoint={ckpt}")
        print(f"[dry_run] methods={methods}")
        print(f"[dry_run] noise_levels={noise_levels}")
        return 0

    policy, env, _ = _load_policy_env_basic(ckpt)
    action_dim = _get_action_dimension(env, policy)
    rows = []

    def param_grid(method: str):
        if method == "ema":
            return [{"alpha": a} for a in ema_alphas]
        if method == "median":
            return [{"k": k} for k in median_ks]
        if method == "kalman":
            return [{"Q": q, "R": r} for q in kalman_qs for r in kalman_rs]
        return [{}]

    def create_filter(method: str, params: Dict):
        if method == "ema":
            return EMAFilter(dim=action_dim, alpha=params["alpha"])
        if method == "median":
            return MedianFilter(dim=action_dim, k=params["k"])
        if method == "kalman":
            return KalmanFilter(dim=action_dim, Q=params["Q"], R=params["R"])
        return None

    for noise_std in noise_levels:
        for method in methods:
            best_reward = -np.inf
            best_success = 0.0
            best_params = None
            for params in param_grid(method):
                rewards = []
                successes = []
                for _ in range(n_rollouts):
                    filt = create_filter(method, params)
                    ep_reward, success = _run_single_rollout(
                        policy=policy,
                        env=env,
                        noise_std=noise_std,
                        method="none",
                        filt=filt,
                        horizon=horizon,
                        obs_keys=None,
                        t_start=40,
                    )
                    rewards.append(ep_reward)
                    successes.append(int(success))
                mean_reward = float(np.mean(rewards))
                if mean_reward > best_reward:
                    best_reward = mean_reward
                    best_success = float(np.mean(successes))
                    best_params = params

            rows.append(
                {
                    "method": f"{method}(best)",
                    "noise_std": noise_std,
                    "mean_reward": best_reward,
                    "success_rate": best_success,
                    "n_rollouts": n_rollouts,
                    "seed": _resolve_default(args.seed, 0),
                    "best_params": str(best_params),
                }
            )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(results_dir, f"filter_sweep_{task}_{timestamp}.csv")
    save_results_csv(
        rows,
        csv_path,
        fieldnames=["method", "noise_std", "mean_reward", "success_rate", "n_rollouts", "seed", "best_params"],
    )
    print_robustness_summary(rows)
    return 0


def _render_frame(env, width: int = 256, height: int = 256, camera_name: str = "agentview"):
    try:
        frame = env.render(mode="rgb_array", height=height, width=width, camera_name=camera_name)
        if frame is not None:
            return frame
    except Exception:
        pass
    sim = getattr(getattr(env, "env", None), "sim", None)
    if sim is not None:
        return sim.render(width=width, height=height, camera_name=camera_name)
    return None


def _run_video_eval(task: str, args: argparse.Namespace) -> int:
    configure_mujoco_env(force_gl_egl=False)
    results_dir = os.path.join(PROJECT_ROOT, "results", task)
    os.makedirs(results_dir, exist_ok=True)

    if task == "square":
        if args.checkpoint_path:
            ckpt = args.checkpoint_path if os.path.isabs(args.checkpoint_path) else os.path.join(PROJECT_ROOT, args.checkpoint_path)
        else:
            ckpt = _find_square_checkpoint_recursive(_resolve_default(args.epoch, 1000), args.checkpoint_name)
    else:
        ckpt = _resolve_checkpoint(_task_single_checkpoint_dir(task), _resolve_default(args.epoch, 600))
    if not ckpt or not os.path.exists(ckpt):
        print("No checkpoint found for video evaluation.")
        return 0 if args.dry_run else 1

    n_rollouts = _resolve_default(args.n_rollouts, 10)
    horizon = _resolve_default(args.horizon, 500 if task == "square" else 400)

    if args.dry_run:
        print(f"[dry_run] task={task} suite=robustness-video checkpoint={ckpt}")
        print(f"[dry_run] noise_levels={VIDEO_NOISE_LEVELS[task]} n_rollouts={n_rollouts} horizon={horizon}")
        return 0

    import imageio

    policy, env, _ = _load_policy_env_basic(ckpt, render_offscreen=True, use_image_obs=False, camera_size=256)
    for noise_std in VIDEO_NOISE_LEVELS[task]:
        for ep in range(n_rollouts):
            obs = env.reset()
            policy.start_episode()
            record_video = ep == 0
            frames = []

            for _ in range(horizon):
                action = policy(obs)
                if noise_std > 0:
                    action = action + np.random.normal(0, noise_std, size=action.shape)
                action = np.clip(action, -1.0, 1.0)
                obs, _, done, _ = env.step(action)
                if record_video:
                    frame = _render_frame(env, 256, 256, "agentview")
                    if frame is not None:
                        frames.append(frame)
                if env.is_success()["task"] or done:
                    break

            if record_video and frames:
                if task == "square":
                    name = os.path.splitext(os.path.basename(ckpt))[0]
                    video_name = f"square_noise_{noise_std:.2f}_{name}.mp4"
                elif task == "can":
                    video_name = f"can_noise_{noise_std:.2f}.mp4"
                else:
                    video_name = f"noise_{noise_std:.2f}.mp4"
                imageio.mimsave(os.path.join(results_dir, video_name), frames, fps=20)

    return 0


def _run_robustness_diffusion(task: str, args: argparse.Namespace) -> int:
    cfg = ROBUSTNESS_CONFIGS[task]
    os.makedirs(cfg["results_dir"], exist_ok=True)

    np.random.seed(_resolve_default(args.seed, 0))
    configure_mujoco_env(force_gl_egl=(task == "can"))

    if task == "square" and args.checkpoint_path:
        ckpt = args.checkpoint_path
        if not os.path.isfile(ckpt):
            print(f"Checkpoint file not found: {ckpt}")
            return 0 if args.dry_run else 1
    elif task == "square" and args.best:
        ckpt = _find_best_square_checkpoint(cfg["checkpoint_dir"])
        if not ckpt:
            print("No best-success square checkpoint found.")
            return 0 if args.dry_run else 1
    else:
        epoch = _resolve_default(args.epoch, cfg["default_epoch"])
        ckpt = _resolve_checkpoint(cfg["checkpoint_dir"], epoch)
        if not ckpt:
            return 0 if args.dry_run else 1

    n_rollouts = _resolve_default(args.n_rollouts, 50)
    horizon = _resolve_default(args.horizon, cfg["default_horizon"])
    seed = _resolve_default(args.seed, 0)
    t_starts = args.t_start if args.t_start else [40]

    if args.dry_run:
        methods_preview = list(cfg["manual_methods"])
        if args.diffusion_model:
            methods_preview.extend([f"diffusion[t={t}]" for t in t_starts])
        print(f"[dry_run] task={task} suite=robustness-diffusion")
        print(f"[dry_run] checkpoint={ckpt}")
        print(f"[dry_run] n_rollouts={n_rollouts} horizon={horizon} seed={seed}")
        print(f"[dry_run] noise_levels={cfg['noise_levels']}")
        print(f"[dry_run] methods={methods_preview}")
        return 0

    methods = list(cfg["manual_methods"])
    obs_keys = None
    if args.diffusion_model:
        from diffusion.model import get_diffusion_obs_keys, load_diffusion_model

        load_diffusion_model(args.diffusion_model)
        obs_keys = get_diffusion_obs_keys()
        methods.extend([f"diffusion[t={t}]" for t in t_starts])

    policy, env, _ = _load_policy_and_environment(ckpt, render_gpu_id=args.render_gpu_id)
    action_dim = _get_action_dimension(env, policy)

    all_results = []
    for noise_std in cfg["noise_levels"]:
        for method in methods:
            base_method = method.split("[")[0]
            t_start = 40
            if base_method == "diffusion" and "[t=" in method:
                t_start = int(method.split("[t=")[1].rstrip("]"))

            rewards = []
            successes = []
            for _ in range(n_rollouts):
                filt = _create_filter(base_method, action_dim)
                ep_reward, success = _run_single_rollout(
                    policy=policy,
                    env=env,
                    noise_std=noise_std,
                    method=base_method,
                    filt=filt,
                    horizon=horizon,
                    obs_keys=obs_keys,
                    t_start=t_start,
                )
                rewards.append(ep_reward)
                successes.append(int(success))

            mean_reward = float(np.mean(rewards))
            success_rate = float(np.mean(successes))
            all_results.append(
                {
                    "noise_std": noise_std,
                    "mean_reward": mean_reward,
                    "success_rate": success_rate,
                    "n_rollouts": n_rollouts,
                    "seed": seed,
                    "method": method,
                    "t_start": t_start if base_method == "diffusion" else None,
                }
            )
            print(
                f"Noise {noise_std:.2f} | Method {method:<18} | "
                f"Reward={mean_reward:7.4f} | Success={success_rate * 100:6.2f}%"
            )

    print_robustness_summary(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        "task": task,
        "epoch": args.epoch,
        "checkpoint_path": ckpt,
        "n_rollouts": n_rollouts,
        "horizon": horizon,
        "seed": seed,
        "noise_levels": cfg["noise_levels"],
        "methods": methods,
        "diffusion_model": args.diffusion_model,
        "t_start": t_starts,
    }

    json_path = os.path.join(cfg["results_dir"], f"robustness_eval_{timestamp}.json")
    csv_path = os.path.join(cfg["results_dir"], f"{cfg['csv_prefix']}_{timestamp}.csv")
    save_results_json(all_results, config, json_path)
    save_results_csv(
        all_results,
        csv_path,
        fieldnames=["method", "noise_std", "mean_reward", "success_rate", "n_rollouts", "seed", "t_start"],
    )
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Unified launcher for evaluation scripts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Tips:\n"
            "  - Use --list to view available task/suite pairs.\n"
            "  - Use -- to pass script-specific flags not listed here.\n"
            "Examples:\n"
            "  python evaluation/eval.py --task can --suite standard --epoch 600\n"
            "  python evaluation/eval.py --task lift --suite robustness --n_rollouts 100\n"
            "  python evaluation/eval.py --task square --suite robustness-diffusion "
            "--diffusion_model diffusion_models/square_diffusion_model.pt --t_start 10 20 40\n"
            "  python evaluation/eval.py --task transport --suite sweep -- --max_runs 5"
        ),
    )

    parser.add_argument("--task", choices=all_tasks(), help="Task name")
    parser.add_argument("--suite", help="Evaluation suite for the task")

    parser.add_argument("--list", action="store_true", help="List all supported task/suite pairs and exit")
    parser.add_argument("--dry_run", action="store_true", help="Print target command without executing")

    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--checkpoint_root", type=str, default=None)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--diffusion_model", type=str, default=None)

    parser.add_argument("--epoch", type=int, default=None)
    parser.add_argument("--n_rollouts", type=int, default=None)
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--render_gpu_id", type=int, default=None)
    parser.add_argument("--epochs", type=str, default="", help="Comma-separated epoch list for transport sweep")

    parser.add_argument("--best", action="store_true")
    parser.add_argument("--all_runs", action="store_true")

    parser.add_argument("--t_start", type=int, nargs="+", default=None)

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list:
        print_supported_pairs()
        return 0

    validation_error = validate_task_suite(args.task, args.suite)
    if validation_error:
        print(f"Error: {validation_error}")
        print("Use --list to inspect valid options.")
        return 2

    # Internal implementations (migrated suites)
    if args.task in ("can", "lift") and args.suite == "standard":
        return _run_standard_eval(args.task, args)

    if args.task == "transport" and args.suite == "standard":
        return _run_transport_standard(args)

    if args.task == "transport" and args.suite == "sweep":
        return _run_transport_sweep(args)

    if args.task in ("can", "lift", "square") and args.suite == "robustness-diffusion":
        return _run_robustness_diffusion(args.task, args)

    if args.task in ("can", "lift") and args.suite == "robustness":
        return _run_simple_robustness(
            task=args.task,
            checkpoint_dir=_task_single_checkpoint_dir(args.task),
            default_epoch=600,
            default_horizon=400,
            noise_levels=[0.0, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0],
            methods=["none"],
            csv_prefix=f"robustness_eval_{args.task}",
            args=args,
            support_best=False,
        )

    if args.task == "lift" and args.suite == "kalman":
        return _run_simple_robustness(
            task="lift",
            checkpoint_dir=_task_single_checkpoint_dir("lift"),
            default_epoch=600,
            default_horizon=400,
            noise_levels=[0.1, 0.2, 0.5, 0.75],
            methods=["kalman"],
            csv_prefix="robustness_eval_lift",
            args=args,
            support_best=False,
        )

    if args.task in ("can", "lift") and args.suite == "filters":
        return _run_filter_sweep(args.task, args)

    if args.task == "square" and args.suite == "filters":
        return _run_simple_robustness(
            task="square",
            checkpoint_dir=os.path.join(PROJECT_ROOT, "checkpoints", "bc_rnn_square", "bc_rnn_square_v2"),
            default_epoch=2000,
            default_horizon=500,
            noise_levels=[0.0, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0],
            methods=["none", "kalman", "ema", "median"],
            csv_prefix="robustness_eval_square",
            args=args,
            support_best=False,
        )

    if args.task == "square" and args.suite == "filters-transformer":
        return _run_simple_robustness(
            task="square",
            checkpoint_dir=os.path.join(PROJECT_ROOT, "checkpoints", "bc_rnn_square", "bc_rnn_square_v3"),
            default_epoch=2000,
            default_horizon=500,
            noise_levels=[0.0, 0.04, 0.08, 0.12, 0.16, 0.2],
            methods=["none", "kalman", "ema", "median"],
            csv_prefix="robustness_eval_square",
            args=args,
            support_best=True,
        )

    if args.task in ("can", "lift", "square") and args.suite == "robustness-video":
        return _run_video_eval(args.task, args)

    print(f"Unsupported internal route: task={args.task}, suite={args.suite}")
    return 2


if __name__ == "__main__":
    sys.exit(main())
