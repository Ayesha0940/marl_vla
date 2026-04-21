"""
Evaluate lift policy robustness with action noise across multiple predefined noise levels.
Compares manual filters (Kalman, EMA, Median) against diffusion-based denoising.

Tests noise standard deviations: [0.1, 0.2, 0.5, 0.75]
Tests methods: ["none", "kalman", "ema", "median", "diffusion"]

Results are saved in both JSON and CSV formats for easy analysis.

Examples:
    # With diffusion
    python eval_robustness_lift.py \\
        --diffusion_model diffusion_models/lift_diffusion_model.pt

    # Without diffusion (manual filters only)
    python eval_robustness_lift.py

    # Sweep t_start values
    python eval_robustness_lift.py \\
        --diffusion_model diffusion_models/lift_diffusion_model.pt \\
        --t_start 10 20 40
"""

import os
import sys
import glob
import argparse
import json
import csv
from datetime import datetime

import numpy as np
from collections import deque


# =========================
# FILTER CLASSES
# =========================

class EMAFilter:
    """Exponential Moving Average filter for action smoothing."""
    def __init__(self, dim, alpha=0.2):
        self.alpha = alpha
        self.x = np.zeros(dim)

    def update(self, a):
        self.x = self.alpha * a + (1 - self.alpha) * self.x
        return self.x


class MedianFilter:
    """Median filter for action noise reduction."""
    def __init__(self, dim, k=5):
        self.buffer = deque(maxlen=k)

    def update(self, a):
        self.buffer.append(a)
        return np.median(self.buffer, axis=0)


class KalmanFilter:
    """Kalman filter for action state estimation."""
    def __init__(self, dim, Q=1e-3, R=1e-2):
        self.dim = dim
        self.x = np.zeros(dim)
        self.P = np.eye(dim)
        self.Q = Q * np.eye(dim)
        self.R = R * np.eye(dim)

    def update(self, z):
        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q
        # Gain
        K = P_pred @ np.linalg.inv(P_pred + self.R)
        # Update
        self.x = x_pred + K @ (z - x_pred)
        self.P = (np.eye(self.dim) - K) @ P_pred
        return self.x


# =========================
# PATH SETUP
# =========================

PROJECT_ROOT   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints/bc_rnn_can/bc_rnn_can')
RESULTS_DIR    = os.path.join(PROJECT_ROOT, 'results', 'can')

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# MUJOCO SETUP
# =========================

def setup_mujoco():
    """Configure LD_LIBRARY_PATH for MuJoCo."""
    paths = [
        os.path.expanduser('~/.mujoco/mujoco210/bin'),
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib/x86_64-linux-gnu/nvidia',
    ]
    current  = os.environ.get('LD_LIBRARY_PATH', '')
    new_paths = [p for p in paths if os.path.exists(p)]
    os.environ['LD_LIBRARY_PATH'] = ":".join(new_paths + [current])
    os.environ.setdefault('MUJOCO_GL', 'egl')
    print("🔧 LD_LIBRARY_PATH configured")
    print(f"🔧 MUJOCO_GL={os.environ.get('MUJOCO_GL')}")


# =========================
# CHECKPOINT FINDER
# =========================

def find_latest_checkpoint(epoch=600):
    """Find the latest BC-RNN checkpoint for a given epoch."""
    pattern = os.path.join(CHECKPOINT_DIR, f'*/models/model_epoch_{epoch}.pth')
    matches = sorted(glob.glob(pattern))
    if matches:
        return matches[-1]
    print("❌ Checkpoint not found.")
    return None


# =========================
# ARGUMENT PARSER
# =========================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Evaluate can policy robustness with action noise and filtering/diffusion methods"
    )
    parser.add_argument('--epoch', type=int, default=600,
                        help='BC-RNN checkpoint epoch to load')
    parser.add_argument('--n_rollouts', type=int, default=50,
                        help='Number of rollouts per configuration')
    parser.add_argument('--horizon', type=int, default=400,
                        help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')

    # Diffusion args
    parser.add_argument('--diffusion_model', type=str, default=None,
                        help='Path to trained diffusion model .pt file. '
                             'If not provided, diffusion method is skipped.')
    parser.add_argument('--t_start', type=int, nargs='+', default=[40],
                        help='Reverse diffusion start step(s). '
                             'Multiple values run separate sweeps. '
                             'Default: 40. Try: --t_start 10 20 40')
        parser.add_argument('--render_gpu_id', type=int, default=None,
                        help='Preferred GPU id for MuJoCo offscreen rendering. '
                            'If not set, script will try visible GPUs in order.')

    return parser.parse_args()


# =========================
# FILTER INITIALIZATION
# =========================

def create_filter(method, action_dim):
    """
    Create filter instance based on method name.
    Returns None for 'none' and 'diffusion' — diffusion is handled
    separately in run_single_rollout().
    """
    if method == "kalman":
        return KalmanFilter(dim=action_dim)
    elif method == "ema":
        return EMAFilter(dim=action_dim)
    elif method == "median":
        return MedianFilter(dim=action_dim)
    else:
        # "none" and "diffusion" both return None here
        return None


# =========================
# POLICY AND ENVIRONMENT LOADING
# =========================

def _candidate_render_gpu_ids(preferred_id=None):
    """Build ordered list of render GPU ids to try for offscreen EGL."""
    if preferred_id is not None:
        return [preferred_id]

    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if visible:
        ids = []
        for tok in visible.split(','):
            tok = tok.strip()
            if tok.isdigit():
                ids.append(int(tok))
        return ids or [0]

    return list(range(8))


def load_policy_and_environment(checkpoint_path, render_gpu_id=None):
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=checkpoint_path,
            device=device,
            verbose=False
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

        print("⚠️  CUDA OOM while loading checkpoint on GPU. Retrying on CPU...")
        torch.cuda.empty_cache()
        cpu_device = torch.device("cpu")
        policy, ckpt_dict = FileUtils.policy_from_checkpoint(
            ckpt_path=checkpoint_path,
            device=cpu_device,
            verbose=False
        )
        device = cpu_device

    print("✅ Policy loaded")
    print(f"🖥️  Policy device: {device}")

    # Check if diffusion model needs camera obs
    from diffusion.model import DIFFUSION_CONSTS
    cond_mode   = DIFFUSION_CONSTS.get("cond_mode", "state")
    needs_vision = cond_mode in ("vision", "state+vision")

    env_meta = ckpt_dict["env_metadata"]

    if needs_vision:
        env_meta['env_kwargs']['camera_names'] = ['agentview']
        env_meta['env_kwargs']['camera_heights'] = 84
        env_meta['env_kwargs']['camera_widths'] = 84
        env_meta['env_kwargs']['use_camera_obs'] = True
        env_meta['env_kwargs']['has_offscreen_renderer'] = True
        print("📷 Camera obs enabled for vision conditioning")

    if not needs_vision:
        env = EnvUtils.create_env_from_metadata(
            env_meta=env_meta,
            render=False,
            render_offscreen=False,
            use_image_obs=False,
        )
        print("✅ Environment created")
        return policy, env

    last_exc = None
    candidates = _candidate_render_gpu_ids(preferred_id=render_gpu_id)
    for gpu_id in candidates:
        try:
            env_meta_try = dict(env_meta)
            env_kwargs_try = dict(env_meta_try.get('env_kwargs', {}))
            env_kwargs_try['render_gpu_device_id'] = int(gpu_id)
            env_meta_try['env_kwargs'] = env_kwargs_try

            print(f"🎯 Trying render_gpu_device_id={gpu_id}")
            env = EnvUtils.create_env_from_metadata(
                env_meta=env_meta_try,
                render=False,
                render_offscreen=True,
                use_image_obs=True,
            )
            print(f"✅ Environment created (render GPU {gpu_id})")
            return policy, env
        except Exception as exc:
            last_exc = exc
            err = str(exc).lower()
            if "framebuffer is not complete" in err or "egl" in err:
                print(f"⚠️  Render GPU {gpu_id} failed: {exc}")
                continue
            raise

    raise RuntimeError(
        "Failed to create MuJoCo offscreen renderer on all candidate GPUs. "
        "Try --render_gpu_id with a less loaded GPU (e.g., 0)."
    ) from last_exc


# =========================
# ACTION DIMENSION DETECTION
# =========================

def get_action_dimension(env, policy):
    """Detect action dimension from environment."""
    try:
        action_dim = env.action_spec[0].shape[0]
    except:
        action_dim = len(policy(env.reset()))
    return action_dim


def ensure_image_obs(obs, env):
    """
    Ensure observation dict contains at least one '*_image' key.

    Some robomimic wrappers may omit camera frames in obs even when
    offscreen rendering is enabled. In that case, render one frame explicitly.
    """
    if 'agentview_image' in obs or any(k.endswith('_image') for k in obs.keys()):
        return obs

    frame = None
    if hasattr(env, 'env') and hasattr(env.env, 'sim'):
        frame = env.env.sim.render(width=84, height=84, camera_name='agentview')
    elif hasattr(env, 'sim'):
        frame = env.sim.render(width=84, height=84, camera_name='agentview')

    if frame is None:
        raise RuntimeError(
            "Diffusion vision conditioning requested, but no image key was found "
            "and offscreen render returned None."
        )

    obs = dict(obs)
    obs['agentview_image'] = frame.astype(np.uint8)
    return obs


# =========================
# SINGLE ROLLOUT EVALUATION
# =========================

def run_single_rollout(policy, env, noise_std, method, filt, horizon,
                       obs_keys=None, t_start=40):
    """
    Run a single episode with noisy actions and optional filtering/denoising.

    Args:
        policy:    BC-RNN policy
        env:       robomimic environment
        noise_std: std of Gaussian action noise
        method:    "none" | "kalman" | "ema" | "median" | "diffusion"
        filt:      filter instance (None for "none" and "diffusion")
        horizon:   max steps per episode
        obs_keys:  list of obs keys for diffusion state conditioning
                   (required when method == "diffusion")
        t_start:   reverse diffusion start step
                   (only used when method == "diffusion")

    Returns:
        tuple: (episode_reward, success)
    """
    obs = env.reset()
    policy.start_episode()
    ep_reward = 0.0
    success   = False

    # Import diffusion utils lazily — only pay the import cost if needed
    if method == "diffusion":
        from diffusion.model import (
            build_cond_vec,
            diffusion_denoise_action,
            diffusion_denoise_action_window,
            DIFFUSION_CONSTS,
        )
        # Choose inference function based on H saved in loaded model
        use_window = DIFFUSION_CONSTS.get("H", 1) > 1

    for step in range(horizon):
        # Get clean action from policy
        action = policy(obs)

        # Add noise
        if noise_std > 0:
            noise        = np.random.normal(0, noise_std, size=action.shape)
            noisy_action = action + noise
        else:
            noisy_action = action

        # -----------------------------------------------
        # Denoising branch
        # -----------------------------------------------
        if method == "diffusion":
            cond_mode = DIFFUSION_CONSTS.get("cond_mode", "state")
            if cond_mode in ("vision", "state+vision"):
                obs = ensure_image_obs(obs, env)
            cond_vec = build_cond_vec(obs, obs_keys, cond_mode)
            if use_window:
                # H>1: denoise full window, execute only step 0
                action = diffusion_denoise_action_window(
                    noisy_action_vec = noisy_action,
                    cond_vec         = cond_vec,
                    t_start          = t_start,
                )
            else:
                # H=1: single step denoising
                action = diffusion_denoise_action(
                    noisy_action_vec = noisy_action,
                    cond_vec         = cond_vec,
                    t_start          = t_start,
                )

        elif filt is not None:
            # Manual filters: kalman, ema, median
            action = filt.update(noisy_action)

        else:
            # No denoising
            action = noisy_action

        # Clip to valid action range
        action = np.clip(action, -1.0, 1.0)

        # Environment step
        obs, reward, done, _ = env.step(action)
        ep_reward += reward

        if env.is_success()["task"]:
            success = True
            break

        if done:
            break

    return ep_reward, success


# =========================
# NOISE LEVEL EVALUATION
# =========================

def evaluate_noise_and_method(policy, env, noise_std, method, action_dim,
                               n_rollouts, horizon, obs_keys=None, t_start=40):
    """
    Evaluate policy performance at a specific noise level with a given method.

    Args:
        policy:     BC-RNN policy
        env:        robomimic environment
        noise_std:  std of Gaussian action noise
        method:     "none" | "kalman" | "ema" | "median" | "diffusion[t=N]"
        action_dim: action space dimension
        n_rollouts: number of rollouts
        horizon:    max steps per episode
        obs_keys:   obs keys for diffusion conditioning
        t_start:    reverse diffusion start step

    Returns:
        dict: result record
    """
    rewards   = []
    successes = []

    # Strip t_start label from method name for filter creation
    base_method = method.split("[")[0]  # "diffusion[t=40]" → "diffusion"

    for ep in range(n_rollouts):
        filt = create_filter(base_method, action_dim)
        ep_reward, success = run_single_rollout(
            policy, env, noise_std, base_method, filt, horizon,
            obs_keys=obs_keys, t_start=t_start
        )
        rewards.append(ep_reward)
        successes.append(int(success))

    mean_reward  = float(np.mean(rewards))
    success_rate = float(np.mean(successes))

    print(f"  📊 Method:       {method}")
    print(f"  📊 Mean Reward:  {mean_reward:.4f}")
    print(f"  📊 Success Rate: {success_rate * 100:.2f}%")

    return {
        "noise_std":    noise_std,
        "mean_reward":  mean_reward,
        "success_rate": success_rate,
        "n_rollouts":   n_rollouts,
        "seed":         0,
        "method":       method,
        "t_start":      t_start if base_method == "diffusion" else None,
    }


# =========================
# RESULTS PRINTING
# =========================

def print_results_summary(results):
    """Print aligned summary table of all results."""
    print("\n" + "=" * 65)
    print("📊 SUMMARY RESULTS")
    print("=" * 65)
    for r in results:
        print(f"Noise {r['noise_std']:4.2f} | Method {r['method']:18s} | "
              f"Reward={r['mean_reward']:7.4f} | Success={r['success_rate']*100:5.2f}%")
    print("=" * 65)


# =========================
# RESULTS SAVING
# =========================

def save_results_to_json(results, config, timestamp):
    """Save results to JSON."""
    json_path = os.path.join(RESULTS_DIR, f"robustness_eval_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump({"results": results, "config": config}, f, indent=4)
    print(f"💾 Saved JSON: {json_path}")
    return json_path


def save_results_to_csv(results, timestamp):
    """Save results to CSV."""
    csv_path = os.path.join(RESULTS_DIR, f"robustness_eval_can_{timestamp}.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['method', 'noise_std', 'mean_reward', 'success_rate',
                      'n_rollouts', 'seed', 't_start']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                'method':       r['method'],
                'noise_std':    r['noise_std'],
                'mean_reward':  f"{r['mean_reward']:.4f}",
                'success_rate': f"{r['success_rate']:.4f}",
                'n_rollouts':   r['n_rollouts'],
                'seed':         r['seed'],
                't_start':      r['t_start'] if r['t_start'] is not None else '',
            })
    print(f"💾 Saved CSV:  {csv_path}")
    return csv_path


# =========================
# MAIN
# =========================

def main():
    args = parse_arguments()
    np.random.seed(args.seed)

    # Setup
    setup_mujoco()

    # -------------------------
    # Load diffusion model
    # -------------------------
    obs_keys = None
    use_diffusion = args.diffusion_model is not None

    if use_diffusion:
        from diffusion.model import load_diffusion_model, get_diffusion_obs_keys
        print(f"\n🌀 Loading diffusion model: {args.diffusion_model}")
        load_diffusion_model(args.diffusion_model)
        obs_keys = get_diffusion_obs_keys()
        print(f"   obs_keys: {obs_keys}")
        print(f"   t_start sweep: {args.t_start}")
    else:
        print("\n⚠️  No diffusion model provided — running manual filters only")

    # -------------------------
    # Build methods list
    # -------------------------
    methods = ["none"]

    if use_diffusion:
        # Add one diffusion entry per t_start value
        for t in args.t_start:
            methods.append(f"diffusion[t={t}]")

    # Noise levels
    noise_levels = [0.1, 0.3, 0.6, 0.9, 1.25, 1.5, 1.75, 2.0]

    # -------------------------
    # Load BC-RNN policy
    # -------------------------
    agent_path = find_latest_checkpoint(args.epoch)
    if not agent_path:
        return 1

    print(f"\n📦 BC-RNN checkpoint: {agent_path}")
    print(f"🌪  Noise levels:      {noise_levels}")
    print(f"🔧  Methods:           {methods}\n")

    policy, env = load_policy_and_environment(agent_path, render_gpu_id=args.render_gpu_id)
    action_dim  = get_action_dimension(env, policy)
    print(f"🔍 Action dimension: {action_dim}")

    # -------------------------
    # Evaluation loop
    # -------------------------
    all_results = []

    for noise_std in noise_levels:
        for method in methods:
            # Parse t_start from method label if diffusion
            base_method = method.split("[")[0]
            t_start = 40  # default
            if base_method == "diffusion" and "[t=" in method:
                t_start = int(method.split("[t=")[1].rstrip("]"))

            print(f"\n🔄 noise_std={noise_std} | method={method}")

            result = evaluate_noise_and_method(
                policy     = policy,
                env        = env,
                noise_std  = noise_std,
                method     = method,
                action_dim = action_dim,
                n_rollouts = args.n_rollouts,
                horizon    = args.horizon,
                obs_keys   = obs_keys,
                t_start    = t_start,
            )
            all_results.append(result)

    # -------------------------
    # Save and print
    # -------------------------
    print_results_summary(all_results)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        "epoch":          args.epoch,
        "n_rollouts":     args.n_rollouts,
        "horizon":        args.horizon,
        "seed":           args.seed,
        "noise_levels":   noise_levels,
        "methods":        methods,
        "diffusion_model": args.diffusion_model,
        "t_start":        args.t_start,
    }

    save_results_to_json(all_results, config, timestamp)
    save_results_to_csv(all_results, timestamp)

    return 0


if __name__ == "__main__":
    sys.exit(main())