"""
Collect clean (obs, action) trajectories from a trained robomimic policy
for diffusion model training.

Task-agnostic: works for Lift, Can, Square, or any robomimic task.
Dims are auto-detected from the checkpoint via shape_metadata.

Supports H=1 (single step) and H>1 (sliding window) via --window_size.

Output .npz file:
    H=1:  states [N, 1, Ds], actions [N, 1, Da]  — one sample per step
    H>1:  states [N, H, Ds], actions [N, H, Da]  — one sample per window
          where states[:,0,:] is the conditioning obs (window start)
          and actions[:,0:H,:] are H consecutive clean actions

Windows are built per-episode with stride=1 (fully overlapping).
Windows never cross episode boundaries.

Examples:
    # H=1 (original behaviour)
    python -m diffusion.collect_diffusion_data \\
        --checkpoint checkpoints/bc_rnn_lift/.../model_epoch_600.pth \\
        --task lift --n_episodes 200

    # H=5 sliding window
    python -m diffusion.collect_diffusion_data \\
        --checkpoint checkpoints/bc_rnn_lift/.../model_epoch_600.pth \\
        --task lift --n_episodes 200 --window_size 5 \\
        --output_path diffusion_data/lift_diffusion_data_H5.npz

    # H=10 (matches BC-RNN sequence length)
    python -m diffusion.collect_diffusion_data \\
        --checkpoint checkpoints/bc_rnn_lift/.../model_epoch_600.pth \\
        --task lift --n_episodes 200 --window_size 10 \\
        --output_path diffusion_data/lift_diffusion_data_H10.npz
"""

import os
import sys
import argparse
import numpy as np


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
    current   = os.environ.get('LD_LIBRARY_PATH', '')
    new_paths = [p for p in paths if os.path.exists(p)]
    os.environ['LD_LIBRARY_PATH'] = ":".join(new_paths + [current])
    print("🔧 LD_LIBRARY_PATH configured")


# =========================
# ARGUMENT PARSER
# =========================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Collect diffusion training data from a trained robomimic policy"
    )
    parser.add_argument(
        '--checkpoint', type=str, required=True,
        help='Path to trained robomimic checkpoint (.pth)'
    )
    parser.add_argument(
        '--task', type=str, required=True,
        help='Task name for bookkeeping (e.g. lift, can, square)'
    )
    parser.add_argument(
        '--n_episodes', type=int, default=200,
        help='Number of successful episodes to collect (default: 200)'
    )
    parser.add_argument(
        '--horizon', type=int, default=400,
        help='Max steps per episode (default: 400)'
    )
    parser.add_argument(
        '--window_size', type=int, default=1,
        help='Sliding window size H. '
             'H=1: one sample per step (original). '
             'H>1: overlapping windows of H consecutive steps. '
             'Default: 1'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Path to save .npz file. '
             'Defaults to diffusion_data/<task>_diffusion_data_H<window_size>.npz'
    )
    parser.add_argument(
        '--only_successful', action='store_true', default=True,
        help='Only keep episodes where task succeeds (default: True)'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed'
    )
    return parser.parse_args()


# =========================
# POLICY AND ENV LOADING
# =========================

def load_policy_and_env(checkpoint_path):
    """
    Load robomimic policy and environment from checkpoint.

    Returns:
        policy:    callable, takes obs dict, returns action np.array [Da]
        env:       robomimic env
        ckpt_dict: raw checkpoint dict (contains shape_metadata)
    """
    import torch
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device: {device}")

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=checkpoint_path,
        device=device,
        verbose=False
    )
    print("✅ Policy loaded")

    env_meta = ckpt_dict["env_metadata"]
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    print("✅ Environment created")

    return policy, env, ckpt_dict


# =========================
# SINGLE EPISODE COLLECTION
# =========================

def collect_episode(policy, env, obs_keys, horizon):
    """
    Roll out the policy for one episode, collecting (obs, action) at every step.
    No windowing here — returns raw per-step arrays.

    Args:
        policy:   robomimic policy
        env:      robomimic env
        obs_keys: list of obs keys to flatten (from get_task_dims)
        horizon:  max steps per episode

    Returns:
        ep_states:  np.array [T, state_dim] — flattened obs at each step
        ep_actions: np.array [T, action_dim] — clean actions at each step
        success:    bool
    """
    from diffusion.model import flatten_obs

    obs = env.reset()
    policy.start_episode()

    ep_states  = []
    ep_actions = []
    success    = False

    for step in range(horizon):
        state_vec = flatten_obs(obs, obs_keys)
        action    = policy(obs)

        ep_states.append(state_vec)
        ep_actions.append(action.copy())

        obs, reward, done, _ = env.step(action)

        if env.is_success()["task"]:
            success = True
            break

        if done:
            break

    ep_states  = np.array(ep_states,  dtype=np.float32)  # [T, Ds]
    ep_actions = np.array(ep_actions, dtype=np.float32)  # [T, Da]

    return ep_states, ep_actions, success


# =========================
# WINDOW BUILDER
# =========================

def build_windows(ep_states, ep_actions, window_size):
    """
    Convert a single episode's per-step arrays into overlapping windows.
    Windows never cross episode boundaries.

    For each valid start index i (0 <= i <= T-H):
        state window:  ep_states[i : i+H]   — H obs vectors
        action window: ep_actions[i : i+H]  — H consecutive actions

    The conditioning signal at inference is states[0] (window start obs).

    Args:
        ep_states:   np.array [T, Ds]
        ep_actions:  np.array [T, Da]
        window_size: int H

    Returns:
        windows_states:  np.array [N_windows, H, Ds]  or None if T < H
        windows_actions: np.array [N_windows, H, Da]  or None if T < H
    """
    T = ep_states.shape[0]

    if T < window_size:
        return None, None

    windows_states  = []
    windows_actions = []

    for i in range(T - window_size + 1):
        windows_states.append(ep_states[i : i + window_size])    # [H, Ds]
        windows_actions.append(ep_actions[i : i + window_size])  # [H, Da]

    windows_states  = np.array(windows_states,  dtype=np.float32)  # [N_w, H, Ds]
    windows_actions = np.array(windows_actions, dtype=np.float32)  # [N_w, H, Da]

    return windows_states, windows_actions


# =========================
# MAIN COLLECTION LOOP
# =========================

def main():
    args = parse_arguments()

    np.random.seed(args.seed)
    setup_mujoco()

    H = args.window_size

    # Default output path includes H so files don't collide across window sizes
    if args.output_path is None:
        os.makedirs("diffusion_data", exist_ok=True)
        args.output_path = f"diffusion_data/{args.task}_diffusion_data_H{H}.npz"

    # Load policy and environment
    policy, env, ckpt_dict = load_policy_and_env(args.checkpoint)

    # Auto-detect dims from checkpoint — works for any robomimic task
    from diffusion.model import get_task_dims
    obs_keys, state_dim, action_dim = get_task_dims(ckpt_dict)

    print(f"\n📐 Task:            {args.task}")
    print(f"📐 obs_keys:        {obs_keys}")
    print(f"📐 state_dim:       {state_dim}")
    print(f"📐 action_dim:      {action_dim}")
    print(f"📐 window_size H:   {H}")
    print(f"📐 Episodes:        {args.n_episodes}")
    print(f"📐 Horizon:         {args.horizon}")
    print(f"📐 Only successful: {args.only_successful}")
    print(f"📐 Output:          {args.output_path}\n")

    all_states  = []  # each entry: [N_windows, H, Ds]
    all_actions = []  # each entry: [N_windows, H, Da]

    n_collected   = 0
    n_success     = 0
    n_failed      = 0
    n_too_short   = 0
    total_windows = 0
    ep            = 0

    while n_collected < args.n_episodes:
        ep += 1

        ep_states, ep_actions, success = collect_episode(
            policy, env, obs_keys, args.horizon
        )

        # Skip failed episodes if requested
        if args.only_successful and not success:
            n_failed += 1
            continue

        # Build windows for this episode
        if H == 1:
            # H=1: each step is its own window
            T           = ep_states.shape[0]
            win_states  = ep_states.reshape(T, 1, state_dim)
            win_actions = ep_actions.reshape(T, 1, action_dim)
        else:
            # H>1: overlapping sliding windows, stride=1
            win_states, win_actions = build_windows(ep_states, ep_actions, H)

            if win_states is None:
                # Episode shorter than window size — skip
                n_too_short += 1
                continue

        all_states.append(win_states)
        all_actions.append(win_actions)

        n_collected   += 1
        total_windows += win_states.shape[0]

        if success:
            n_success += 1

        if n_collected % 20 == 0 or n_collected == args.n_episodes:
            print(f"  Collected {n_collected:4d}/{args.n_episodes} episodes | "
                  f"success={n_success} | failed={n_failed} | "
                  f"too_short={n_too_short} | "
                  f"windows_so_far={total_windows}")

    # Concatenate across all episodes: [N_total_windows, H, dim]
    states  = np.concatenate(all_states,  axis=0)  # [N, H, Ds]
    actions = np.concatenate(all_actions, axis=0)  # [N, H, Da]

    N, H_check, _ = states.shape
    assert H_check == H, f"Window size mismatch: {H_check} vs {H}"

    print(f"\n✅ Collection complete")
    print(f"   states:          {states.shape}  [N, H, Ds]")
    print(f"   actions:         {actions.shape}  [N, H, Da]")
    print(f"   Total windows:   {N}")
    print(f"   Avg windows/ep:  {N / n_collected:.1f}")

    # Save — window_size saved explicitly so train script reads correct H
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    np.savez(
        args.output_path,
        states      = states,
        actions     = actions,
        obs_keys    = np.array(obs_keys),
        state_dim   = state_dim,
        action_dim  = action_dim,
        task        = args.task,
        window_size = H,
    )
    print(f"💾 Saved to: {args.output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())