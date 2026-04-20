"""
Collect clean (obs, cond) trajectories from a trained robomimic policy
for diffusion model training.

Task-agnostic: works for Lift, Can, Square, or any robomimic task.
Dims are auto-detected from the checkpoint via shape_metadata.

Conditioning modes (--cond_mode):
    state:        cond = flatten_obs()           [Ds]   — original behaviour
    vision:       cond = encode_image()          [512]  — ResNet-18 features
    state+vision: cond = concat(state, vision)   [Ds+512]

Supports H=1 (single step) and H>1 (sliding window) via --window_size.

Output .npz file:
    conds   [N, H, Dc]  — conditioning vectors (Dc depends on cond_mode)
    actions [N, H, Da]  — clean actions
    cond_mode, cond_dim, action_dim, obs_keys, task, window_size saved as metadata

Examples:
    # State conditioning (original)
    python -m diffusion.collect_diffusion_data \\
        --checkpoint checkpoints/bc_rnn_square/.../model_epoch_2000.pth \\
        --task square --n_episodes 200 --cond_mode state

    # Vision conditioning
    python -m diffusion.collect_diffusion_data \\
        --checkpoint checkpoints/bc_rnn_square/.../model_epoch_2000.pth \\
        --task square --n_episodes 200 --cond_mode vision \\
        --output_path diffusion_data/square_diffusion_data_H1_vision.npz

    # State + Vision
    python -m diffusion.collect_diffusion_data \\
        --checkpoint checkpoints/bc_rnn_square/.../model_epoch_2000.pth \\
        --task square --n_episodes 200 --cond_mode state+vision \\
        --output_path diffusion_data/square_diffusion_data_H1_statevision.npz
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
        '/usr/lib/nvidia',
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
        '--cond_mode', type=str, default='state',
        choices=['state', 'vision', 'state+vision'],
        help='Conditioning mode: state | vision | state+vision (default: state)'
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
        help='Sliding window size H (default: 1)'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Path to save .npz file. '
             'Defaults to diffusion_data/<task>_diffusion_data_H<H>_<cond_mode>.npz'
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

def load_policy_and_env(checkpoint_path, cond_mode):
    """
    Load robomimic policy and environment from checkpoint.
    Enables camera obs when cond_mode requires vision.

    Returns:
        policy:    robomimic RolloutPolicy
        env:       robomimic EnvRobosuite
        ckpt_dict: raw checkpoint dict
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

    # Enable camera obs when vision conditioning is needed
    needs_vision = cond_mode in ('vision', 'state+vision')
    if needs_vision:
        env_meta['env_kwargs']['camera_names']          = ['agentview']
        env_meta['env_kwargs']['camera_heights']        = 84
        env_meta['env_kwargs']['camera_widths']         = 84
        env_meta['env_kwargs']['use_camera_obs']        = True
        env_meta['env_kwargs']['has_offscreen_renderer'] = True
        print("📷 Camera obs enabled (agentview 84x84)")

    env = EnvUtils.create_env_from_metadata(
        env_meta       = env_meta,
        render         = False,
        render_offscreen = needs_vision,
        use_image_obs  = needs_vision,
    )
    print("✅ Environment created")

    return policy, env, ckpt_dict


def ensure_image_obs(obs, env):
    """
    Ensure observation dict contains at least one '*_image' key.

    Some robomimic env wrappers may not include camera frames in obs even when
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
            "Vision conditioning requested, but no image key was found and "
            "offscreen render returned None."
        )

    obs = dict(obs)
    obs['agentview_image'] = frame.astype(np.uint8)
    return obs


# =========================
# SINGLE EPISODE COLLECTION
# =========================

def collect_episode(policy, env, obs_keys, cond_mode, horizon):
    """
    Roll out the policy for one episode, collecting (cond, action) at every step.

    Args:
        policy:    robomimic policy
        env:       robomimic env
        obs_keys:  list of state obs keys (from get_task_dims)
        cond_mode: 'state' | 'vision' | 'state+vision'
        horizon:   max steps per episode

    Returns:
        ep_conds:   np.array [T, Dc] — conditioning vectors at each step
        ep_actions: np.array [T, Da] — clean actions at each step
        success:    bool
    """
    from diffusion.model import build_cond_vec

    obs = env.reset()
    policy.start_episode()

    ep_conds   = []
    ep_actions = []
    success    = False

    for step in range(horizon):
        if cond_mode in ('vision', 'state+vision'):
            obs = ensure_image_obs(obs, env)

        # Build conditioning vector — handles all three modes
        cond_vec = build_cond_vec(obs, obs_keys, cond_mode)

        # Get clean action from policy
        action = policy(ob=obs, goal=None)

        ep_conds.append(cond_vec)
        ep_actions.append(action.copy())

        obs, reward, done, _ = env.step(action)

        if env.is_success()["task"]:
            success = True
            break

        if done:
            break

    ep_conds   = np.array(ep_conds,   dtype=np.float32)  # [T, Dc]
    ep_actions = np.array(ep_actions, dtype=np.float32)  # [T, Da]

    return ep_conds, ep_actions, success


# =========================
# WINDOW BUILDER
# =========================

def build_windows(ep_conds, ep_actions, window_size):
    """
    Convert a single episode into overlapping windows.
    Windows never cross episode boundaries.

    Returns:
        windows_conds:   np.array [N_windows, H, Dc] or None
        windows_actions: np.array [N_windows, H, Da] or None
    """
    T = ep_conds.shape[0]

    if T < window_size:
        return None, None

    windows_conds   = []
    windows_actions = []

    for i in range(T - window_size + 1):
        windows_conds.append(ep_conds[i : i + window_size])
        windows_actions.append(ep_actions[i : i + window_size])

    windows_conds   = np.array(windows_conds,   dtype=np.float32)
    windows_actions = np.array(windows_actions, dtype=np.float32)

    return windows_conds, windows_actions


# =========================
# MAIN COLLECTION LOOP
# =========================

def main():
    args = parse_arguments()

    np.random.seed(args.seed)
    setup_mujoco()

    H         = args.window_size
    cond_mode = args.cond_mode

    # Default output path encodes H and cond_mode so files don't collide
    if args.output_path is None:
        os.makedirs("diffusion_data", exist_ok=True)
        cond_suffix = cond_mode.replace('+', '_plus_')
        args.output_path = (
            f"diffusion_data/{args.task}_diffusion_data"
            f"_H{H}_{cond_suffix}.npz"
        )

    # Load policy and environment
    policy, env, ckpt_dict = load_policy_and_env(args.checkpoint, cond_mode)

    # Auto-detect dims from checkpoint
    from diffusion.model import get_task_dims, get_cond_dim
    obs_keys, state_dim, action_dim = get_task_dims(ckpt_dict)
    cond_dim = get_cond_dim(cond_mode, state_dim)

    print(f"\n📐 Task:            {args.task}")
    print(f"📐 cond_mode:       {cond_mode}")
    print(f"📐 obs_keys:        {obs_keys}")
    print(f"📐 state_dim:       {state_dim}")
    print(f"📐 cond_dim:        {cond_dim}")
    print(f"📐 action_dim:      {action_dim}")
    print(f"📐 window_size H:   {H}")
    print(f"📐 Episodes:        {args.n_episodes}")
    print(f"📐 Horizon:         {args.horizon}")
    print(f"📐 Only successful: {args.only_successful}")
    print(f"📐 Output:          {args.output_path}\n")

    all_conds   = []
    all_actions = []

    n_collected   = 0
    n_success     = 0
    n_failed      = 0
    n_too_short   = 0
    total_windows = 0

    while n_collected < args.n_episodes:

        ep_conds, ep_actions, success = collect_episode(
            policy, env, obs_keys, cond_mode, args.horizon
        )

        if args.only_successful and not success:
            n_failed += 1
            continue

        # Build windows
        if H == 1:
            T           = ep_conds.shape[0]
            win_conds   = ep_conds.reshape(T, 1, cond_dim)
            win_actions = ep_actions.reshape(T, 1, action_dim)
        else:
            win_conds, win_actions = build_windows(ep_conds, ep_actions, H)
            if win_conds is None:
                n_too_short += 1
                continue

        all_conds.append(win_conds)
        all_actions.append(win_actions)

        n_collected   += 1
        total_windows += win_conds.shape[0]

        if success:
            n_success += 1

        if n_collected % 20 == 0 or n_collected == args.n_episodes:
            print(f"  Collected {n_collected:4d}/{args.n_episodes} episodes | "
                  f"success={n_success} | failed={n_failed} | "
                  f"too_short={n_too_short} | windows={total_windows}")

    # Concatenate
    conds   = np.concatenate(all_conds,   axis=0)  # [N, H, Dc]
    actions = np.concatenate(all_actions, axis=0)  # [N, H, Da]

    N, H_check, Dc = conds.shape
    assert H_check == H,   f"Window size mismatch: {H_check} vs {H}"
    assert Dc == cond_dim, f"cond_dim mismatch: {Dc} vs {cond_dim}"

    print(f"\n✅ Collection complete")
    print(f"   conds:           {conds.shape}   [N, H, Dc]")
    print(f"   actions:         {actions.shape}  [N, H, Da]")
    print(f"   Total windows:   {N}")
    print(f"   Avg windows/ep:  {N / n_collected:.1f}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    np.savez(
        args.output_path,
        conds       = conds,
        actions     = actions,
        obs_keys    = np.array(obs_keys),
        state_dim   = state_dim,
        cond_dim    = cond_dim,
        action_dim  = action_dim,
        cond_mode   = cond_mode,
        task        = args.task,
        window_size = H,
    )
    print(f"💾 Saved to: {args.output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())