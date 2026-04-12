"""
Collect clean (obs, action) trajectories from a trained robomimic policy
for diffusion model training.

Task-agnostic: works for Lift, Can, Square, or any robomimic task.
Dims are auto-detected from the checkpoint via shape_metadata.

Output: .npz file with:
    states:   [N, 1, state_dim]   — flattened obs at each step
    actions:  [N, 1, action_dim]  — clean action at each step

Each (state, action) pair is a single timestep (H=1).
N = total steps collected across all episodes.

Examples:
    python collect_diffusion_data.py \\
        --checkpoint /path/to/model_epoch_600.pth \\
        --task lift \\
        --n_episodes 200 \\
        --output_path /path/to/lift_diffusion_data.npz

    python collect_diffusion_data.py \\
        --checkpoint /path/to/model_epoch_600.pth \\
        --task can \\
        --n_episodes 200 \\
        --output_path /path/to/can_diffusion_data.npz
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
    current = os.environ.get('LD_LIBRARY_PATH', '')
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
        help='Number of episodes to collect'
    )
    parser.add_argument(
        '--horizon', type=int, default=400,
        help='Max steps per episode'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Path to save .npz file. Defaults to diffusion_data/<task>_diffusion_data.npz'
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
        policy:    callable, takes obs dict, returns action np.array
        env:       robomimic env
        ckpt_dict: raw checkpoint dict (for shape_metadata)
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
    Roll out the policy for one episode and collect (obs, action) pairs.

    Args:
        policy:   robomimic policy
        env:      robomimic env
        obs_keys: list of obs keys to flatten (from get_task_dims)
        horizon:  max steps

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
        # Flatten obs to vector
        state_vec = flatten_obs(obs, obs_keys)

        # Get clean action from policy (no noise)
        action = policy(obs)

        ep_states.append(state_vec)
        ep_actions.append(action.copy())

        # Step environment
        obs, reward, done, _ = env.step(action)

        # Check success
        if env.is_success()["task"]:
            success = True
            break

        if done:
            break

    ep_states  = np.array(ep_states,  dtype=np.float32)  # [T, state_dim]
    ep_actions = np.array(ep_actions, dtype=np.float32)  # [T, action_dim]

    return ep_states, ep_actions, success


# =========================
# MAIN COLLECTION LOOP
# =========================

def main():
    args = parse_arguments()

    # Seed
    np.random.seed(args.seed)

    # MuJoCo setup
    setup_mujoco()

    # Default output path
    if args.output_path is None:
        os.makedirs("diffusion_data", exist_ok=True)
        args.output_path = f"diffusion_data/{args.task}_diffusion_data.npz"

    # Load policy and environment
    policy, env, ckpt_dict = load_policy_and_env(args.checkpoint)

    # Auto-detect dims from checkpoint
    from diffusion.model import get_task_dims
    obs_keys, state_dim, action_dim = get_task_dims(ckpt_dict)

    print(f"\n📐 Task:       {args.task}")
    print(f"📐 obs_keys:   {obs_keys}")
    print(f"📐 state_dim:  {state_dim}")
    print(f"📐 action_dim: {action_dim}")
    print(f"📐 Episodes:   {args.n_episodes}")
    print(f"📐 Horizon:    {args.horizon}")
    print(f"📐 Only successful: {args.only_successful}")
    print(f"📐 Output:     {args.output_path}\n")

    # Collection loop
    all_states  = []  # each entry: [T, state_dim]
    all_actions = []  # each entry: [T, action_dim]

    n_collected = 0
    n_success   = 0
    n_failed    = 0
    ep          = 0

    while n_collected < args.n_episodes:
        ep += 1
        ep_states, ep_actions, success = collect_episode(
            policy, env, obs_keys, args.horizon
        )

        if args.only_successful and not success:
            n_failed += 1
            if ep % 20 == 0:
                print(f"  Episode {ep:4d} | FAILED (skipped) | "
                      f"collected={n_collected}/{args.n_episodes} | "
                      f"failed={n_failed}")
            continue

        # Each timestep becomes one training sample (H=1)
        # Reshape to [T, 1, dim] so shape is consistent with MPE format
        T = ep_states.shape[0]
        all_states.append(ep_states.reshape(T, 1, state_dim))    # [T, 1, Ds]
        all_actions.append(ep_actions.reshape(T, 1, action_dim)) # [T, 1, Da]

        n_collected += 1
        if success:
            n_success += 1

        if n_collected % 20 == 0 or n_collected == args.n_episodes:
            print(f"  Collected {n_collected:4d}/{args.n_episodes} episodes | "
                  f"success={n_success} | failed={n_failed} | "
                  f"steps_so_far={sum(s.shape[0] for s in all_states)}")

    # Concatenate all timesteps: [N_total_steps, 1, dim]
    states  = np.concatenate(all_states,  axis=0)  # [N, 1, Ds]
    actions = np.concatenate(all_actions, axis=0)  # [N, 1, Da]

    print(f"\n✅ Collection complete")
    print(f"   states:  {states.shape}")
    print(f"   actions: {actions.shape}")
    print(f"   Total timesteps: {states.shape[0]}")

    # Save
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    np.savez(
        args.output_path,
        states=states,
        actions=actions,
        obs_keys=np.array(obs_keys),   # save for reference
        state_dim=state_dim,
        action_dim=action_dim,
        task=args.task,
    )
    print(f"💾 Saved to: {args.output_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())