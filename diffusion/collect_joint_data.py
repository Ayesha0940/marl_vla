"""
Collect on-policy (state, action) trajectories from a trained BC-RNN policy
and save as a robomimic-format HDF5 for joint denoiser training.

WHY: The joint denoiser trained on human demo HDF5 learns to recover the demo
action distribution, not the policy's action distribution. These are different
manifolds — the denoiser at deployment is projecting onto the wrong one.
Fix: collect rollouts from the trained policy and train the denoiser on those.

Output HDF5 format matches robomimic exactly, so JointDenoiserDataset reads it
directly without any changes.

Usage:
    python -m diffusion.collect_joint_data \\
        --checkpoint checkpoints/bc_rnn_lift/bc_rnn_lift/20260405174006/models/model_epoch_600.pth \\
        --output_path diffusion_data/lift_policy_rollouts.hdf5 \\
        --n_episodes 200

Then train:
    python -m diffusion.train_joint_denoiser \\
        --bc_rnn_ckpt checkpoints/bc_rnn_lift/bc_rnn_lift/20260405174006/models/model_epoch_600.pth \\
        --hdf5_path   diffusion_data/lift_policy_rollouts.hdf5 \\
        --anchor      A1 \\
        --output_path diffusion_models/joint_a1_lift.pt
"""

import argparse
import os
import sys

import h5py
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Obs keys to save in HDF5 beyond the policy obs_keys.
# These are needed by anchor variants (A3 proprio) even if the policy doesn't use them.
_EXTRA_OBS_KEYS = ["robot0_joint_pos", "robot0_joint_vel"]


def _setup_mujoco():
    paths = [
        os.path.expanduser("~/.mujoco/mujoco210/bin"),
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu/nvidia",
        "/usr/lib/nvidia",
    ]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    for p in paths:
        if os.path.exists(p) and p not in current.split(":"):
            current = f"{p}:{current}" if current else p
    os.environ["LD_LIBRARY_PATH"] = current


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to trained BC-RNN checkpoint (.pth)")
    p.add_argument("--output_path", type=str,
                   default="diffusion_data/lift_policy_rollouts.hdf5",
                   help="Output HDF5 path. Default: diffusion_data/lift_policy_rollouts.hdf5")
    p.add_argument("--n_episodes", type=int, default=200,
                   help="Number of episodes to collect. Default: 200")
    p.add_argument("--only_successful", action="store_true", default=True,
                   help="Only keep successful episodes (default True)")
    p.add_argument("--episode_horizon", type=int, default=400,
                   help="Max steps per episode. Default: 400")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def _load_policy_and_env(checkpoint_path: str, device_str: str):
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
            torch.cuda.empty_cache()
            device = torch.device("cpu")
            policy, ckpt_dict = FileUtils.policy_from_checkpoint(
                ckpt_path=checkpoint_path, device=device, verbose=False
            )
        else:
            raise

    env = EnvUtils.create_env_from_metadata(
        env_meta=ckpt_dict["env_metadata"],
        render=False, render_offscreen=False, use_image_obs=False,
    )
    return policy, env, ckpt_dict


def _collect_episode(policy, env, obs_keys_to_save: list, episode_horizon: int):
    """
    Roll out policy for one episode.

    Returns:
        ep: dict of np.arrays — 'actions' (T, Da) and each obs key (T, dim)
        success: bool
    """
    obs = env.reset()
    policy.start_episode()

    # Check which extra keys the env actually provides
    available_keys = [k for k in obs_keys_to_save if k in obs]

    buffers = {k: [] for k in available_keys}
    actions = []
    success = False

    for _ in range(episode_horizon):
        ac = policy(ob=obs, goal=None)
        if hasattr(ac, "cpu"):
            ac = ac.cpu().numpy()
        ac = np.asarray(ac).flatten().astype(np.float32)

        for k in available_keys:
            buffers[k].append(np.asarray(obs[k], dtype=np.float32).flatten().copy())
        actions.append(ac.copy())

        obs, _, done, _ = env.step(ac)

        if env.is_success()["task"]:
            success = True
            break
        if done:
            break

    ep = {"actions": np.array(actions, dtype=np.float32)}
    for k in available_keys:
        ep[k] = np.array(buffers[k], dtype=np.float32)

    return ep, success


def main():
    args = parse_args()
    _setup_mujoco()
    np.random.seed(args.seed)

    print(f"Checkpoint:    {args.checkpoint}")
    print(f"Output:        {args.output_path}")
    print(f"Episodes:      {args.n_episodes}")
    print(f"Only success:  {args.only_successful}")

    policy, env, ckpt_dict = _load_policy_and_env(args.checkpoint, args.device)

    sm = ckpt_dict["shape_metadata"]
    policy_obs_keys = list(sm["all_obs_keys"])
    obs_keys_to_save = list(dict.fromkeys(policy_obs_keys + _EXTRA_OBS_KEYS))

    print(f"Policy obs keys: {policy_obs_keys}")
    print(f"Keys to save:    {obs_keys_to_save}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    n_collected = 0
    n_failed    = 0
    total_steps = 0

    with h5py.File(args.output_path, "w") as f:
        grp = f.create_group("data")

        while n_collected < args.n_episodes:
            ep, success = _collect_episode(
                policy, env, obs_keys_to_save, args.episode_horizon
            )

            if args.only_successful and not success:
                n_failed += 1
                continue

            demo_key = f"demo_{n_collected}"
            dg = grp.create_group(demo_key)
            dg.create_dataset("actions", data=ep["actions"])

            obs_grp = dg.create_group("obs")
            for k in obs_keys_to_save:
                if k in ep:
                    obs_grp.create_dataset(k, data=ep[k])

            n_collected += 1
            total_steps += ep["actions"].shape[0]

            if n_collected % 20 == 0 or n_collected == args.n_episodes:
                print(f"  Collected {n_collected:4d}/{args.n_episodes} | "
                      f"failed={n_failed} | total_steps={total_steps}")

    print(f"\nSaved {n_collected} episodes ({total_steps} steps) to: {args.output_path}")


if __name__ == "__main__":
    main()
