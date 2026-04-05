import os
import numpy as np
import torch
from tqdm import trange

import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils


# =========================
# CONFIG
# =========================
CKPT = "/home/axs0940/miniconda3/envs/vla_marl/lib/python3.10/site-packages/robomimic/checkpoints/bc_baseline/transport_bc_baseline/20260404203751/models/model_epoch_100.pth"

DATASET = "datasets/transport/ph/low_dim_v141.hdf5"

N_EP = 50
MAX_STEPS = 700

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# LOAD POLICY
# =========================
policy, _ = FileUtils.policy_from_checkpoint(
    ckpt_path=CKPT,
    device=device,
    verbose=True
)
print("✅ Policy loaded")


# =========================
# CREATE ENV
# =========================
env_meta = FileUtils.get_env_metadata_from_dataset(DATASET)

env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    render=False,
    render_offscreen=False,
    use_image_obs=False,
)

print("✅ Environment created")


# =========================
# EVALUATION
# =========================
rewards = []
successes = []

for ep in trange(N_EP, desc="Evaluating BC"):

    obs = env.reset()
    policy.start_episode()

    ep_reward = 0.0
    success = False

    for step in range(MAX_STEPS):

        # 🔥 CORRECT: pass raw obs directly
        action = policy(obs)

        # Clip action
        action = np.clip(action, -1.0, 1.0)

        obs, reward, done, _ = env.step(action)
        ep_reward += reward

        # Success check
        if env.is_success()["task"]:
            success = True
            break

        if done:
            break
    print(f"Episode {ep+1}/{N_EP} — Reward: {ep_reward:.4f} — Success: {success}")
    rewards.append(ep_reward)
    successes.append(int(success))


# =========================
# RESULTS
# =========================
print("\n" + "=" * 60)
print(f"BC Baseline — {N_EP} Episodes")
print("-" * 60)
print(f"Mean Reward:    {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
print(f"Success Rate:   {np.mean(successes) * 100:.2f}%")
print(f"Max Reward:     {np.max(rewards):.4f}")
print("=" * 60)


# =========================
# SAVE
# =========================
os.makedirs(os.path.expanduser("~/marl_vla/results"), exist_ok=True)

np.save(
    os.path.expanduser("~/marl_vla/results/bc_baseline_eval.npy"),
    {
        "rewards": rewards,
        "successes": successes,
    },
)

print("✅ Results saved")