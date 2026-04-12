#!/usr/bin/env python

import os
import sys
import glob
import argparse
import json
from datetime import datetime

# =========================
# PATH SETUP
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints/bc_rnn_square/bc_rnn_square')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
VIDEO_DIR = os.path.join(RESULTS_DIR, "videos")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(VIDEO_DIR, exist_ok=True)

# =========================
# MUJOCO SETUP
# =========================
def setup_mujoco():
    paths = [
        os.path.expanduser('~/.mujoco/mujoco210/bin'),
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib/x86_64-linux-gnu/nvidia',
    ]

    current = os.environ.get('LD_LIBRARY_PATH', '')
    new_paths = [p for p in paths if os.path.exists(p)]
    os.environ['LD_LIBRARY_PATH'] = ":".join(new_paths + [current])

    print("🔧 LD_LIBRARY_PATH set")


# =========================
# CHECKPOINT
# =========================
def find_latest_checkpoint(epoch=1000):
    pattern = os.path.join(CHECKPOINT_DIR, f'*/models/model_epoch_{epoch}.pth')
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=1000)
    parser.add_argument('--n_rollouts', type=int, default=10)
    parser.add_argument('--horizon', type=int, default=500)
    args = parser.parse_args()

    noise_levels = [0.1, 0.2, 0.5, 1.0]

    setup_mujoco()

    agent_path = find_latest_checkpoint(args.epoch)
    if not agent_path:
        print("❌ No checkpoint found")
        return

    print(f"\n📦 Using checkpoint: {agent_path}")

    # =========================
    # IMPORTS (after setup)
    # =========================
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import numpy as np
    import torch
    import imageio

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=agent_path,
        device=device,
        verbose=False
    )

    print("✅ Policy loaded")

    env_meta = ckpt_dict["env_metadata"]

    env_meta["env_kwargs"]["camera_names"] = ["agentview"]
    env_meta["env_kwargs"]["camera_heights"] = 256
    env_meta["env_kwargs"]["camera_widths"] = 256

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )

    print("✅ Environment created")

    # =========================
    # LOOP OVER NOISE
    # =========================
    for noise_std in noise_levels:
        print(f"\n🌪 Noise = {noise_std}")

        for ep in range(args.n_rollouts):

            obs = env.reset()
            policy.start_episode()

            record_video = (ep == 0)
            frames = []

            for step in range(args.horizon):

                action = policy(obs)

                # 🔥 ADD NOISE
                if noise_std > 0:
                    noise = np.random.normal(0, noise_std, size=action.shape)
                    action = action + noise

                action = np.clip(action, -1.0, 1.0)

                obs, reward, done, _ = env.step(action)

                # 🎥 RECORD FRAME
                if record_video:
                    try:
                        frame = env.env.sim.render(
                            width=256,
                            height=256,
                            camera_name="agentview"
                        )
                        if frame is not None:
                            frames.append(frame)
                        else:
                            print("⚠️ Got None frame")
                    except Exception as e:
                        print("⚠️ Render error:", e)

                if env.is_success()["task"]:
                    break

                if done:
                    break

            # =========================
            # SAVE VIDEO
            # =========================
            if record_video:
                print(f"📊 Frames collected: {len(frames)}")

                if len(frames) > 0:
                    video_path = os.path.join(
                        VIDEO_DIR,
                        f"noise_{noise_std:.2f}.mp4"
                    )

                    imageio.mimsave(video_path, frames, fps=20)
                    print(f"🎥 Saved: {video_path}")
                else:
                    print("❌ No frames → video not saved")

    print("\n✅ DONE")


if __name__ == "__main__":
    main()