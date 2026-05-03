"""Generate configuration for the custom Lift diffusion policy."""

import os
import json

# Get the project root directory (parent of training/ directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Diffusion policy config (custom format, not robomimic)
cfg = {
    "algo_name": "diffusion",
    "task": "lift",
    "experiment": {
        "name": "lift_diffusion_policy",
    },
    "train": {
        "data": os.path.join(PROJECT_ROOT, "datasets/lift/ph/low_dim_v141.hdf5"),
        "output_dir": os.path.join(PROJECT_ROOT, "checkpoints/lift_diffusion_policy"),
        "num_epochs": 500,
        "batch_size": 256,
        "num_workers": 2,
        "log_freq": 10,
    },
    "model": {
        "obs_keys": [
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
            "object",
        ],
        "obs_horizon": 16,
        "action_horizon": 8,
        "hidden_dim": 512,
        "time_emb_dim": 128,
    },
    "diffusion": {
        "num_steps": 100,
        "beta_start": 1e-4,
        "beta_end": 2e-2,
    },
    "optimizer": {
        "lr": 1e-4,
        "weight_decay": 1e-5,
    },
}

config_dir = os.path.join(PROJECT_ROOT, 'configs')
os.makedirs(config_dir, exist_ok=True)
config_file = os.path.join(config_dir, 'diffusion_lift.json')
with open(config_file, 'w') as f:
    json.dump(cfg, f, indent=4)
print(f'✅ Config saved to {config_file}')
