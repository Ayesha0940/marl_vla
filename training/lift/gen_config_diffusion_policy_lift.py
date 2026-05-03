import os, json, sys
from robomimic.config import config_factory

# Get the project root directory (parent of training/ directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SUPPORTED_ALGOS = ['bc', 'bcq', 'cql', 'iql', 'gl', 'hbc', 'iris', 'td3_bc']

try:
    # Use robomimic's native diffusion_policy algorithm when available
    cfg = config_factory('diffusion_policy')
except Exception:
    print(
        'This robomimic installation does not include diffusion_policy.\n'
        'Installed version: 0.3.0\n'
        f'Supported algos: {", ".join(SUPPORTED_ALGOS)}\n\n'
        'To use a native robomimic diffusion policy, install a robomimic build '
        'that registers diffusion_policy. Otherwise, use the custom diffusion '
        'training pipeline in this repo.'
    )
    sys.exit(1)
cfg.experiment.name = 'bc_diffusion_lift'
cfg.experiment.validate = False
cfg.experiment.rollout.enabled = False
cfg.experiment.render_video = False

# Dataset and output
cfg.train.data = os.path.join(PROJECT_ROOT, 'datasets/lift/ph/low_dim_v141.hdf5')
cfg.train.output_dir = os.path.join(PROJECT_ROOT, 'checkpoints/bc_diffusion_lift/')
cfg.train.num_epochs = 500
cfg.train.batch_size = 256
cfg.train.action_config['actions']['normalization'] = 'min_max'
cfg.train.seq_length = 16
cfg.train.frame_stack = 1
cfg.train.hdf5_load_next_obs = False

# Diffusion horizon settings
cfg.algo.horizon.observation_horizon = 2
cfg.algo.horizon.prediction_horizon = 16
cfg.algo.horizon.action_horizon = 8
cfg.algo.unet.down_dims = [64, 128, 256]

# Diffusion scheduler (DDPM - Denoising Diffusion Probabilistic Model)
cfg.algo.ddpm.enabled = True
cfg.algo.ddpm.num_train_timesteps = 100
cfg.algo.ddpm.num_inference_timesteps = 10
cfg.algo.ddim.enabled = False  # Use DDPM, not DDIM

# Observation modalities for Lift task
cfg.observation.modalities.obs.low_dim = [
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'object',
]

config_dir = os.path.join(PROJECT_ROOT, 'configs')
os.makedirs(config_dir, exist_ok=True)
config_file = os.path.join(config_dir, 'diffusion_policy_lift.json')
with open(config_file, 'w') as f:
    json.dump(cfg, f, indent=4)
print(f'✅ Robomimic Diffusion Policy config saved to {config_file}')
