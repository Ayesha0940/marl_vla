import os, json
from robomimic.config import config_factory

# Get the project root directory (parent of training/ directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = config_factory('bc')
cfg.experiment.name = 'bc_rnn_square_v3'
cfg.experiment.validate = False
cfg.experiment.rollout.enabled = True
cfg.experiment.save.every_n_epochs = 100
cfg.experiment.save.on_best_validation = False
cfg.experiment.save.on_best_rollout_return = False
cfg.experiment.save.on_best_rollout_success_rate = True
cfg.experiment.render_video = False
cfg.train.data = os.path.join(PROJECT_ROOT, 'datasets/square/ph/low_dim_v141.hdf5')
cfg.train.output_dir = os.path.join(PROJECT_ROOT, 'checkpoints/bc_rnn_square/')
cfg.train.num_epochs = 2000
cfg.train.batch_size = 32
cfg.train.seq_length = 10
cfg.train.hdf5_normalize_obs = False

# Enable RNN and disable transformer for a stable square baseline
cfg.algo.rnn.enabled = True
cfg.algo.rnn.horizon = 10
cfg.algo.rnn.hidden_dim = 400
cfg.algo.rnn.rnn_type = 'LSTM'
cfg.algo.rnn.num_layers = 2
cfg.algo.transformer.enabled = False

# Smaller MLP head for stable BC training
cfg.algo.actor_layer_dims = [400, 400]

cfg.observation.modalities.obs.low_dim = [
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'robot0_joint_pos',
    'object',
]

config_dir = os.path.join(PROJECT_ROOT, 'configs')
os.makedirs(config_dir, exist_ok=True)
config_file = os.path.join(config_dir, 'bc_rnn_square.json')
with open(config_file, 'w') as f:
    json.dump(cfg, f, indent=4)
print(f'Config saved to {config_file}')
