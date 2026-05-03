import os, json
from robomimic.config import config_factory

# Get the project root directory (parent of training/ directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = config_factory('bc')
cfg.experiment.name = 'bc_rnn_lift'
cfg.experiment.validate = False
cfg.experiment.rollout.enabled = False
cfg.experiment.render_video = False
cfg.train.data = os.path.join(PROJECT_ROOT, 'datasets/lift/ph/low_dim_v141.hdf5')
cfg.train.output_dir = os.path.join(PROJECT_ROOT, 'checkpoints/bc_rnn_lift/')
cfg.train.num_epochs = 600
cfg.train.batch_size = 16
cfg.train.seq_length = 10
cfg.algo.rnn.enabled = True
cfg.algo.rnn.horizon = 10
cfg.algo.rnn.hidden_dim = 400
cfg.algo.rnn.rnn_type = 'LSTM'
cfg.algo.rnn.num_layers = 2
cfg.observation.modalities.obs.low_dim = [
    'robot0_eef_pos',
    'robot0_eef_quat',
    'robot0_gripper_qpos',
    'object',
]
config_dir = os.path.join(PROJECT_ROOT, 'configs')
os.makedirs(config_dir, exist_ok=True)
config_file = os.path.join(config_dir, 'bc_rnn_lift.json')
with open(config_file, 'w') as f:
    json.dump(cfg, f, indent=4)
print(f'Config saved to {config_file}')