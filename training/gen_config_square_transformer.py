import os, json
from robomimic.config import config_factory

# Get the project root directory (parent of training/ directory)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

cfg = config_factory('bc')
cfg.experiment.name = 'bc_rnn_square_v3'
cfg.experiment.validate = False
cfg.experiment.rollout.enabled = False
cfg.experiment.render_video = False
cfg.train.data = os.path.join(PROJECT_ROOT, 'datasets/square/ph/low_dim_v141.hdf5')
cfg.train.output_dir = os.path.join(PROJECT_ROOT, 'checkpoints/bc_rnn_square/')
cfg.train.num_epochs = 2000
cfg.train.batch_size = 16
cfg.train.seq_length = 10

# Disable RNN
cfg.algo.rnn.enabled = False

# Enable Transformer
cfg.algo.transformer.enabled = True
cfg.algo.transformer.context_length = 10
cfg.algo.transformer.embed_dim = 512
cfg.algo.transformer.num_layers = 6
cfg.algo.transformer.num_heads = 8
cfg.algo.transformer.emb_dropout = 0.1
cfg.algo.transformer.attn_dropout = 0.1
cfg.algo.transformer.block_output_dropout = 0.1
cfg.algo.transformer.supervise_all_steps = True

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