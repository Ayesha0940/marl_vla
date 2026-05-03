import os, json
from robomimic.config import config_factory

cfg = config_factory('bc')
cfg.experiment.name = 'bc_rnn_can'
cfg.experiment.validate = False
cfg.experiment.rollout.enabled = False
cfg.experiment.render_video = False
cfg.train.data = os.path.expanduser('~/marl_vla/datasets/can/ph/low_dim_v141.hdf5')
cfg.train.output_dir = os.path.expanduser('~/marl_vla/checkpoints/bc_rnn_can/')
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
with open(os.path.expanduser('~/marl_vla/configs/bc_rnn_can.json'), 'w') as f:
    json.dump(cfg, f, indent=4)
print('Config saved')