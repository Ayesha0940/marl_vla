import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import torch

import os
os.environ['LD_LIBRARY_PATH'] = os.environ.get('LD_LIBRARY_PATH', '') + ':/usr/lib/x86_64-linux-gnu'

checkpoint_path = "/home/axs0940/marl_vla/checkpoints/bc_rnn_lift/bc_rnn_lift/20260405174006/models/model_epoch_600.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

policy, ckpt_dict = FileUtils.policy_from_checkpoint(
    ckpt_path=checkpoint_path,
    device=device,
    verbose=False
)

env_meta = ckpt_dict["env_metadata"]
env = EnvUtils.create_env_from_metadata(
    env_meta=env_meta,
    render=False,
    render_offscreen=False,
    use_image_obs=False,
)

obs = env.reset()
print("Type:", type(obs))
print("Keys:", obs.keys() if isinstance(obs, dict) else "flat array")

if isinstance(obs, dict):
    for k, v in obs.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
    total = sum(v.flatten().shape[0] for v in obs.values())
    print(f"\nTotal flattened dim: {total}")