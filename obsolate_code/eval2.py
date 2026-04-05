import os, numpy as np, torch
import robosuite as suite
import robomimic.utils.file_utils as FileUtils
from tqdm import trange

CKPT = os.path.expanduser(
    '~/marl_vla/checkpoints/bc_rnn/bc_rnn_transport_osc/20260405042346/models/model_epoch_1000.pth')
N_EP  = 50
MAX_S = 700
device = torch.device('cuda')

policy, _ = FileUtils.policy_from_checkpoint(ckpt_path=CKPT, device=device, verbose=False)
print('Policy loaded')

controller_config = {
    'type': 'OSC_POSE',
    'input_max': 1, 'input_min': -1,
    'output_max': [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    'output_min': [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    'kp': 150, 'damping': 1,
    'impedance_mode': 'fixed',
    'kp_limits': [0, 300], 'damping_limits': [0, 10],
    'position_limits': None, 'orientation_limits': None,
    'uncouple_pos_ori': True, 'control_delta': True,
    'interpolation': None, 'ramp_ratio': 0.2
}

env = suite.make(
    'TwoArmTransport',
    robots=['Panda', 'Panda'],
    env_configuration='single-arm-opposed',
    controller_configs=controller_config,
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
    ignore_done=True,
    control_freq=20,
    reward_shaping=False,
)
print('Action spec shape:', env.action_spec[0].shape)

rewards, successes, subgoals = [], [], []

for ep in trange(N_EP, desc='BC-RNN OSC'):
    raw_obs = env.reset()
    policy.start_episode()
    ep_reward = 0.0
    milestones = []

    for step in range(MAX_S):
        obs_dict = {
            'robot0_joint_pos_cos': raw_obs['robot0_joint_pos_cos'],
            'robot0_joint_pos_sin': raw_obs['robot0_joint_pos_sin'],
            'robot0_joint_vel':     raw_obs['robot0_joint_vel'],
            'robot0_eef_pos':       raw_obs['robot0_eef_pos'],
            'robot0_eef_quat':      raw_obs['robot0_eef_quat'],
            'robot0_gripper_qpos':  raw_obs['robot0_gripper_qpos'],
            'robot0_gripper_qvel':  raw_obs['robot0_gripper_qvel'],
            'robot1_joint_pos_cos': raw_obs['robot1_joint_pos_cos'],
            'robot1_joint_pos_sin': raw_obs['robot1_joint_pos_sin'],
            'robot1_joint_vel':     raw_obs['robot1_joint_vel'],
            'robot1_eef_pos':       raw_obs['robot1_eef_pos'],
            'robot1_eef_quat':      raw_obs['robot1_eef_quat'],
            'robot1_gripper_qpos':  raw_obs['robot1_gripper_qpos'],
            'robot1_gripper_qvel':  raw_obs['robot1_gripper_qvel'],
            'object':               raw_obs['object-state'],
        }

        action = policy(obs_dict)
        raw_obs, reward, done, _ = env.step(action)
        ep_reward += reward
        if reward > 0.5:
            milestones.append(step)
    print(f"Episode {ep+1}/{N_EP} — Reward: {ep_reward:.4f} — Milestones: {milestones}")
    rewards.append(ep_reward)
    successes.append(int(ep_reward >= 4.5))
    subgoals.append(min(5, len(milestones)))

env.close()

print()
print('='*55)
print(f'BC-RNN OSC — {N_EP} episodes')
print('-'*55)
print(f'  Mean reward:    {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}')
print(f'  Success rate:   {np.mean(successes)*100:.1f}%')
print(f'  Mean sub-goals: {np.mean(subgoals):.2f} / 5')
print(f'  Max reward:     {max(rewards):.4f}')
print(f'  Any reward > 0: {sum(r > 0 for r in rewards)} / {N_EP}')
print('='*55)

os.makedirs(os.path.expanduser('~/marl_vla/results'), exist_ok=True)
np.save(os.path.expanduser('~/marl_vla/results/bc_rnn_osc_eval.npy'),
    {'rewards': rewards, 'successes': successes, 'subgoals': subgoals})
print('Saved.')