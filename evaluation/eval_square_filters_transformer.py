"""
Evaluate square policy robustness with action noise across multiple predefined noise levels.

Tests noise standard deviations: [0.0, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0]

Results are saved in both JSON and CSV formats for easy analysis.

Examples:
python eval_robustness_square.py --n_rollouts 100
python eval_robustness_square.py --epoch 500 --seed 42

tmux new -s square_eval
tmux attach -t square_eval
"""

import os
import sys
import glob
import subprocess
import argparse
import json
import csv
import re
from datetime import datetime

import numpy as np
from collections import deque
import robomimic.utils.tensor_utils as TensorUtils
import robomimic.algo.algo as AlgoModule


class EMAFilter:
    """Exponential Moving Average filter for action smoothing."""
    def __init__(self, dim, alpha=0.2):
        self.alpha = alpha
        self.x = np.zeros(dim)

    def update(self, a):
        self.x = self.alpha * a + (1 - self.alpha) * self.x
        return self.x


class MedianFilter:
    """Median filter for action noise reduction."""
    def __init__(self, dim, k=5):
        self.buffer = deque(maxlen=k)

    def update(self, a):
        self.buffer.append(a)
        return np.median(self.buffer, axis=0)

class KalmanFilter:
    """Kalman filter for action state estimation."""
    def __init__(self, dim, Q=1e-3, R=1e-2):
        import numpy as np
        self.dim = dim
        self.x = np.zeros(dim)
        self.P = np.eye(dim)
        self.Q = Q * np.eye(dim)
        self.R = R * np.eye(dim)

    def update(self, z):

        # Predict
        x_pred = self.x
        P_pred = self.P + self.Q

        # Gain
        K = P_pred @ np.linalg.inv(P_pred + self.R)

        # Update
        self.x = x_pred + K @ (z - x_pred)
        self.P = (np.eye(self.dim) - K) @ P_pred

        return self.x

# =========================
# PATH SETUP
# =========================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints/bc_rnn_square/bc_rnn_square_v3')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results', 'square')

os.makedirs(RESULTS_DIR, exist_ok=True)


# =========================
# MUJOCO SETUP
# =========================
def setup_mujoco():
    """Configure LD_LIBRARY_PATH for MuJoCo."""
    paths = [
        os.path.expanduser('~/.mujoco/mujoco210/bin'),
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib/x86_64-linux-gnu/nvidia',
    ]

    current = os.environ.get('LD_LIBRARY_PATH', '')

    new_paths = [p for p in paths if os.path.exists(p)]
    updated = ":".join(new_paths + [current])

    os.environ['LD_LIBRARY_PATH'] = updated

    print("🔧 LD_LIBRARY_PATH set to:")
    print(updated)


# =========================
# CHECKPOINT FINDER
# =========================
def find_latest_checkpoint(epoch=1000):
    """Find the latest checkpoint for a given epoch."""
    pattern = os.path.join(CHECKPOINT_DIR, f'*/models/model_epoch_{epoch}.pth')
    matches = sorted(glob.glob(pattern))

    if matches:
        return matches[-1]

    print("❌ Checkpoint not found.")
    return None


def find_best_checkpoint():
    """Find the checkpoint file with the best rollout success rate.

    The script expects checkpoints saved with a filename suffix like:
    model_epoch_1100_NutAssemblySquare_success_0.7.pth
    """
    pattern = os.path.join(CHECKPOINT_DIR, '*/models/model_epoch_*.pth')
    candidates = sorted(glob.glob(pattern))

    best_ckpt = None
    best_success = -1.0
    success_pattern = re.compile(r'_success_([0-9]+(?:\.[0-9]+)?)\.pth$')

    for ckpt in candidates:
        m = success_pattern.search(ckpt)
        if m:
            success = float(m.group(1))
            if success > best_success:
                best_success = success
                best_ckpt = ckpt

    if best_ckpt is not None:
        print(f"✅ Best checkpoint found: {best_ckpt} (success={best_success})")
        return best_ckpt

    print("❌ No best-success checkpoint file found in the checkpoint directory.")
    print("Please use --epoch or --checkpoint_path instead.")
    return None


# =========================
# ARGUMENT PARSER
# =========================
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate square policy robustness with action noise and filtering methods"
    )
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Optional explicit checkpoint file path to evaluate')
    parser.add_argument('--epoch', type=int, default=2000,
                        help='Checkpoint epoch to load when not using --checkpoint_path or --best')
    parser.add_argument('--best', action='store_true',
                        help='Select the best rollout-success checkpoint available in the checkpoint directory')
    parser.add_argument('--n_rollouts', type=int, default=50, help='Number of rollouts per configuration')
    parser.add_argument('--horizon', type=int, default=500, help='Maximum steps per episode')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    return parser.parse_args()


# =========================
# FILTER INITIALIZATION
# =========================
def create_filter(method, action_dim):
    """Create filter instance based on method name.
    
    Args:
        method: Filter method ("none", "kalman", "ema", "median")
        action_dim: Dimension of action space
        
    Returns:
        Filter instance or None
    """
    if method == "kalman":
        return KalmanFilter(dim=action_dim)
    elif method == "ema":
        return EMAFilter(dim=action_dim)
    elif method == "median":
        return MedianFilter(dim=action_dim)
    else:
        return None


# =========================
# POLICY AND ENVIRONMENT LOADING
# =========================
def load_policy_and_environment(checkpoint_path):
    """Load policy and environment from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        tuple: (policy, env)
    """
    import robomimic.utils.file_utils as FileUtils
    import robomimic.utils.env_utils as EnvUtils
    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load policy
    policy, ckpt_dict = FileUtils.policy_from_checkpoint(
        ckpt_path=checkpoint_path,
        device=device,
        verbose=False
    )
    print("✅ Policy loaded")

    # Create environment
    env_meta = ckpt_dict["env_metadata"]
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
    )
    print("✅ Environment created")

    return policy, env


# =========================
# ACTION DIMENSION DETECTION
# =========================
def get_action_dimension(env, policy):
    try:
        return env.action_dimension
    except AttributeError:
        return 7  # hardcode for Square (OSC_POSE = 7)


# =========================
# SINGLE ROLLOUT EVALUATION
# =========================
from collections import deque

POLICY_OBS_KEYS = [
    'robot0_eef_pos', 'robot0_eef_quat',
    'robot0_gripper_qpos', 'robot0_joint_pos', 'object'
]
CONTEXT_LENGTH = 10

def run_single_rollout(policy, env, noise_std, filt, horizon):
    import numpy as np
    import torch

    obs = env.reset()
    policy.start_episode()
    ep_reward = 0.0
    success = False

    for step in range(horizon):
        with torch.no_grad():
            ac = policy(obs)
            if isinstance(ac, torch.Tensor):
                ac = ac.cpu().numpy()
            elif isinstance(ac, (list, tuple)):
                ac = np.asarray(ac[0]) if len(ac) > 0 else np.array([])

        if noise_std > 0:
            ac = ac + np.random.normal(0, noise_std, size=ac.shape)
        if filt is not None:
            ac = filt.update(ac)
        ac = np.clip(ac, -1.0, 1.0)

        obs, reward, done, _ = env.step(ac)
        ep_reward += reward

        if env.is_success()["task"]:
            success = True
            break
        if done:
            break

    return ep_reward, success


# =========================
# NOISE LEVEL EVALUATION
# =========================
def evaluate_noise_and_method(policy, env, noise_std, method, action_dim, n_rollouts, horizon):
    """
    Evaluate policy performance at a specific noise level with a filtering method.
    
    Args:
        policy: Agent policy
        env: Environment instance
        noise_std: Standard deviation of action noise
        method: Filtering method ("none", "kalman", "ema", "median")
        action_dim: Action dimension
        n_rollouts: Number of rollouts per configuration
        horizon: Maximum steps per episode
        
    Returns:
        dict: Results including mean reward and success rate
    """
    rewards = []
    successes = []

    for ep in range(n_rollouts):
        filt = create_filter(method, action_dim)
        ep_reward, success = run_single_rollout(policy, env, noise_std, filt, horizon)
        
        rewards.append(ep_reward)
        successes.append(int(success))

    # Compute statistics
    mean_reward = float(np.mean(rewards))
    success_rate = float(np.mean(successes))

    # Print results for this configuration
    print(f"  📊 Method:         {method}")
    print(f"  📊 Mean Reward:    {mean_reward:.4f}")
    print(f"  📊 Success Rate:   {success_rate * 100:.2f}%")

    return {
        "noise_std": noise_std,
        "mean_reward": mean_reward,
        "success_rate": success_rate,
        "n_rollouts": n_rollouts,
        "seed": 0,
        "method": method
    }


# =========================
# RESULTS PRINTING
# =========================
def print_results_summary(results):
    """Print summary of all evaluation results.
    
    Args:
        results: List of result dictionaries
    """
    print("\n" + "=" * 60)
    print("📊 SUMMARY RESULTS")
    print("=" * 60)
    for result in results:
        print(f"Noise {result['noise_std']:4.2f}, Method {result['method']:8s}: "
              f"Reward={result['mean_reward']:7.4f}, Success={result['success_rate']*100:5.2f}%")
    print("=" * 60)


# =========================
# RESULTS SAVING
# =========================
def save_results_to_json(results, config, timestamp):
    """Save evaluation results to JSON file.
    
    Args:
        results: List of result dictionaries
        config: Configuration dictionary
        timestamp: Timestamp string for filename
        
    Returns:
        str: Path to saved JSON file
    """
    json_filename = f"robustness_eval_{timestamp}.json"
    json_path = os.path.join(RESULTS_DIR, json_filename)
    
    with open(json_path, "w") as f:
        json.dump({
            "results": results,
            "config": config
        }, f, indent=4)

    print(f"💾 Saved JSON results to: {json_path}")
    return json_path


def save_results_to_csv(results, timestamp):
    """Save evaluation results to CSV file.
    
    Args:
        results: List of result dictionaries
        timestamp: Timestamp string for filename
        
    Returns:
        str: Path to saved CSV file
    """
    csv_filename = f"robustness_eval_square_{timestamp}.csv"
    csv_path = os.path.join(RESULTS_DIR, csv_filename)
    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['method', 'noise_std', 'mean_reward', 'success_rate', 'n_rollouts', 'seed']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write header
        writer.writeheader()
        
        # Write data rows
        for result in results:
            writer.writerow({
                'method': result['method'],
                'noise_std': result['noise_std'],
                'mean_reward': f"{result['mean_reward']:.4f}",
                'success_rate': f"{result['success_rate']:.4f}",
                'n_rollouts': result['n_rollouts'],
                'seed': result['seed']
            })

    print(f"💾 Saved CSV results to: {csv_path}")
    return csv_path


# =========================
# MAIN
# =========================
def main():
    """Main evaluation loop."""
    # Parse arguments
    args = parse_arguments()
    
    # Define noise levels and filtering methods to test
    noise_levels = [0.0]
    methods = ["none"]

    # Setup environment
    setup_mujoco()

    # Select checkpoint
    if args.checkpoint_path:
        agent_path = args.checkpoint_path
        if not os.path.isfile(agent_path):
            print(f"❌ Checkpoint file not found: {agent_path}")
            return 1
    elif args.best:
        agent_path = find_best_checkpoint()
        if not agent_path:
            return 1
    else:
        agent_path = find_latest_checkpoint(args.epoch)
        if not agent_path:
            return 1

    print(f"\n📦 Using checkpoint: {agent_path}")
    if args.best:
        print("⭐ Evaluating best rollout-success checkpoint")
    print(f"🌪 Testing noise stds: {noise_levels}")
    print(f"🔧 Testing methods: {methods}")

    # Load policy and environment
    policy, env = load_policy_and_environment(agent_path)

    # Detect action dimension
    action_dim = get_action_dimension(env, policy)
    print(f"🔍 Action dimension: {action_dim}")

    # Run evaluation for all noise levels and methods
    all_results = []
    for noise_std in noise_levels:
        for method in methods:
            print(f"\n🔄 Testing noise_std = {noise_std}, method = {method}")
            result = evaluate_noise_and_method(
                policy, env, noise_std, method, action_dim, 
                args.n_rollouts, args.horizon
            )
            all_results.append(result)

    # Print summary
    print_results_summary(all_results)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        "epoch": args.epoch,
        "n_rollouts": args.n_rollouts,
        "horizon": args.horizon,
        "seed": args.seed,
        "noise_levels": noise_levels,
        "methods": methods
    }
    
    save_results_to_json(all_results, config, timestamp)
    save_results_to_csv(all_results, timestamp)

    return 0


if __name__ == "__main__":
    sys.exit(main())