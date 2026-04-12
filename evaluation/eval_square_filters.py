"""
Evaluate square policy robustness with action noise across multiple predefined noise levels.

Tests noise standard deviations: [0.0, 0.05, 0.1, 0.2, 0.5, 0.75, 1.0]

Results are saved in both JSON and CSV formats for easy analysis.

Examples:
python eval_robustness_square.py --n_rollouts 100
python eval_robustness_square.py --epoch 500 --seed 42

tmux new -s square_eval
"""

import os
import sys
import glob
import subprocess
import argparse
import json
import csv
from datetime import datetime

import numpy as np
from collections import deque

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
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints/bc_rnn_square/bc_rnn_square_v2')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')

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


# =========================
# ARGUMENT PARSER
# =========================
def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate square policy robustness with action noise and filtering methods"
    )
    parser.add_argument('--epoch', type=int, default=2000, help='Checkpoint epoch to load')
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
    """Detect action dimension from environment or policy.
    
    Args:
        env: Environment instance
        policy: Policy instance
        
    Returns:
        int: Action dimension
    """
    try:
        action_dim = env.action_spec[0].shape[0]
    except:
        action_dim = len(policy(env.reset()))
    return action_dim


# =========================
# SINGLE ROLLOUT EVALUATION
# =========================
def run_single_rollout(policy, env, noise_std, filt, horizon):
    """
    Run a single episode with noisy actions and optional filtering.
    
    Args:
        policy: Agent policy
        env: Environment instance
        noise_std: Standard deviation of action noise
        filt: Filter instance (or None for no filtering)
        horizon: Maximum steps per episode
        
    Returns:
        tuple: (episode_reward, success)
    """
    obs = env.reset()
    policy.start_episode()
    ep_reward = 0.0
    success = False

    for step in range(horizon):
        # Get action from policy
        action = policy(obs)

        # Add noise if specified
        if noise_std > 0:
            noise = np.random.normal(0, noise_std, size=action.shape)
            noisy_action = action + noise
        else:
            noisy_action = action

        # Apply filter if specified
        if filt is not None:
            action = filt.update(noisy_action)
        else:
            action = noisy_action

        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Take environment step
        obs, reward, done, _ = env.step(action)
        ep_reward += reward

        # Check for task success
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
    noise_levels = [0.0, 0.1, 0.2, 0.5, 0.75]
    methods = ["none", "kalman", "ema", "median"]

    # Setup environment
    setup_mujoco()

    # Load checkpoint
    agent_path = find_latest_checkpoint(args.epoch)
    if not agent_path:
        return 1

    print(f"\n📦 Using checkpoint: {agent_path}")
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