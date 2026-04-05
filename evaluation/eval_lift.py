#!/usr/bin/env python
"""
Generic evaluation script for lift task that works across different machines and directories.
Usage: python eval_lift.py [--epoch EPOCH] [--n_rollouts N] [--horizon H] [--seed SEED]
"""

import os
import sys
import glob
import subprocess
import argparse

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints/bc_rnn_lift/bc_rnn_lift')

def setup_mujoco():
    """Robust MuJoCo + NVIDIA path setup"""

    paths = [
        os.path.expanduser('~/.mujoco/mujoco210/bin'),
        '/usr/lib/x86_64-linux-gnu',
        '/usr/lib/x86_64-linux-gnu/nvidia',   # 🔥 IMPORTANT
    ]

    current = os.environ.get('LD_LIBRARY_PATH', '')

    updated = False
    for p in paths:
        if os.path.exists(p) and p not in current:
            current = f"{p}:{current}"
            updated = True

    if updated:
        os.environ['LD_LIBRARY_PATH'] = current
        print("Updated LD_LIBRARY_PATH:")
        print(current)

def find_latest_checkpoint(epoch=600):
    """Find the latest checkpoint or a specific epoch"""
    # Look for any subdirectory with the model
    pattern = os.path.join(CHECKPOINT_DIR, '*/models/model_epoch_{}.pth'.format(epoch))
    matches = glob.glob(pattern)
    
    if matches:
        return matches[0]
    
    # If specific epoch not found, list available epochs
    pattern = os.path.join(CHECKPOINT_DIR, '*/models/model_epoch_*.pth')
    available = glob.glob(pattern)
    
    if available:
        print(f"Available checkpoints:")
        for ckpt in sorted(available):
            print(f"  - {ckpt}")
        print(f"\nError: Checkpoint for epoch {epoch} not found!")
    else:
        print(f"Error: No checkpoints found in {CHECKPOINT_DIR}")
        print("Please run training first: python train_lift.py")
    
    return None

def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained lift policy',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python eval_lift.py\n'
               '  python eval_lift.py --epoch 600 --n_rollouts 50\n'
               '  python eval_lift.py --seed 42'
    )
    parser.add_argument('--epoch', type=int, default=600, help='Checkpoint epoch (default: 600)')
    parser.add_argument('--n_rollouts', type=int, default=50, help='Number of rollouts (default: 50)')
    parser.add_argument('--horizon', type=int, default=400, help='Rollout horizon (default: 400)')
    parser.add_argument('--seed', type=int, default=0, help='Random seed (default: 0)')
    
    args = parser.parse_args()
    
    # Set up MuJoCo environment
    setup_mujoco()
    
    # Find checkpoint
    agent_path = find_latest_checkpoint(args.epoch)
    if not agent_path:
        return 1
    
    print(f"Using checkpoint: {agent_path}")
    
    # Find robomimic run_trained_agent script dynamically
    try:
        import robomimic.scripts.run_trained_agent as run_module
        run_script = run_module.__file__
    except ImportError:
        print("Error: robomimic package not found. Please install it first.")
        return 1
    
    # Build command
    cmd = [
        sys.executable,
        run_script,
        '--agent', agent_path,
        '--n_rollouts', str(args.n_rollouts),
        '--horizon', str(args.horizon),
        '--seed', str(args.seed),
    ]
    
    print(f"\nRunning evaluation...")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, env=os.environ.copy())
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())
