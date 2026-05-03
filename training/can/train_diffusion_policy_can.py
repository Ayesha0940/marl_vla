#!/usr/bin/env python
"""
Generic training script for diffusion policy on can task using robomimic.
Usage: python training/can/train_diffusion_policy_can.py
"""

import os
import sys
import subprocess

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))


def supports_diffusion_policy():
    try:
        from robomimic.config import config_factory
        config_factory('diffusion_policy')
        return True
    except Exception:
        return False

def main():
    if not supports_diffusion_policy():
        print(
            'This robomimic installation does not include diffusion_policy.\n'
            'Installed version: 0.3.0\n'
            'Supported algos: bc, bcq, cql, iql, gl, hbc, iris, td3_bc\n\n'
            'Install a robomimic build that registers diffusion_policy, or use '
            'the custom diffusion training pipeline in this repo.'
        )
        return 1

    # Step 1: Generate config if it doesn't exist
    config_file = os.path.join(PROJECT_ROOT, 'configs/diffusion_policy_can.json')
    if not os.path.exists(config_file):
        print("Generating diffusion policy config file...")
        gen_config_script = os.path.join(TRAINING_DIR, 'gen_config_diffusion_policy_can.py')
        result = subprocess.run([sys.executable, gen_config_script], cwd=TRAINING_DIR)
        if result.returncode != 0:
            print("Error generating config file!")
            return 1
    
    # Step 2: Find robomimic train.py dynamically
    try:
        import robomimic.scripts.train as train_module
        train_script = train_module.__file__
    except ImportError:
        print("Error: robomimic package not found. Please install it first.")
        return 1
    
    # Step 3: Run training with the generated config
    print(f"\nStarting Diffusion Policy training with config: {config_file}")
    print(f"Using robomimic train script: {train_script}")
    
    cmd = [sys.executable, train_script, '--config', config_file]
    result = subprocess.run(cmd)
    
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())
