#!/usr/bin/env python
"""
Generic training script for square task that works across different machines and directories.
Usage: python train_square.py
"""

import os
import sys
import subprocess

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))

def main():
    # Step 1: Generate config if it doesn't exist
    config_file = os.path.join(PROJECT_ROOT, 'configs/bc_rnn_square.json')
    if not os.path.exists(config_file):
        print("Generating config file...")
        gen_config_script = os.path.join(TRAINING_DIR, 'gen_config_square.py')
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
    print(f"\nStarting training with config: {config_file}")
    print(f"Using robomimic train script: {train_script}")
    
    cmd = [sys.executable, train_script, '--config', config_file]
    result = subprocess.run(cmd)
    
    return result.returncode

if __name__ == '__main__':
    sys.exit(main())
