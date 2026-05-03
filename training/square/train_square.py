#!/usr/bin/env python
"""
Generic training script for square task that works across different machines and directories.
Usage: python train_square.py
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
import subprocess
from datetime import datetime

# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))


def setup_mujoco():
    """Configure LD_LIBRARY_PATH for MuJoCo before training."""
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


def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Launch square training with optional seed and output directory overrides.'
    )
    parser.add_argument('--seed', type=int, default=None,
                        help='Optional seed override for this training run')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Optional name for the output checkpoint subdirectory')
    parser.add_argument('--config', type=str, default=None,
                        help='Optional base config file to use instead of configs/bc_rnn_square.json')
    return parser.parse_args()


def make_run_config(base_config_path, seed, output_dir):
    with open(base_config_path, 'r') as f:
        config = json.load(f)

    if seed is not None:
        config['train']['seed'] = seed

    config['train']['output_dir'] = output_dir

    if 'experiment' in config and isinstance(config['experiment'], dict):
        base_name = config['experiment'].get('name', 'bc_rnn_square')
        suffix = os.path.basename(output_dir)
        config['experiment']['name'] = f"{base_name}_{suffix}"

    override_name = f"bc_rnn_square_{os.path.basename(output_dir)}.json"
    override_path = os.path.join(PROJECT_ROOT, 'configs', override_name)
    with open(override_path, 'w') as f:
        json.dump(config, f, indent=4)

    return override_path


def find_latest_log_file(output_dir):
    pattern = os.path.join(output_dir, '**', 'logs', 'log.txt')
    matches = glob.glob(pattern, recursive=True)
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def export_rollout_csv_from_log(log_path, output_dir):
    if not log_path or not os.path.isfile(log_path):
        return None

    with open(log_path, 'r') as f:
        lines = f.readlines()

    records = []
    epoch = None
    capture = False
    json_lines = []

    for line in lines:
        line = line.rstrip('\n')
        epoch_match = re.match(r'^Epoch\s+(\d+)\s+Rollouts\s+took', line)
        if epoch_match:
            epoch = int(epoch_match.group(1))
            capture = True
            json_lines = []
            continue

        if capture:
            if line.strip().startswith('{'):
                json_lines = [line]
                continue
            if json_lines:
                json_lines.append(line)
                if line.strip() == '}':
                    try:
                        data = json.loads('\n'.join(json_lines))
                    except json.JSONDecodeError:
                        capture = False
                        epoch = None
                        json_lines = []
                        continue
                    data['epoch'] = epoch
                    records.append(data)
                    capture = False
                    epoch = None
                    json_lines = []

    if not records:
        return None

    csv_path = os.path.join(output_dir, 'rollout_eval_metrics.csv')
    fieldnames = ['epoch', 'Horizon', 'Return', 'Success_Rate', 'Time_Episode', 'time']

    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {key: record.get(key, '') for key in fieldnames}
            writer.writerow(row)

    return csv_path


def main():
    args = parse_arguments()

    # Step 1: Generate base config if it doesn't exist
    base_config_file = args.config or os.path.join(PROJECT_ROOT, 'configs/bc_rnn_square.json')
    if not os.path.exists(base_config_file):
        print("Generating config file...")
        gen_config_script = os.path.join(TRAINING_DIR, 'gen_config_square.py')
        result = subprocess.run([sys.executable, gen_config_script], cwd=TRAINING_DIR)
        if result.returncode != 0:
            print("Error generating config file!")
            return 1

    # Step 1b: Override config for per-seed run if requested
    config_file = base_config_file
    output_dir = None
    if args.seed is not None or args.run_name is not None:
        run_name = args.run_name
        if run_name is None:
            run_name = f"seed_{args.seed}" if args.seed is not None else datetime.now().strftime('run_%Y%m%d_%H%M%S')
        output_dir = os.path.join(PROJECT_ROOT, 'checkpoints/bc_rnn_square', run_name)
        config_file = make_run_config(base_config_file, args.seed, output_dir)

    # Step 2: Find robomimic train.py dynamically
    try:
        import robomimic.scripts.train as train_module
        train_script = train_module.__file__
    except ImportError:
        print("Error: robomimic package not found. Please install it first.")
        return 1

    # Step 3: Run training with the selected config
    print(f"\nStarting training with config: {config_file}")
    print(f"Using robomimic train script: {train_script}")

    setup_mujoco()
    cmd = [sys.executable, train_script, '--config', config_file]
    result = subprocess.run(cmd)

    if result.returncode == 0 and output_dir is not None:
        if os.path.isdir(output_dir):
            log_file = find_latest_log_file(output_dir)
            csv_file = export_rollout_csv_from_log(log_file, output_dir)
            if csv_file:
                print(f"💾 Saved rollout evaluation CSV to: {csv_file}")
            else:
                print("⚠️ No rollout evaluation records found in log file.")
        else:
            print(f"⚠️ Output directory not found: {output_dir}")

    return result.returncode

if __name__ == '__main__':
    sys.exit(main())
