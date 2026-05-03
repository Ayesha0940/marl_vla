#!/usr/bin/env python
"""
Train BC-RNN for the transport task.

Usage:
  python training/train_transport.py
  python training/train_transport.py --config configs/bc_rnn_transport.json
  python training/train_transport.py --gen-config-only
"""

import argparse
import os
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAINING_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(description="Launch transport BC-RNN training")
    parser.add_argument(
        "--config",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "bc_rnn_transport.json"),
        help="Path to training config JSON",
    )
    parser.add_argument(
        "--overwrite-config",
        action="store_true",
        help="Regenerate config even if it already exists",
    )
    parser.add_argument(
        "--gen-config-only",
        action="store_true",
        help="Only generate config and exit",
    )
    return parser.parse_args()


def resolve_config_path(config_arg):
    if os.path.isabs(config_arg):
        return config_arg
    return os.path.join(PROJECT_ROOT, config_arg)


def ensure_config(config_path, overwrite=False):
    gen_script = os.path.join(TRAINING_DIR, "gen_config_transport.py")

    if os.path.exists(config_path) and not overwrite:
        print(f"Using existing config: {config_path}")
        return 0

    cmd = [sys.executable, gen_script, "--output", config_path]
    if overwrite:
        cmd.append("--overwrite")

    print("Generating transport config...")
    result = subprocess.run(cmd, cwd=TRAINING_DIR)
    return result.returncode


def find_robomimic_train_script():
    try:
        import robomimic.scripts.train as train_module
    except ImportError:
        return None
    return train_module.__file__


def main():
    args = parse_args()
    config_path = resolve_config_path(args.config)

    rc = ensure_config(config_path, overwrite=args.overwrite_config)
    if rc != 0:
        print("Failed to generate config.")
        return rc

    if args.gen_config_only:
        print("Config generation completed.")
        return 0

    train_script = find_robomimic_train_script()
    if train_script is None:
        print("Error: robomimic is not installed in this environment.")
        return 1

    cmd = [sys.executable, train_script, "--config", config_path]
    print(f"Starting transport training with config: {config_path}")
    print(f"Using robomimic train script: {train_script}")

    result = subprocess.run(cmd)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
