#!/usr/bin/env python
"""
Generic evaluation script for transport task.

By default it evaluates the final checkpoint (largest epoch) from the latest run
under the transport checkpoint directory, with horizon=700.

Usage:
  python evaluation/eval_transport.py
  python evaluation/eval_transport.py --epoch 1000 --n_rollouts 50
  python evaluation/eval_transport.py --run_dir checkpoints/bc_rnn/bc_rnn_transport_tuned/bc_rnn_transport_tuned/20260419213349
  python evaluation/eval_transport.py --all_runs
"""

import argparse
import glob
import os
import re
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_ROOT = os.path.join(
    PROJECT_ROOT,
    "checkpoints",
    "bc_rnn",
    "bc_rnn_transport_tuned",
    "bc_rnn_transport_tuned",
)


def setup_mujoco():
    """Robust MuJoCo + NVIDIA path setup."""
    paths = [
        os.path.expanduser("~/.mujoco/mujoco210/bin"),
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu/nvidia",
    ]

    current = os.environ.get("LD_LIBRARY_PATH", "")
    updated = False
    for p in paths:
        if os.path.exists(p) and p not in current:
            current = f"{p}:{current}"
            updated = True

    if updated:
        os.environ["LD_LIBRARY_PATH"] = current
        print("Updated LD_LIBRARY_PATH:")
        print(current)


def list_run_dirs(root_dir):
    if not os.path.isdir(root_dir):
        return []
    runs = []
    for name in os.listdir(root_dir):
        run_dir = os.path.join(root_dir, name)
        if os.path.isdir(os.path.join(run_dir, "models")):
            runs.append(run_dir)
    runs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return runs


def parse_epoch_from_path(path):
    match = re.search(r"model_epoch_(\d+)\.pth$", os.path.basename(path))
    if match is None:
        return None
    return int(match.group(1))


def find_checkpoint(run_dir, epoch=None):
    models_dir = os.path.join(run_dir, "models")
    if not os.path.isdir(models_dir):
        return None

    if epoch is not None:
        target = os.path.join(models_dir, f"model_epoch_{epoch}.pth")
        return target if os.path.exists(target) else None

    all_ckpts = glob.glob(os.path.join(models_dir, "model_epoch_*.pth"))
    if not all_ckpts:
        return None

    all_ckpts = [p for p in all_ckpts if parse_epoch_from_path(p) is not None]
    if not all_ckpts:
        return None

    all_ckpts.sort(key=lambda p: parse_epoch_from_path(p), reverse=True)
    return all_ckpts[0]


def run_eval(run_script, agent_path, n_rollouts, horizon, seed):
    cmd = [
        sys.executable,
        run_script,
        "--agent",
        agent_path,
        "--n_rollouts",
        str(n_rollouts),
        "--horizon",
        str(horizon),
        "--seed",
        str(seed),
    ]

    print("\nRunning evaluation...")
    print(f"Command: {' '.join(cmd)}\n")

    return subprocess.run(cmd, env=os.environ.copy()).returncode


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained transport policy",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluation/eval_transport.py\n"
            "  python evaluation/eval_transport.py --epoch 1000 --n_rollouts 50\n"
            "  python evaluation/eval_transport.py --horizon 700 --seed 42\n"
            "  python evaluation/eval_transport.py --all_runs"
        ),
    )
    parser.add_argument(
        "--checkpoint_root",
        type=str,
        default=CHECKPOINT_ROOT,
        help="Directory containing transport run subdirectories",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help="Specific run directory to evaluate (overrides auto-selection)",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="Checkpoint epoch to evaluate. Default: final epoch in run.",
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=50,
        help="Number of rollouts (default: 50)",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=700,
        help="Rollout horizon (default: 700)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed (default: 0)",
    )
    parser.add_argument(
        "--all_runs",
        action="store_true",
        help="Evaluate final (or requested epoch) checkpoints for all runs",
    )

    args = parser.parse_args()

    # Set up MuJoCo environment
    setup_mujoco()

    # Find robomimic run_trained_agent script dynamically
    try:
        import robomimic.scripts.run_trained_agent as run_module

        run_script = run_module.__file__
    except ImportError:
        print("Error: robomimic package not found. Please install it first.")
        return 1

    # Resolve run directories
    if args.run_dir is not None:
        run_dirs = [
            args.run_dir
            if os.path.isabs(args.run_dir)
            else os.path.join(PROJECT_ROOT, args.run_dir)
        ]
    else:
        root = (
            args.checkpoint_root
            if os.path.isabs(args.checkpoint_root)
            else os.path.join(PROJECT_ROOT, args.checkpoint_root)
        )
        run_dirs = list_run_dirs(root)

    if not run_dirs:
        print("Error: No transport runs found.")
        print(f"Checked root: {args.checkpoint_root}")
        return 1

    if not args.all_runs:
        run_dirs = [run_dirs[0]]

    failures = 0
    for idx, run_dir in enumerate(run_dirs, start=1):
        ckpt = find_checkpoint(run_dir, epoch=args.epoch)
        if ckpt is None:
            requested = f"epoch {args.epoch}" if args.epoch is not None else "final epoch"
            print(f"\n[{idx}/{len(run_dirs)}] Skipping {run_dir}")
            print(f"Reason: could not find {requested} checkpoint")
            failures += 1
            continue

        print(f"\n[{idx}/{len(run_dirs)}] Using run: {run_dir}")
        print(f"Checkpoint: {ckpt}")

        rc = run_eval(
            run_script=run_script,
            agent_path=ckpt,
            n_rollouts=args.n_rollouts,
            horizon=args.horizon,
            seed=args.seed,
        )
        if rc != 0:
            failures += 1

    if failures > 0:
        print(f"\nCompleted with failures: {failures}/{len(run_dirs)}")
        return 1

    print("\nEvaluation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
