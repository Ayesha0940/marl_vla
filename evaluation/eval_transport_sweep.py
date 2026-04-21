#!/usr/bin/env python
"""
Sweep transport checkpoints and report success-rate ranking.

This script evaluates selected epochs from one run directory using
robomimic.scripts.run_trained_agent at a fixed horizon (default 700), then
prints a sorted summary by success rate.

Usage:
  python evaluation/eval_transport_sweep.py --run_dir checkpoints/bc_rnn/bc_rnn_transport_gmm/bc_rnn_transport_gmm/<run_id>
  python evaluation/eval_transport_sweep.py --run_dir <run_dir> --epochs 200,400,600,800,1000 --n_rollouts 100
"""

import argparse
import glob
import os
import re
import subprocess
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def setup_mujoco_env(env):
    paths = [
        os.path.expanduser("~/.mujoco/mujoco210/bin"),
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu/nvidia",
    ]
    current = env.get("LD_LIBRARY_PATH", "")
    for p in paths:
        if os.path.exists(p) and p not in current:
            current = f"{p}:{current}"
    env["LD_LIBRARY_PATH"] = current


def parse_args():
    parser = argparse.ArgumentParser(description="Sweep transport checkpoints")
    parser.add_argument("--run_dir", required=True, type=str, help="Run directory containing models/")
    parser.add_argument("--epochs", type=str, default="", help="Comma-separated epochs, e.g. 200,400,600")
    parser.add_argument("--n_rollouts", type=int, default=100)
    parser.add_argument("--horizon", type=int, default=700)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def parse_epoch(path):
    m = re.search(r"model_epoch_(\d+)\.pth$", path)
    return int(m.group(1)) if m else None


def find_checkpoints(models_dir, epochs_csv):
    ckpts = glob.glob(os.path.join(models_dir, "model_epoch_*.pth"))
    ckpts = [(parse_epoch(p), p) for p in ckpts]
    ckpts = [(e, p) for e, p in ckpts if e is not None]
    ckpts.sort(key=lambda x: x[0])

    if not epochs_csv:
        # Default sparse sweep over training trajectory.
        candidates = [100, 200, 300, 400, 500, 650, 800, 900, 1000]
    else:
        candidates = [int(x.strip()) for x in epochs_csv.split(",") if x.strip()]

    by_epoch = {e: p for e, p in ckpts}
    out = [(e, by_epoch[e]) for e in candidates if e in by_epoch]
    return out


def extract_stats(text):
    s = re.search(r'"Success_Rate"\s*:\s*([0-9.]+)', text)
    r = re.search(r'"Return"\s*:\s*([0-9.]+)', text)
    h = re.search(r'"Horizon"\s*:\s*([0-9.]+)', text)
    return (
        float(s.group(1)) if s else None,
        float(r.group(1)) if r else None,
        float(h.group(1)) if h else None,
    )


def main():
    args = parse_args()
    run_dir = args.run_dir if os.path.isabs(args.run_dir) else os.path.join(PROJECT_ROOT, args.run_dir)
    models_dir = os.path.join(run_dir, "models")
    if not os.path.isdir(models_dir):
        print(f"models directory not found: {models_dir}")
        return 1

    checkpoints = find_checkpoints(models_dir, args.epochs)
    if not checkpoints:
        print("no checkpoints found for requested epochs")
        return 1

    env = os.environ.copy()
    setup_mujoco_env(env)

    rows = []
    for epoch, ckpt in checkpoints:
        cmd = [
            "/home/axs0940/miniconda3/envs/vla_marl/bin/python",
            "-m",
            "robomimic.scripts.run_trained_agent",
            "--agent",
            ckpt,
            "--n_rollouts",
            str(args.n_rollouts),
            "--horizon",
            str(args.horizon),
            "--seed",
            str(args.seed),
        ]
        print(f"\nEvaluating epoch {epoch} ...")
        result = subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, capture_output=True, text=True)
        out = (result.stdout or "") + "\n" + (result.stderr or "")
        sr, ret, hor = extract_stats(out)
        rows.append((epoch, sr, ret, hor, result.returncode))
        print(f"epoch={epoch} success={sr} return={ret} horizon={hor} rc={result.returncode}")

    rows_sorted = sorted(rows, key=lambda x: (-1 if x[1] is None else -x[1], x[0]))

    print("\nSummary (best first):")
    for epoch, sr, ret, hor, rc in rows_sorted:
        print(f"epoch={epoch:4d} success={sr} return={ret} horizon={hor} rc={rc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
