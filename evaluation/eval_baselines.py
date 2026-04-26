#!/usr/bin/env python3
"""
Run BASE-clean and BASE-noisy once for the whole study.

Output CSV columns:
    alpha_s, alpha_a, base_clean, base_noisy

base_clean is the same for every noise level (clean policy, no noise),
but is repeated on each row for easy joining with per-variant results.

Usage:
    python evaluation/eval_baselines.py \
        --output_csv results/lift/joint_denoiser/baselines.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_BC_CKPT = os.path.join(
    PROJECT_ROOT,
    "checkpoints", "bc_rnn_lift", "bc_rnn_lift",
    "20260405174006", "models", "model_epoch_600.pth",
)
DEFAULT_OUTPUT = os.path.join(PROJECT_ROOT, "results", "lift", "joint_denoiser", "baselines.csv")
DEFAULT_ALPHA_S = [0.01, 0.02, 0.03, 0.04, 0.05]
DEFAULT_ALPHA_A = [0.05, 0.1, 0.2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bc_rnn_ckpt",  default=DEFAULT_BC_CKPT)
    p.add_argument("--alpha_s",      default=",".join(str(v) for v in DEFAULT_ALPHA_S))
    p.add_argument("--alpha_a",      default=",".join(str(v) for v in DEFAULT_ALPHA_A))
    p.add_argument("--n_rollouts",   type=int, default=25)
    p.add_argument("--episode_horizon", type=int, default=400)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--output_csv",   default=DEFAULT_OUTPUT)
    p.add_argument("--device",       default="auto")
    return p.parse_args()


def _configure_mujoco():
    paths = [
        os.path.expanduser("~/.mujoco/mujoco210/bin"),
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib/x86_64-linux-gnu/nvidia",
    ]
    current = os.environ.get("LD_LIBRARY_PATH", "")
    for p in paths:
        if os.path.exists(p) and p not in current.split(":"):
            current = f"{p}:{current}" if current else p
    os.environ["LD_LIBRARY_PATH"] = current


def main():
    args = parse_args()
    _configure_mujoco()

    alpha_s_list = [float(x) for x in args.alpha_s.split(",")]
    alpha_a_list = [float(x) for x in args.alpha_a.split(",")]

    import torch
    from evaluation.eval_joint_denoiser import (
        _load_policy_and_env, _run_rollout_clean, _run_rollout_noisy, _corrupt_obs
    )

    device_str = args.device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") \
             if device_str == "auto" else torch.device(device_str)
    print(f"Device: {device}")

    policy, env, ckpt_dict = _load_policy_and_env(args.bc_rnn_ckpt, device_str)
    obs_keys = list(ckpt_dict["shape_metadata"]["all_obs_keys"])

    # BASE-clean: run once
    print(f"\n[BASE-clean] {args.n_rollouts} rollouts ...")
    clean_successes = []
    for i in range(args.n_rollouts):
        env.reset()
        _, s = _run_rollout_clean(policy, env, args.episode_horizon, args.seed + i)
        clean_successes.append(s)
    base_clean_sr = float(np.mean(clean_successes))
    print(f"  success_rate = {base_clean_sr:.4f}")

    # BASE-noisy: one cell at a time
    rows = []
    total = len(alpha_s_list) * len(alpha_a_list)
    cell = 0
    for a_s in alpha_s_list:
        for a_a in alpha_a_list:
            cell += 1
            print(f"\n[BASE-noisy {cell}/{total}] alpha_s={a_s}  alpha_a={a_a}")
            noisy_successes = []
            for i in range(args.n_rollouts):
                seed = args.seed + int(a_s * 1000) + int(a_a * 1000) * 1000 + i
                env.reset()
                _, s = _run_rollout_noisy(
                    policy, env, obs_keys, a_s, a_a, args.episode_horizon, seed
                )
                noisy_successes.append(s)
            base_noisy_sr = float(np.mean(noisy_successes))
            print(f"  success_rate = {base_noisy_sr:.4f}")
            rows.append({
                "alpha_s":    f"{a_s:.3f}",
                "alpha_a":    f"{a_a:.3f}",
                "base_clean": f"{base_clean_sr:.4f}",
                "base_noisy": f"{base_noisy_sr:.4f}",
            })

    os.makedirs(os.path.dirname(os.path.abspath(args.output_csv)), exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["alpha_s", "alpha_a", "base_clean", "base_noisy"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved: {args.output_csv}")


if __name__ == "__main__":
    main()
