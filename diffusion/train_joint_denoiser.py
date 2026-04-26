"""
Train the joint (state, action) denoiser for Robomimic Lift.

Reads the demo HDF5 directly via JointDenoiserDataset — no rollout collection needed.
Obs keys are auto-detected from the BC-RNN checkpoint so state_dim matches eval exactly.

Saves a self-describing checkpoint readable by evaluation/eval_joint_denoiser.py.

Examples:
    # Train A1 anchor (default)
    python -m diffusion.train_joint_denoiser \
        --bc_rnn_ckpt checkpoints/bc_rnn_lift/bc_rnn_lift/20260405174006/models/model_epoch_600.pth \
        --hdf5_path   datasets/lift/ph/low_dim_v141.hdf5 \
        --anchor      A1 \
        --output_path diffusion_models/joint_a1_lift.pt

    # Different anchor
    python -m diffusion.train_joint_denoiser \
        --bc_rnn_ckpt checkpoints/bc_rnn_lift/bc_rnn_lift/20260405174006/models/model_epoch_600.pth \
        --hdf5_path   datasets/lift/ph/low_dim_v141.hdf5 \
        --anchor      A3 \
        --output_path diffusion_models/joint_a3_lift.pt
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def parse_args():
    p = argparse.ArgumentParser(description="Train joint (state, action) denoiser")
    p.add_argument("--bc_rnn_ckpt", type=str, required=True,
                   help="Path to BC-RNN checkpoint. Used to auto-detect obs_keys.")
    p.add_argument("--hdf5_path", type=str,
                   default="datasets/lift/ph/low_dim_v141.hdf5",
                   help="Path to robomimic Lift HDF5 demo file.")
    p.add_argument("--anchor", type=str, default="A1",
                   choices=["A0","A1","A2","A3","A4","A5","A6","A7","A8"],
                   help="Anchor variant. Default: A1")
    p.add_argument("--horizon", type=int, default=16,
                   help="Temporal window length H. Default: 16")
    p.add_argument("--diffusion_steps", type=int, default=100,
                   help="Number of DDPM diffusion steps T. Default: 100")
    p.add_argument("--epochs", type=int, default=200,
                   help="Training epochs. Default: 200")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Batch size. Default: 256")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Adam learning rate. Default: 1e-4")
    p.add_argument("--lam", type=float, default=None,
                   help="State loss weight λ. Default: action_dim/state_dim")
    p.add_argument("--aug_alpha_s_max", type=float, default=0.05,
                   help="Max deployment state noise std for augmentation. Default: 0.05")
    p.add_argument("--aug_alpha_a_max", type=float, default=0.20,
                   help="Max deployment action noise std for augmentation. Default: 0.20")
    p.add_argument("--noise_schedule", type=str, default="uniform",
                   choices=["uniform", "asymmetric"],
                   help="'uniform': sample alpha_s ~ U[0, max]. "
                        "'asymmetric': bias alpha_s toward [0.03,0.05] via Beta(3,1). "
                        "Default: uniform")
    p.add_argument("--no_warm_start", action="store_true",
                   help="Disable warm-start separation: eps target uses noisy x0 instead "
                        "of clean x0. By default (off), the loss targets clean x0 so the "
                        "model learns to pull off-manifold inputs back to clean.")
    p.add_argument("--gripper_k", type=int, default=5,
                   help="Gripper history length K for A2/A6/A8. Default: 5")
    p.add_argument("--channel_sizes", type=str, default="64,128,256",
                   help="U-Net channel sizes c0,c1,c2. Default: 64,128,256")
    p.add_argument("--output_path", type=str, default=None,
                   help="Output .pt path. Default: diffusion_models/joint_<anchor>_lift.pt")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10,
                   help="Print loss every N epochs. Default: 10")
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # ── Device ─────────────────────────────────────────────────────────────────
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"Device: {device}")

    # ── Default output path ────────────────────────────────────────────────────
    if args.output_path is None:
        os.makedirs("diffusion_models", exist_ok=True)
        args.output_path = f"diffusion_models/joint_{args.anchor.lower()}_lift.pt"

    channel_sizes = tuple(int(x) for x in args.channel_sizes.split(","))
    assert len(channel_sizes) == 3, "--channel_sizes must be three comma-separated ints"

    # ── Auto-detect obs_keys from BC-RNN checkpoint ────────────────────────────
    print(f"\nLoading obs_keys from BC-RNN checkpoint: {args.bc_rnn_ckpt}")
    bc_ckpt = torch.load(args.bc_rnn_ckpt, map_location="cpu", weights_only=False)
    sm = bc_ckpt["shape_metadata"]
    obs_keys = list(sm["all_obs_keys"])
    action_dim_bc = int(sm["ac_dim"])
    print(f"  obs_keys:   {obs_keys}")
    print(f"  action_dim: {action_dim_bc}")

    # ── Dataset ────────────────────────────────────────────────────────────────
    from diffusion.dataset import JointDenoiserDataset

    print(f"\nBuilding dataset from: {args.hdf5_path}")
    ds = JointDenoiserDataset(
        hdf5_path          = args.hdf5_path,
        horizon            = args.horizon,
        obs_keys           = obs_keys,
        gripper_k          = args.gripper_k,
        normalize          = True,
        aug_alpha_s_max    = args.aug_alpha_s_max,
        aug_alpha_a_max    = args.aug_alpha_a_max,
        noise_schedule     = args.noise_schedule,
    )
    print(f"  Windows:    {len(ds)}")
    print(f"  state_dim:  {ds.state_dim}")
    print(f"  action_dim: {ds.action_dim}")
    print(f"  object_dim: {ds.object_dim}")
    print(f"  proprio_dim:{ds.proprio_dim}")

    assert ds.action_dim == action_dim_bc, (
        f"action_dim mismatch: dataset={ds.action_dim}, BC-RNN={action_dim_bc}. "
        "Check obs_keys or HDF5 file."
    )

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=(device.type == "cuda"))

    # ── Model + Anchor ─────────────────────────────────────────────────────────
    from diffusion.joint_unet import JointUNet1D
    from diffusion.anchors import build_anchor
    from diffusion.model import make_beta_schedule

    model = JointUNet1D(
        state_dim     = ds.state_dim,
        action_dim    = ds.action_dim,
        anchor_dim    = 128,
        time_emb_dim  = 128,
        channel_sizes = channel_sizes,
    ).to(device)

    anchor = build_anchor(
        args.anchor,
        object_dim  = ds.object_dim,
        proprio_dim = ds.proprio_dim,
        gripper_k   = args.gripper_k,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) + \
               sum(p.numel() for p in anchor.parameters())
    print(f"\nModel params: {n_params:,}")
    print(f"Anchor:       {args.anchor}")
    print(f"Channel sizes:{channel_sizes}")

    # ── Diffusion schedule ────────────────────────────────────────────────────
    _, alphas, alphas_bar = make_beta_schedule(args.diffusion_steps)
    alphas_bar = alphas_bar.to(device)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(anchor.parameters()),
        lr=args.lr,
    )

    from diffusion.joint_unet import joint_diffusion_loss

    print(f"\nTraining")
    print(f"  epochs:          {args.epochs}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  diffusion_steps: {args.diffusion_steps}")
    print(f"  horizon H:       {args.horizon}")
    print(f"  lr:              {args.lr}")
    print(f"  lam:             {args.lam if args.lam is not None else 'auto (Da/Ds)'}")
    print(f"  aug_alpha_s_max: {args.aug_alpha_s_max}")
    print(f"  aug_alpha_a_max: {args.aug_alpha_a_max}")
    print(f"  noise_schedule:  {args.noise_schedule}")
    print(f"  warm_start_sep:  {not args.no_warm_start}")
    print(f"  output:          {args.output_path}\n")

    best_loss = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        anchor.train()
        epoch_total = epoch_la = epoch_ls = 0.0
        n_batches = 0

        for batch in loader:
            x0_state_clean  = batch["state"].to(device)         # (B, H, D_s) clean
            x0_action_clean = batch["action"].to(device)        # (B, H, D_a) clean
            x0_state_noisy  = batch["state_noisy"].to(device)   # (B, H, D_s) augmented
            x0_action_noisy = batch["action_noisy"].to(device)  # (B, H, D_a) augmented

            # Anchor inputs (exclude state/action keys)
            _skip = {"state", "action", "state_noisy", "action_noisy"}
            traj = {k: v.to(device) for k, v in batch.items() if k not in _skip}

            anchor_emb = anchor.compute(traj)   # (B, D_c)

            # Forward process from noisy x0.
            # With warm-start separation (default): eps target recovers CLEAN x0.
            # Without (--no_warm_start): standard DDPM, eps target is noisy x0.
            total, la, ls = joint_diffusion_loss(
                model, x0_state_noisy, x0_action_noisy, anchor_emb, alphas_bar,
                lam=args.lam,
                x0_state_clean=None if args.no_warm_start else x0_state_clean,
                x0_action_clean=None if args.no_warm_start else x0_action_clean,
            )

            optimizer.zero_grad()
            total.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(anchor.parameters()), 1.0
            )
            optimizer.step()

            epoch_total += total.item()
            epoch_la    += la.item()
            epoch_ls    += ls.item()
            n_batches   += 1

        epoch_total /= n_batches
        epoch_la    /= n_batches
        epoch_ls    /= n_batches

        if epoch_total < best_loss:
            best_loss  = epoch_total
            best_state = {
                "model":  {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "anchor": {k: v.cpu().clone() for k, v in anchor.state_dict().items()},
            }

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs} | "
                  f"total={epoch_total:.5f}  loss_a={epoch_la:.5f}  loss_s={epoch_ls:.5f} | "
                  f"best={best_loss:.5f}")

    # ── Save checkpoint ────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    norm = ds.get_normalization_stats()
    torch.save(
        {
            # Weights
            "model_state_dict":  best_state["model"],
            "anchor_state_dict": best_state["anchor"],
            # Architecture
            "state_dim":         ds.state_dim,
            "action_dim":        ds.action_dim,
            "horizon":           args.horizon,
            "diffusion_steps":   args.diffusion_steps,
            "channel_sizes":     channel_sizes,
            "anchor_dim":        128,
            "time_emb_dim":      128,
            # Anchor meta
            "anchor_id":         args.anchor,
            "object_dim":        ds.object_dim,
            "proprio_dim":       ds.proprio_dim,
            "gripper_k":         args.gripper_k,
            # Normalization (numpy arrays)
            "state_mean":        norm["state_mean"],
            "state_std":         norm["state_std"],
            "action_mean":       norm["action_mean"],
            "action_std":        norm["action_std"],
            # Training metadata
            "obs_keys":          obs_keys,
            "best_loss":         best_loss,
            "epochs":            args.epochs,
        },
        args.output_path,
    )
    print(f"\nSaved best checkpoint to: {args.output_path}")
    print(f"  best_loss:  {best_loss:.6f}")
    print(f"  state_dim:  {ds.state_dim}")
    print(f"  action_dim: {ds.action_dim}")
    print(f"  object_dim: {ds.object_dim}")


if __name__ == "__main__":
    main()
