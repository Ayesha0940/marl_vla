"""
Train the action-only denoiser for Robomimic Lift.

Reads the demo HDF5 directly via JointDenoiserDataset — no rollout collection needed.
Obs keys are auto-detected from the BC-RNN checkpoint so state_dim matches eval exactly.

The model denoises action conditioned on the (noise-augmented) state, learning to
recover clean actions even when the state sensor is slightly corrupted at deployment.

  Input:  [noisy_action_t; state_noisy]  — D_a + D_s input channels
  Target: eps that recovers clean action  (warm-start separation, default on)
  Loss:   MSE on predicted action noise only

Examples:
    python -m diffusion.train_action_denoiser_lift \\
        --bc_rnn_ckpt checkpoints/bc_rnn_lift/bc_rnn_lift/20260405174006/models/model_epoch_600.pth \\
        --hdf5_path datasets/lift/ph/low_dim_v141.hdf5 \\
        --anchor A7 \\
        --output_path diffusion_models/action_a7_lift.pt

    # A1 anchor (object pose at t=0)
    python -m diffusion.train_action_denoiser_lift \\
        --bc_rnn_ckpt checkpoints/bc_rnn_lift/bc_rnn_lift/20260405174006/models/model_epoch_600.pth \\
        --anchor A1 \\
        --output_path diffusion_models/action_a1_lift.pt
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
    p = argparse.ArgumentParser(description="Train action-only denoiser for Lift")
    p.add_argument("--bc_rnn_ckpt", type=str, required=True,
                   help="BC-RNN checkpoint — used to auto-detect obs_keys and action_dim")
    p.add_argument("--hdf5_path", type=str,
                   default="datasets/lift/ph/low_dim_v141.hdf5",
                   help="Path to robomimic Lift HDF5 demo file")
    p.add_argument("--anchor", type=str, default="A1",
                   choices=["A0","A1","A2","A3","A4","A5","A6","A7","A8"],
                   help="Anchor variant. Default: A1")
    p.add_argument("--horizon", type=int, default=16,
                   help="Temporal window length H. Default: 16")
    p.add_argument("--diffusion_steps", type=int, default=100,
                   help="DDPM diffusion steps T. Default: 100")
    p.add_argument("--epochs", type=int, default=200,
                   help="Training epochs. Default: 200")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--channel_sizes", type=int, nargs=3, default=[64, 128, 256],
                   help="UNet channel sizes (c0 c1 c2). Default: 64 128 256")
    p.add_argument("--aug_alpha_s_max", type=float, default=0.05,
                   help="Max state noise std for augmentation. Default: 0.05")
    p.add_argument("--aug_alpha_a_max", type=float, default=0.20,
                   help="Max action noise std for augmentation. Default: 0.20")
    p.add_argument("--noise_schedule", type=str, default="uniform",
                   choices=["uniform", "asymmetric"],
                   help="'uniform': alpha_s ~ U[0, max]. 'asymmetric': Beta(3,1) bias. Default: uniform")
    p.add_argument("--no_warm_start", action="store_true",
                   help="Disable warm-start: eps target uses noisy x0 instead of clean x0")
    p.add_argument("--gripper_k", type=int, default=5,
                   help="Gripper history length K for A2/A6/A8. Default: 5")
    p.add_argument("--output_path", type=str, default=None,
                   help="Output .pt path. Default: diffusion_models/action_<anchor>_lift.pt")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cuda", "cpu"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log_every", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = (
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else torch.device(args.device)
    )
    print(f"Device: {device}")

    if args.output_path is None:
        os.makedirs("diffusion_models", exist_ok=True)
        args.output_path = f"diffusion_models/action_{args.anchor.lower()}_lift.pt"

    # Auto-detect obs_keys from BC-RNN checkpoint
    print(f"\nLoading obs_keys from BC-RNN checkpoint: {args.bc_rnn_ckpt}")
    bc_ckpt = torch.load(args.bc_rnn_ckpt, map_location="cpu", weights_only=False)
    sm = bc_ckpt["shape_metadata"]
    obs_keys      = list(sm["all_obs_keys"])
    action_dim_bc = int(sm["ac_dim"])
    print(f"  obs_keys:   {obs_keys}")
    print(f"  action_dim: {action_dim_bc}")

    # Dataset
    from diffusion.dataset import JointDenoiserDataset
    print(f"\nBuilding dataset from: {args.hdf5_path}")
    ds = JointDenoiserDataset(
        hdf5_path       = args.hdf5_path,
        horizon         = args.horizon,
        obs_keys        = obs_keys,
        gripper_k       = args.gripper_k,
        normalize       = True,
        aug_alpha_s_max = args.aug_alpha_s_max,
        aug_alpha_a_max = args.aug_alpha_a_max,
        noise_schedule  = args.noise_schedule,
    )
    print(f"  Windows:    {len(ds)}")
    print(f"  state_dim:  {ds.state_dim}")
    print(f"  action_dim: {ds.action_dim}")
    print(f"  object_dim: {ds.object_dim}")
    print(f"  proprio_dim:{ds.proprio_dim}")

    assert ds.action_dim == action_dim_bc, (
        f"action_dim mismatch: dataset={ds.action_dim}, BC-RNN={action_dim_bc}"
    )

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
    )

    # Model + Anchor
    from diffusion.action_denoiser_unet import ActionDenoisingUNet1D, action_denoising_loss
    from diffusion.anchors import build_anchor
    from diffusion.model import make_beta_schedule

    channel_sizes = tuple(args.channel_sizes)
    model = ActionDenoisingUNet1D(
        state_dim     = ds.state_dim,
        action_dim    = ds.action_dim,
        anchor_dim    = 128,
        channel_sizes = channel_sizes,
    ).to(device)

    anchor = build_anchor(
        args.anchor,
        object_dim  = ds.object_dim,
        proprio_dim = ds.proprio_dim,
        gripper_k   = args.gripper_k,
    ).to(device)

    n_params = (sum(p.numel() for p in model.parameters()) +
                sum(p.numel() for p in anchor.parameters()))
    print(f"\nModel params: {n_params:,}  Anchor: {args.anchor}")
    print(f"channel_sizes: {channel_sizes}")

    _, alphas, alphas_bar = make_beta_schedule(args.diffusion_steps)
    alphas_bar = alphas_bar.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(anchor.parameters()), lr=args.lr,
    )

    print(f"\nTraining")
    print(f"  epochs:          {args.epochs}")
    print(f"  batch_size:      {args.batch_size}")
    print(f"  diffusion_steps: {args.diffusion_steps}")
    print(f"  horizon H:       {args.horizon}")
    print(f"  lr:              {args.lr}")
    print(f"  aug_alpha_s_max: {args.aug_alpha_s_max}")
    print(f"  aug_alpha_a_max: {args.aug_alpha_a_max}")
    print(f"  noise_schedule:  {args.noise_schedule}")
    print(f"  warm_start_sep:  {not args.no_warm_start}")
    print(f"  output:          {args.output_path}\n")

    best_loss  = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        anchor.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in loader:
            x0_action_clean = batch["action"].to(device)        # (B, H, D_a) clean
            x0_action_noisy = batch["action_noisy"].to(device)  # (B, H, D_a) noise-augmented
            state_ctx       = batch["state_noisy"].to(device)   # (B, H, D_s) noise-augmented

            _skip = {"state", "action", "state_noisy", "action_noisy"}
            traj  = {k: v.to(device) for k, v in batch.items() if k not in _skip}
            anchor_emb = anchor.compute(traj)                   # (B, D_c)

            loss = action_denoising_loss(
                model,
                x0_action       = x0_action_noisy,
                state_ctx       = state_ctx,
                anchor_emb      = anchor_emb,
                alphas_bar      = alphas_bar,
                x0_action_clean = None if args.no_warm_start else x0_action_clean,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(anchor.parameters()), 1.0,
            )
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        epoch_loss /= n_batches

        if epoch_loss < best_loss:
            best_loss  = epoch_loss
            best_state = {
                "model":  {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "anchor": {k: v.cpu().clone() for k, v in anchor.state_dict().items()},
            }

        if epoch % args.log_every == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs} | loss={epoch_loss:.5f} | best={best_loss:.5f}")

    # Save checkpoint
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    norm = ds.get_normalization_stats()
    torch.save(
        {
            "model_state_dict":  best_state["model"],
            "anchor_state_dict": best_state["anchor"],
            # Architecture — used by eval loader to reconstruct model
            "arch":              "action_unet",
            "state_dim":         ds.state_dim,
            "action_dim":        ds.action_dim,
            "horizon":           args.horizon,
            "diffusion_steps":   args.diffusion_steps,
            "anchor_dim":        128,
            "channel_sizes":     channel_sizes,
            "time_emb_dim":      128,
            # Anchor meta
            "anchor_id":         args.anchor,
            "object_dim":        ds.object_dim,
            "proprio_dim":       ds.proprio_dim,
            "gripper_k":         args.gripper_k,
            # Normalization
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
    print(f"\nSaved: {args.output_path}")
    print(f"  best_loss:  {best_loss:.6f}")
    print(f"  state_dim:  {ds.state_dim}")
    print(f"  action_dim: {ds.action_dim}")


if __name__ == "__main__":
    main()
