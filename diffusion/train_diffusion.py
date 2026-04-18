"""
Train a DDPM-style diffusion model on (cond, action) pairs collected
from a trained robomimic policy.

Task-agnostic: works for Lift, Can, Square, or any robomimic task.
All dims and cond_mode are read from the .npz file produced by collect_diffusion_data.py.

The saved checkpoint is self-describing — it contains everything needed
to load and run the model at eval time (dims, cond_mode, obs_keys, normalization stats).

Conditioning modes (set during collection, auto-read here):
    state:        cond_dim = Ds
    vision:       cond_dim = 512
    state+vision: cond_dim = Ds + 512

Examples:
    # State conditioning (original)
    python -m diffusion.train_diffusion \\
        --data_path diffusion_data/square_diffusion_data_H1_state.npz \\
        --task square

    # Vision conditioning
    python -m diffusion.train_diffusion \\
        --data_path diffusion_data/square_diffusion_data_H1_vision.npz \\
        --output_path diffusion_models/square_diffusion_model_vision.pt \\
        --task square

    # State + Vision
    python -m diffusion.train_diffusion \\
        --data_path diffusion_data/square_diffusion_data_H1_state_plus_vision.npz \\
        --output_path diffusion_models/square_diffusion_model_statevision.pt \\
        --task square
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F


# =========================
# ARGUMENT PARSER
# =========================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train diffusion model for action denoising"
    )
    parser.add_argument(
        '--data_path', type=str, required=True,
        help='Path to .npz file from collect_diffusion_data.py'
    )
    parser.add_argument(
        '--output_path', type=str, default=None,
        help='Path to save trained model .pt file. '
             'Defaults to diffusion_models/<task>_diffusion_model_<cond_mode>.pt'
    )
    parser.add_argument(
        '--task', type=str, default=None,
        help='Task name. Auto-read from .npz if not provided.'
    )
    parser.add_argument(
        '--diffusion_steps', type=int, default=100,
        help='Number of diffusion steps T (default: 100)'
    )
    parser.add_argument(
        '--hidden_dim', type=int, default=256,
        help='Hidden dim for diffusion model MLP (default: 256)'
    )
    parser.add_argument(
        '--batch_size', type=int, default=256,
        help='Training batch size (default: 256)'
    )
    parser.add_argument(
        '--num_epochs', type=int, default=100,
        help='Number of training epochs (default: 100)'
    )
    parser.add_argument(
        '--lr', type=float, default=1e-4,
        help='Learning rate (default: 1e-4)'
    )
    parser.add_argument(
        '--seed', type=int, default=0,
        help='Random seed'
    )
    return parser.parse_args()


# =========================
# MAIN TRAINING LOOP
# =========================

def main():
    args = parse_arguments()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Force CPU — matches MPE testbed decision, avoids CUBLAS issues
    device = torch.device("cpu")
    print(f"🖥  Device: {device} (forced, matches MPE testbed)")

    # -------------------------
    # Load dataset
    # -------------------------
    print(f"\n📂 Loading data from: {args.data_path}")
    data = np.load(args.data_path, allow_pickle=True)

    conds   = data["conds"]    # [N, H, Dc]
    actions = data["actions"]  # [N, H, Da]

    N,  H,  Dc = conds.shape
    _,  _,  Da = actions.shape

    # Read metadata — all saved by collect_diffusion_data.py
    obs_keys   = list(data["obs_keys"])
    state_dim  = int(data["state_dim"])
    cond_dim   = int(data["cond_dim"])
    action_dim = int(data["action_dim"])
    cond_mode  = str(data["cond_mode"])
    task       = str(data["task"]) if args.task is None else args.task

    assert Dc == cond_dim,   f"cond_dim mismatch: {Dc} vs {cond_dim}"
    assert Da == action_dim, f"action_dim mismatch: {Da} vs {action_dim}"

    print(f"✅ Dataset loaded")
    print(f"   task:       {task}")
    print(f"   cond_mode:  {cond_mode}")
    print(f"   N:          {N} windows")
    print(f"   H:          {H}")
    print(f"   cond_dim:   {Dc}")
    print(f"   action_dim: {Da}")
    print(f"   obs_keys:   {obs_keys}")

    # Default output path encodes cond_mode so files don't collide
    if args.output_path is None:
        os.makedirs("diffusion_models", exist_ok=True)
        cond_suffix = cond_mode.replace('+', '_plus_')
        args.output_path = f"diffusion_models/{task}_diffusion_model_{cond_suffix}.pt"

    # -------------------------
    # Build tensors
    # -------------------------
    conds_t   = torch.from_numpy(conds).float()    # [N, H, Dc]
    actions_t = torch.from_numpy(actions).float()  # [N, H, Da]

    # Normalize actions — same approach as MPE testbed
    act_mean  = actions_t.mean(dim=(0, 1), keepdim=True)  # [1, 1, Da]
    act_std   = actions_t.std(dim=(0, 1),  keepdim=True) + 1e-6
    actions_t = (actions_t - act_mean) / act_std

    print(f"\n📊 Action normalization")
    print(f"   mean: {act_mean.squeeze().numpy().round(4)}")
    print(f"   std:  {act_std.squeeze().numpy().round(4)}")

    # -------------------------
    # Build model
    # -------------------------
    from diffusion.model import TrajectoryDiffusion, make_beta_schedule, q_sample

    model = TrajectoryDiffusion(
        horizon    = H,
        action_dim = Da,
        cond_dim   = Dc,   # automatically correct for all 3 modes
        hidden_dim = args.hidden_dim,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 Model | cond_mode={cond_mode} | cond_dim={Dc} | params={total_params:,}")

    # -------------------------
    # Diffusion schedule
    # -------------------------
    betas, alphas, alphas_bar = make_beta_schedule(args.diffusion_steps)
    betas      = betas.to(device)
    alphas     = alphas.to(device)
    alphas_bar = alphas_bar.to(device)

    # -------------------------
    # Optimizer
    # -------------------------
    optimizer   = torch.optim.Adam(model.parameters(), lr=args.lr)
    num_batches = max(1, N // args.batch_size)

    print(f"\n🚀 Training")
    print(f"   epochs:          {args.num_epochs}")
    print(f"   batch_size:      {args.batch_size}")
    print(f"   batches/epoch:   {num_batches}")
    print(f"   diffusion_steps: {args.diffusion_steps}")
    print(f"   lr:              {args.lr}")
    print(f"   output:          {args.output_path}\n")

    best_loss = float('inf')

    for epoch in range(args.num_epochs):

        # Shuffle
        perm      = torch.randperm(N)
        conds_t   = conds_t[perm]
        actions_t = actions_t[perm]

        epoch_loss = 0.0

        for b in range(num_batches):
            start = b * args.batch_size
            end   = min(N, (b + 1) * args.batch_size)

            x0   = actions_t[start:end].to(device)        # [B, H, Da]
            cond = conds_t[start:end, 0, :].to(device)    # [B, Dc] — condition on first step

            B = x0.shape[0]

            t   = torch.randint(0, args.diffusion_steps, (B,), device=device)
            eps = torch.randn_like(x0)

            x_t      = q_sample(x0, t, eps, alphas_bar)
            eps_pred = model(x_t, t, cond)

            loss = F.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * B

        epoch_loss /= N

        if epoch_loss < best_loss:
            best_loss = epoch_loss

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:4d}/{args.num_epochs} | "
                  f"loss: {epoch_loss:.6f} | best: {best_loss:.6f}")

    # -------------------------
    # Save checkpoint — self-describing
    # -------------------------
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    torch.save(
        {
            # Model
            "model_state_dict": model.state_dict(),
            # Architecture
            "horizon":          H,
            "action_dim":       Da,
            "cond_dim":         Dc,
            "hidden_dim":       args.hidden_dim,
            "diffusion_steps":  args.diffusion_steps,
            # Normalization
            "act_mean":         act_mean,
            "act_std":          act_std,
            # Task + conditioning metadata
            "obs_keys":         obs_keys,
            "cond_mode":        cond_mode,   # saved so eval scripts know what to build
            "state_dim":        state_dim,
            "task":             task,
            # Training info
            "n_samples":        N,
            "num_epochs":       args.num_epochs,
            "final_loss":       epoch_loss,
            "best_loss":        best_loss,
        },
        args.output_path,
    )

    print(f"\n💾 Saved to: {args.output_path}")
    print(f"   cond_mode:  {cond_mode}")
    print(f"   cond_dim:   {Dc}")
    print(f"   final_loss: {epoch_loss:.6f}")
    print(f"   best_loss:  {best_loss:.6f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())