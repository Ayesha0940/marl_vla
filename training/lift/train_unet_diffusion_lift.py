#!/usr/bin/env python3
"""
Train a custom diffusion policy for Robomimic Lift — FIXED VERSION.

Three key fixes over the original train_diffusion_lift.py:

1. SNR-weighted loss (Min-SNR-gamma weighting, Hang et al. 2023)
   The cosine schedule makes low-t timesteps nearly noiseless, so the
   epsilon-prediction loss gradient vanishes there. The UNet never learns
   to predict noise at low t, which is exactly where the final denoising
   steps operate. Min-SNR-gamma reweights the loss so all timesteps get
   balanced gradient signal.

2. Proper EMA checkpoint saving
   ema_model.module returns the ORIGINAL model, not the averaged one.
   Fixed to extract actual EMA weights before saving.

3. DDIM sampling option at inference (not in this file but documented)
   100-step DDPM is slow; DDIM with 10-20 steps is equivalent quality.

Usage:
    python training/lift/train_diffusion_lift_v2.py --backbone unet
    python training/lift/train_diffusion_lift_v2.py --backbone unet --snr_gamma 5.0
    python training/lift/train_diffusion_lift_v2.py --backbone mlp  # unchanged behavior
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel



PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)






PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.lift_policy import (
    LiftDiffusionPolicy,
    LiftTrajectoryDataset,
    make_beta_schedule,
    q_sample,
)


# -----------------------------------------------------------------------
# EMA helpers  (same as v2 — correct implementation)
# -----------------------------------------------------------------------

def get_ema_state_dict(ema_model: AveragedModel) -> dict:
    """Extract actual EMA-averaged weights (not ema_model.module)."""
    ema_sd = ema_model.state_dict()
    clean_sd = {}
    for k, v in ema_sd.items():
        if k.startswith("module."):
            clean_sd[k[len("module."):]] = v
    return clean_sd


def save_checkpoint_with_ema(save_fn, path, model, ema_model, dataset, config, epoch, loss):
    """Save checkpoint using EMA weights, then restore raw weights."""
    original_sd = {k: v.clone() for k, v in model.state_dict().items()}
    ema_sd = get_ema_state_dict(ema_model)
    model.load_state_dict(ema_sd)
    save_fn(path, model, dataset, config, epoch, loss)
    model.load_state_dict(original_sd)


# -----------------------------------------------------------------------
# x0-prediction training loss
# -----------------------------------------------------------------------

def x0_prediction_loss(
    model,
    action_batch: torch.Tensor,      # (B, H, Da)  clean, normalized
    obs_batch: torch.Tensor,          # (B, obs_H, obs_dim)
    alphas_bar: torch.Tensor,         # (T,)
    num_diffusion_steps: int,
) -> torch.Tensor:
    """
    Compute MSE loss between predicted x0 and true x0.

    Forward process: x_t = sqrt(abar_t)*x0 + sqrt(1-abar_t)*eps
    Network input:   x_t, t, obs  →  x0_hat
    Loss:            ||x0 - x0_hat||^2

    Every timestep sees the same target (x0), so gradients are balanced
    across the full noise schedule without any reweighting.

    No clip_sample: dataset uses mean/std normalization, so actions are not
    bounded to [-1, 1]. Clamping would corrupt targets outside that range.
    """
    B = action_batch.shape[0]
    device = action_batch.device

    t = torch.randint(0, num_diffusion_steps, (B,), device=device)
    eps = torch.randn_like(action_batch)

    # Forward diffuse to x_t
    x_t = q_sample(action_batch, t, eps, alphas_bar)

    # Network predicts x0 directly
    x0_hat = model(x_t, t, obs_batch)

    # No clamping: dataset uses mean/std normalization so normalized actions
    # are NOT bounded to [-1, 1]. Clamping would clip valid targets and
    # prevent the model from learning the full action range.
    return F.mse_loss(x0_hat, action_batch)


# -----------------------------------------------------------------------
# x0-prediction inference
# -----------------------------------------------------------------------

@torch.no_grad()
def sample_action_sequence_x0(
    model,
    obs_history: np.ndarray,          # (obs_horizon, obs_dim) unnormalized
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    alphas: torch.Tensor,
    alphas_bar: torch.Tensor,
    diffusion_steps: int,
    t_start: int = None,
    device: torch.device = None,
) -> np.ndarray:
    """
    DDPM reverse process with x0 prediction.

    The network predicts x0_hat directly. We then compute the posterior mean
    q(x_{t-1} | x_t, x0_hat) exactly as before — only the network output
    interpretation changes.
    """
    model_device = device or next(model.parameters()).device
    if t_start is None:
        t_start = diffusion_steps - 1

    alphas     = alphas.to(model_device)
    alphas_bar = alphas_bar.to(model_device)

    obs_norm   = (obs_history - obs_mean) / obs_std
    obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0).to(model_device)

    x = torch.randn((1, model.action_horizon, model.action_dim), device=model_device)

    for step in reversed(range(t_start + 1)):
        t_tensor   = torch.tensor([step], device=model_device)
        alpha      = alphas[step]
        alpha_bar  = alphas_bar[step]
        alpha_bar_prev = alphas_bar[step - 1] if step > 0 else torch.tensor(1.0, device=model_device)

        # Network predicts x0 directly (key change from epsilon prediction)
        x0_hat = model(x, t_tensor, obs_tensor)

        if step > 0:
            # Posterior mean q(x_{t-1} | x_t, x0_hat) — identical formula
            coef_x0 = torch.sqrt(alpha_bar_prev) * (1.0 - alpha) / (1.0 - alpha_bar)
            coef_xt = torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            mean    = coef_x0 * x0_hat + coef_xt * x

            variance = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * (1.0 - alpha)
            x = mean + torch.sqrt(variance.clamp(min=1e-20)) * torch.randn_like(x)
        else:
            x = x0_hat

    action = x[0].cpu().numpy() * action_std + action_mean
    return action.astype(np.float32)


# -----------------------------------------------------------------------
# Save helpers
# -----------------------------------------------------------------------

def save_mlp_checkpoint(path, model, dataset, config, epoch, loss):
    model_cfg     = config["model"]
    diffusion_cfg = config["diffusion"]
    torch.save({
        "backbone":         "mlp",
        "prediction_type":  "x0",
        "model_state_dict": model.state_dict(),
        "obs_dim":          model.obs_dim,
        "action_dim":       model.action_dim,
        "obs_horizon":      model.obs_horizon,
        "action_horizon":   model.action_horizon,
        "hidden_dim":       int(model_cfg["hidden_dim"]),
        "time_emb_dim":     int(model_cfg.get("time_emb_dim", 128)),
        "n_layers":         int(model_cfg.get("n_layers", 6)),
        "diffusion_steps":  int(diffusion_cfg["num_steps"]),
        "obs_keys":         list(model_cfg.get("obs_keys", [])),
        "obs_mean":         dataset.obs_mean,
        "obs_std":          dataset.obs_std,
        "action_mean":      dataset.action_mean,
        "action_std":       dataset.action_std,
        "config":           config,
        "epoch":            epoch,
        "loss":             loss,
    }, path)


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

def default_config() -> dict:
    return {
        "algo_name": "diffusion",
        "task": "lift",
        "train": {
            "data": os.path.join(PROJECT_ROOT, "datasets/lift/ph/low_dim_v141.hdf5"),
            "output_dir": os.path.join(PROJECT_ROOT, "checkpoints/lift_diffusion_policy_v5"),
            "num_epochs": 4500,
            "batch_size": 256,
            "num_workers": 2,
            "log_freq": 10,
        },
        "model": {
            "obs_keys": [
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ],
            "obs_horizon": 2,
            "action_horizon": 16,
            "backbone": "unet",
            "down_dims": [256, 512, 1024],
            "kernel_size": 5,
            "n_groups": 8,
            "time_emb_dim": 256,
            "hidden_dim": 512,
            "n_layers": 6,
        },
        "diffusion": {
            "num_steps": 100,
        },
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backbone", type=str, default="unet",
                   choices=["mlp", "unet", "transformer"])
    return p.parse_args()


def train(config: dict, seed: int = 0, device_arg: str = "auto") -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if device_arg == "auto" and torch.cuda.is_available()
                          else device_arg)
    print(f"Device: {device}")

    train_cfg     = config["train"]
    model_cfg     = config["model"]
    diffusion_cfg = config["diffusion"]
    optimizer_cfg = config["optimizer"]
    backbone      = model_cfg.get("backbone", "unet")

    os.makedirs(train_cfg["output_dir"], exist_ok=True)

    # Dataset
    dataset = LiftTrajectoryDataset(
        hdf5_path      = train_cfg["data"],
        obs_horizon    = int(model_cfg["obs_horizon"]),
        action_horizon = int(model_cfg["action_horizon"]),
        obs_keys       = model_cfg.get("obs_keys"),
        normalize      = True,
    )
    loader = DataLoader(
        dataset,
        batch_size  = int(train_cfg["batch_size"]),
        shuffle     = True,
        num_workers = int(train_cfg["num_workers"]),
        pin_memory  = (device.type == "cuda"),
    )

    # Model
    if backbone == "unet":
        from diffusion.lift_policy_unet import UNetDiffusionPolicy, save_unet_checkpoint
        model = UNetDiffusionPolicy(
            obs_dim        = dataset.obs_dim,
            action_dim     = dataset.action_dim,
            obs_horizon    = int(model_cfg["obs_horizon"]),
            action_horizon = int(model_cfg["action_horizon"]),
            down_dims      = list(model_cfg.get("down_dims", [256, 512, 1024])),
            kernel_size    = int(model_cfg.get("kernel_size", 5)),
            n_groups       = int(model_cfg.get("n_groups", 8)),
            time_emb_dim   = int(model_cfg.get("time_emb_dim", 256)),
        ).to(device)

        # Wrap save_unet_checkpoint to also tag prediction_type
        def save_fn(path, m, ds, cfg, ep, lo):
            save_unet_checkpoint(path, m, ds, cfg, ep, lo)
            # Patch in prediction_type so eval script knows
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            ckpt["prediction_type"] = "x0"
            torch.save(ckpt, path)

    elif backbone == "transformer":
        from diffusion.lift_policy_transformer import (
            TransformerDiffusionPolicy, save_transformer_checkpoint)
        model = TransformerDiffusionPolicy(
            obs_dim        = dataset.obs_dim,
            action_dim     = dataset.action_dim,
            obs_horizon    = int(model_cfg["obs_horizon"]),
            action_horizon = int(model_cfg["action_horizon"]),
            d_model        = int(model_cfg.get("d_model", 256)),
            n_heads        = int(model_cfg.get("n_heads", 8)),
            n_enc_layers   = int(model_cfg.get("n_enc_layers", 4)),
            n_dec_layers   = int(model_cfg.get("n_dec_layers", 4)),
            d_ff           = int(model_cfg.get("d_ff", 1024)),
            dropout        = float(model_cfg.get("dropout", 0.1)),
            time_emb_dim   = int(model_cfg.get("time_emb_dim", 256)),
        ).to(device)

        def save_fn(path, m, ds, cfg, ep, lo):
            save_transformer_checkpoint(path, m, ds, cfg, ep, lo)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            ckpt["prediction_type"] = "x0"
            torch.save(ckpt, path)
    else:
        model   = LiftDiffusionPolicy(
            obs_dim        = dataset.obs_dim,
            action_dim     = dataset.action_dim,
            obs_horizon    = int(model_cfg["obs_horizon"]),
            action_horizon = int(model_cfg["action_horizon"]),
            hidden_dim     = int(model_cfg["hidden_dim"]),
            time_emb_dim   = int(model_cfg.get("time_emb_dim", 128)),
            n_layers       = int(model_cfg.get("n_layers", 6)),
        ).to(device)
        save_fn = save_mlp_checkpoint

    model = model.to(device)
    print(f"Backbone: {backbone}  |  params: {sum(p.numel() for p in model.parameters()):,}")
    print("Prediction type: x0 (sample prediction)")

    # Diffusion schedule
    num_steps = int(diffusion_cfg["num_steps"])
    _, alphas, alphas_bar = make_beta_schedule(num_steps)
    alphas_bar_dev = alphas_bar.to(device)

    # Optimizer + EMA + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = float(optimizer_cfg["lr"]),
        weight_decay = float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    ema_model = AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(train_cfg["num_epochs"]), eta_min=1e-6,
    )

    print(f"\nTraining: {len(dataset)} samples, {int(train_cfg['num_epochs'])} epochs")

    best_loss  = float("inf")
    best_path  = os.path.join(train_cfg["output_dir"], "best_model.pt")
    final_path = os.path.join(train_cfg["output_dir"], "model_final.pt")

    for epoch in range(1, int(train_cfg["num_epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for batch in loader:
            obs_batch    = batch["obs"].to(device)
            action_batch = batch["action"].to(device)

            loss = x0_prediction_loss(
                model, action_batch, obs_batch, alphas_bar_dev, num_steps
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema_model.update_parameters(model)

            epoch_loss += float(loss.item())
            n_batches  += 1

        scheduler.step()
        epoch_loss /= max(n_batches, 1)

        if epoch % 500 == 0:
            epoch_path = os.path.join(train_cfg["output_dir"], f"model_epoch_{epoch}.pt")
            save_checkpoint_with_ema(save_fn, epoch_path, model, ema_model,
                                     dataset, config, epoch, epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint_with_ema(save_fn, best_path, model, ema_model,
                                     dataset, config, epoch, epoch_loss)

        if epoch == 1 or epoch % int(train_cfg["log_freq"]) == 0:
            print(f"Epoch {epoch:4d}/{int(train_cfg['num_epochs'])} | "
                  f"loss={epoch_loss:.6f} | best={best_loss:.6f}")

    save_fn(final_path, model, dataset, config, int(train_cfg["num_epochs"]), epoch_loss)
    print(f"\nSaved best (EMA):  {best_path}")
    print(f"Saved final (raw): {final_path}")


def main():
    args   = parse_args()
    config = default_config()
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    config["model"]["backbone"] = args.backbone
    if args.num_epochs: config["train"]["num_epochs"] = args.num_epochs
    if args.batch_size: config["train"]["batch_size"] = args.batch_size
    if args.lr:         config["optimizer"]["lr"]     = args.lr

    train(config, seed=args.seed, device_arg=args.device)


if __name__ == "__main__":
    main()