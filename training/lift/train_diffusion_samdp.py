#!/usr/bin/env python3
"""
SA-MDP UNet diffusion policy trainer for Robomimic Lift.

Identical to train_unet_diffusion_lift.py except the loss adds an observation-
robustness regularizer (SA-MDP, Zhang et al. 2020) on top of standard x0
prediction.

Loss:
    loss = loss_bc + kappa * loss_samdp

    loss_bc    = MSE(model(x_t, t, obs),       x0)         # vanilla x0 loss
    loss_samdp = MSE(model(x_t, t, obs_tilde), x0_hat.detach())  # consistency

obs_tilde uses Option C (random sigma per batch, per-dim):
    sigma_max_norm = sigma_max / obs_std   # (obs_dim,) — normalized per-dim
    alpha          = Uniform(0, sigma_max_norm) per sample
    obs_tilde      = obs + alpha * randn_like(obs)

sigma_max defaults to 0.05 (raw state space), matching aug_alpha_s_max from
the joint-denoiser dataset and the dominant eval perturbation level. The
per-dim conversion makes the perturbation isotropic in raw space, key-by-key.

Usage:
    # Default run (kappa=1.0, sigma_max=0.05)
    python training/lift/train_diffusion_samdp.py

    # kappa sweep
    python training/lift/train_diffusion_samdp.py --kappa 0.3 --output_dir checkpoints/lift_samdp_k03
    python training/lift/train_diffusion_samdp.py --kappa 1.0 --output_dir checkpoints/lift_samdp_k10
    python training/lift/train_diffusion_samdp.py --kappa 3.0 --output_dir checkpoints/lift_samdp_k30
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

from diffusion.lift_policy import (
    LiftTrajectoryDataset,
    make_beta_schedule,
    q_sample,
)
from diffusion.lift_policy_unet import UNetDiffusionPolicy, save_unet_checkpoint


# ---------------------------------------------------------------------------
# EMA helpers  (identical to vanilla)
# ---------------------------------------------------------------------------

def get_ema_state_dict(ema_model: AveragedModel) -> dict:
    ema_sd = ema_model.state_dict()
    clean_sd = {}
    for k, v in ema_sd.items():
        if k.startswith("module."):
            clean_sd[k[len("module."):]] = v
    return clean_sd


def save_checkpoint_with_ema(save_fn, path, model, ema_model, dataset, config, epoch, loss):
    original_sd = {k: v.clone() for k, v in model.state_dict().items()}
    ema_sd = get_ema_state_dict(ema_model)
    model.load_state_dict(ema_sd)
    save_fn(path, model, dataset, config, epoch, loss)
    model.load_state_dict(original_sd)


# ---------------------------------------------------------------------------
# SA-MDP loss
# ---------------------------------------------------------------------------

def samdp_x0_loss(
    model,
    action_batch:    torch.Tensor,   # (B, action_horizon, action_dim)  normalized
    obs_batch:       torch.Tensor,   # (B, obs_horizon, obs_dim)         normalized
    alphas_bar:      torch.Tensor,   # (T,)
    num_diffusion_steps: int,
    sigma_max_norm:  torch.Tensor,   # (obs_dim,) — sigma_max / obs_std, per-dim
    kappa:           float,
    device:          torch.device,
) -> tuple[torch.Tensor, float, float]:
    """
    Returns (total_loss, bc_scalar, samdp_scalar).

    One clean forward pass feeds loss_bc and provides the detached target for
    loss_samdp. One perturbed forward pass computes loss_samdp. Both
    gradients flow through a single loss.backward() call.

    sigma_max_norm is (obs_dim,): sigma_max_raw / obs_std per dimension.
    This keeps the perturbation isotropic in raw state space — alpha*randn in
    normalized space corresponds to sigma_max_raw*randn in raw space for every
    key independently. Pre-computed once in train() and passed here.
    """
    B = action_batch.shape[0]
    t   = torch.randint(0, num_diffusion_steps, (B,), device=device)
    eps = torch.randn_like(action_batch)
    x_t = q_sample(action_batch, t, eps, alphas_bar)

    # Clean branch — gradient flows for loss_bc; result detached for loss_samdp target
    x0_hat   = model(x_t, t, obs_batch)
    loss_bc  = F.mse_loss(x0_hat, action_batch)

    # Per-dim perturbation: alpha (B, 1, obs_dim) broadcasts over (B, obs_horizon, obs_dim)
    # alpha ~ U(0, sigma_max_norm) per sample, per dim
    alpha     = torch.rand(B, 1, 1, device=device) * sigma_max_norm
    obs_tilde = obs_batch + alpha * torch.randn_like(obs_batch)

    x0_hat_tilde = model(x_t, t, obs_tilde)
    loss_samdp   = F.mse_loss(x0_hat_tilde, x0_hat.detach())

    total = loss_bc + kappa * loss_samdp
    return total, loss_bc.item(), loss_samdp.item()


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def default_config() -> dict:
    return {
        "algo_name": "diffusion_samdp",
        "task": "lift",
        "train": {
            "data":        os.path.join(PROJECT_ROOT, "datasets/lift/ph/low_dim_v141.hdf5"),
            "output_dir":  os.path.join(PROJECT_ROOT, "checkpoints/lift_samdp_unet"),
            "num_epochs":  4500,
            "batch_size":  256,
            "num_workers": 2,
            "log_freq":    10,
        },
        "model": {
            "obs_keys": [
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
                "object",
            ],
            "obs_horizon":    2,
            "action_horizon": 16,
            "down_dims":      [256, 512, 1024],
            "kernel_size":    5,
            "n_groups":       8,
            "time_emb_dim":   256,
        },
        "diffusion": {
            "num_steps": 100,
        },
        "optimizer": {
            "lr":           1e-4,
            "weight_decay": 1e-5,
        },
        "samdp": {
            "kappa":     1.0,
            "sigma_max": 0.05,   # raw state space; matches aug_alpha_s_max in dataset.py
        },
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SA-MDP UNet diffusion policy trainer — Lift")
    p.add_argument("--config",      type=str,   default=None)
    p.add_argument("--num_epochs",  type=int,   default=None)
    p.add_argument("--batch_size",  type=int,   default=None)
    p.add_argument("--lr",          type=float, default=None)
    p.add_argument("--output_dir",  type=str,   default=None)
    p.add_argument("--device",      type=str,   default="auto")
    p.add_argument("--seed",        type=int,   default=0)
    p.add_argument(
        "--kappa",     type=float, default=None,
        help="SA-MDP regularization weight (default 1.0). Sweep: 0.3 / 1.0 / 3.0",
    )
    p.add_argument(
        "--sigma_max", type=float, default=None,
        help="Max obs perturbation in raw state space. "
             "Converted per-dim to normalized space via sigma_max / obs_std internally. "
             "Default 0.05 matches aug_alpha_s_max in dataset.py.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(config: dict, seed: int = 0, device_arg: str = "auto") -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(
        "cuda" if device_arg == "auto" and torch.cuda.is_available() else device_arg
    )
    print(f"Device: {device}")

    train_cfg     = config["train"]
    model_cfg     = config["model"]
    diffusion_cfg = config["diffusion"]
    optimizer_cfg = config["optimizer"]
    samdp_cfg     = config["samdp"]

    kappa     = float(samdp_cfg["kappa"])
    sigma_max = float(samdp_cfg["sigma_max"])

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

    print(f"Backbone: unet_samdp | params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"SA-MDP: kappa={kappa}  sigma_max={sigma_max} (raw space, per-dim conversion)")

    # Checkpoint save — patches in SA-MDP metadata on top of save_unet_checkpoint
    def save_fn(path, m, ds, cfg, ep, lo):
        save_unet_checkpoint(path, m, ds, cfg, ep, lo)
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ckpt["prediction_type"] = "x0"
        ckpt["backbone"]        = "unet_samdp"
        ckpt["samdp_kappa"]        = kappa
        ckpt["samdp_sigma_max"]    = sigma_max
        ckpt["samdp_sigma_space"]  = "normalized"
        torch.save(ckpt, path)

    # Per-dim sigma: convert raw sigma_max to normalized space per obs dimension
    obs_std_tensor = torch.from_numpy(dataset.obs_std).float().to(device)  # (obs_dim,)
    sigma_max_norm = sigma_max / obs_std_tensor                             # (obs_dim,)

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
        epoch_loss = epoch_bc = epoch_samdp = 0.0
        n_batches = 0

        for batch in loader:
            obs_batch    = batch["obs"].to(device)
            action_batch = batch["action"].to(device)

            loss, bc_item, samdp_item = samdp_x0_loss(
                model, action_batch, obs_batch,
                alphas_bar_dev, num_steps,
                sigma_max_norm, kappa, device,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ema_model.update_parameters(model)

            epoch_loss  += float(loss.item())
            epoch_bc    += bc_item
            epoch_samdp += samdp_item
            n_batches   += 1

        scheduler.step()
        n = max(n_batches, 1)
        epoch_loss  /= n
        epoch_bc    /= n
        epoch_samdp /= n

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
                  f"loss={epoch_loss:.6f} | bc={epoch_bc:.6f} | "
                  f"samdp={epoch_samdp:.6f} | best={best_loss:.6f}")

    save_fn(final_path, model, dataset, config, int(train_cfg["num_epochs"]), epoch_loss)
    print(f"\nSaved best (EMA):  {best_path}")
    print(f"Saved final (raw): {final_path}")


def main():
    args   = parse_args()
    config = default_config()
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    if args.num_epochs:  config["train"]["num_epochs"]    = args.num_epochs
    if args.batch_size:  config["train"]["batch_size"]    = args.batch_size
    if args.lr:          config["optimizer"]["lr"]        = args.lr
    if args.output_dir:  config["train"]["output_dir"]    = args.output_dir
    if args.kappa:       config["samdp"]["kappa"]         = args.kappa
    if args.sigma_max:   config["samdp"]["sigma_max"]     = args.sigma_max

    train(config, seed=args.seed, device_arg=args.device)


if __name__ == "__main__":
    main()
