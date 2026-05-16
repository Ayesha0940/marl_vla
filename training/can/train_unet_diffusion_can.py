#!/usr/bin/env python3
"""Train a UNet diffusion policy for Robomimic Can (x0-prediction).

Key design choices:
  - x0 prediction: network predicts clean actions directly; no SNR reweighting needed
  - EMA with decay 0.9999 for stable best_model.pt
  - CosineAnnealingLR; grad clipping at 1.0
  - action_horizon=8 (matches working MLP setup; exec_horizon=2 at eval)

Usage:
    python training/can/train_unet_diffusion_can.py
    python training/can/train_unet_diffusion_can.py --num_epochs 6000 --lr 1e-4
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from diffusion.can_policy import (
    CanTrajectoryDataset,
    make_beta_schedule,
    q_sample,
)
from diffusion.can_policy_unet import UNetDiffusionPolicy, save_unet_checkpoint


# -----------------------------------------------------------------------
# EMA helpers
# -----------------------------------------------------------------------

def get_ema_state_dict(ema_model: AveragedModel) -> dict:
    ema_sd = ema_model.state_dict()
    return {k[len("module."):]: v for k, v in ema_sd.items() if k.startswith("module.")}


def save_checkpoint_with_ema(path, model, ema_model, dataset, config, epoch, loss):
    original_sd = {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(get_ema_state_dict(ema_model))
    save_unet_checkpoint(path, model, dataset, config, epoch, loss)
    # Tag prediction_type so eval script knows
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    ckpt["prediction_type"] = "x0"
    torch.save(ckpt, path)
    model.load_state_dict(original_sd)


# -----------------------------------------------------------------------
# x0-prediction loss
# -----------------------------------------------------------------------

def x0_prediction_loss(
    model,
    action_batch: torch.Tensor,   # (B, H, Da) clean, normalized
    obs_batch: torch.Tensor,       # (B, obs_H, obs_dim)
    alphas_bar: torch.Tensor,      # (T,)
    num_diffusion_steps: int,
) -> torch.Tensor:
    B = action_batch.shape[0]
    device = action_batch.device

    t = torch.randint(0, num_diffusion_steps, (B,), device=device)
    eps = torch.randn_like(action_batch)
    x_t = q_sample(action_batch, t, eps, alphas_bar)
    x0_hat = model(x_t, t, obs_batch)
    return F.mse_loss(x0_hat, action_batch)


# -----------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------

def default_config() -> dict:
    return {
        "algo_name": "diffusion",
        "task": "can",
        "train": {
            "data": os.path.join(PROJECT_ROOT, "datasets/can/ph/low_dim_v141.hdf5"),
            "output_dir": os.path.join(PROJECT_ROOT, "checkpoints/can_diffusion_policy_unet"),
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
            "action_horizon": 8,
            "down_dims": [256, 512, 1024],
            "kernel_size": 5,
            "n_groups": 8,
            "time_emb_dim": 256,
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
    p = argparse.ArgumentParser(description="Train UNet diffusion policy for Can")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--num_epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--action_horizon", type=int, default=None)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=0)
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

    os.makedirs(train_cfg["output_dir"], exist_ok=True)

    dataset = CanTrajectoryDataset(
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

    n_params = sum(p.numel() for p in model.parameters())
    print(f"UNet params: {n_params:,}  |  dataset: {len(dataset)} samples")
    print(f"action_horizon={model_cfg['action_horizon']}  obs_horizon={model_cfg['obs_horizon']}")
    print("Prediction type: x0 (sample prediction)")

    num_steps = int(diffusion_cfg["num_steps"])
    _, alphas, alphas_bar = make_beta_schedule(num_steps)
    alphas_bar_dev = alphas_bar.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr           = float(optimizer_cfg["lr"]),
        weight_decay = float(optimizer_cfg.get("weight_decay", 1e-5)),
    )
    ema_model = AveragedModel(
        model,
        multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=int(train_cfg["num_epochs"]), eta_min=1e-6,
    )

    print(f"\nTraining for {int(train_cfg['num_epochs'])} epochs")

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

            loss = x0_prediction_loss(model, action_batch, obs_batch, alphas_bar_dev, num_steps)

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
            save_checkpoint_with_ema(epoch_path, model, ema_model, dataset, config, epoch, epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint_with_ema(best_path, model, ema_model, dataset, config, epoch, epoch_loss)

        if epoch == 1 or epoch % int(train_cfg["log_freq"]) == 0:
            print(f"Epoch {epoch:4d}/{int(train_cfg['num_epochs'])} | "
                  f"loss={epoch_loss:.6f} | best={best_loss:.6f}")

    # Save final raw (non-EMA) weights
    save_unet_checkpoint(final_path, model, dataset, config,
                         int(train_cfg["num_epochs"]), epoch_loss)
    ckpt = torch.load(final_path, map_location="cpu", weights_only=False)
    ckpt["prediction_type"] = "x0"
    torch.save(ckpt, final_path)

    print(f"\nSaved best (EMA):  {best_path}")
    print(f"Saved final (raw): {final_path}")


def main():
    args   = parse_args()
    config = default_config()
    if args.config:
        with open(args.config) as f:
            config = json.load(f)

    if args.num_epochs:    config["train"]["num_epochs"]       = args.num_epochs
    if args.batch_size:    config["train"]["batch_size"]       = args.batch_size
    if args.lr:            config["optimizer"]["lr"]           = args.lr
    if args.action_horizon: config["model"]["action_horizon"]  = args.action_horizon

    train(config, seed=args.seed, device_arg=args.device)


if __name__ == "__main__":
    main()
