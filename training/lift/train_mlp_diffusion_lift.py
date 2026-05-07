#!/usr/bin/env python3
"""Train a custom diffusion policy for Robomimic Lift."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.swa_utils import AveragedModel, SWALR


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from diffusion.lift_policy import LiftDiffusionPolicy, LiftTrajectoryDataset, make_beta_schedule, q_sample


def default_config() -> dict:
    return {
        "algo_name": "diffusion",
        "task": "lift",
        "experiment": {"name": "lift_diffusion_policy"},
        "train": {
            "data": os.path.join(PROJECT_ROOT, "datasets/lift/ph/low_dim_v141.hdf5"),
            "output_dir": os.path.join(PROJECT_ROOT, "checkpoints/lift_diffusion_policy"),
            "num_epochs": 2000,
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
            "hidden_dim": 512,
            "time_emb_dim": 128,
            "n_layers": 6,
        },
        "diffusion": {
            "num_steps": 100,
            "beta_start": 1e-4,
            "beta_end": 2e-2,
        },
        "optimizer": {
            "lr": 1e-4,
            "weight_decay": 1e-5,
        },
    }


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def save_config(path: str, config: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=4)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a custom diffusion policy on Lift")
    parser.add_argument("--config", type=str, default=None, help="Path to JSON config")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--obs_horizon", type=int, default=None)
    parser.add_argument("--action_horizon", type=int, default=None)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def merge_overrides(config: dict, args: argparse.Namespace) -> dict:
    if args.num_epochs is not None:
        config["train"]["num_epochs"] = args.num_epochs
    if args.batch_size is not None:
        config["train"]["batch_size"] = args.batch_size
    if args.obs_horizon is not None:
        config["model"]["obs_horizon"] = args.obs_horizon
    if args.action_horizon is not None:
        config["model"]["action_horizon"] = args.action_horizon
    if args.hidden_dim is not None:
        config["model"]["hidden_dim"] = args.hidden_dim
    if args.diffusion_steps is not None:
        config["diffusion"]["num_steps"] = args.diffusion_steps
    if args.lr is not None:
        config["optimizer"]["lr"] = args.lr
    return config


def build_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def train(config: dict, seed: int = 0, device_arg: str = "auto") -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = build_device(device_arg)
    print(f"Device: {device}")

    train_cfg = config["train"]
    model_cfg = config["model"]
    diffusion_cfg = config["diffusion"]
    optimizer_cfg = config["optimizer"]

    os.makedirs(train_cfg["output_dir"], exist_ok=True)

    dataset = LiftTrajectoryDataset(
        hdf5_path=train_cfg["data"],
        obs_horizon=int(model_cfg["obs_horizon"]),
        action_horizon=int(model_cfg["action_horizon"]),
        obs_keys=model_cfg.get("obs_keys"),
        normalize=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=(device.type == "cuda"),
    )

    model = LiftDiffusionPolicy(
        obs_dim=dataset.obs_dim,
        action_dim=dataset.action_dim,
        obs_horizon=int(model_cfg["obs_horizon"]),
        action_horizon=int(model_cfg["action_horizon"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        time_emb_dim=int(model_cfg.get("time_emb_dim", 128)),
        n_layers=int(model_cfg.get("n_layers", 6)),
    ).to(device)

    betas, alphas, alphas_bar = make_beta_schedule(
        int(diffusion_cfg["num_steps"]),
        float(diffusion_cfg.get("beta_start", 1e-4)),
        float(diffusion_cfg.get("beta_end", 2e-2)),
    )
    alphas_bar = alphas_bar.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(optimizer_cfg["lr"]),
        weight_decay=float(optimizer_cfg.get("weight_decay", 0.0)),
    )
    
    ema_model = AveragedModel(model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=1000,      # restart every 1000 epochs
        T_mult=1,
        eta_min=1e-6,
    )

    print("Training summary")
    print(f"  samples:          {len(dataset)}")
    print(f"  obs_dim:          {dataset.obs_dim}")
    print(f"  action_dim:       {dataset.action_dim}")
    print(f"  obs_horizon:      {model_cfg['obs_horizon']}")
    print(f"  action_horizon:   {model_cfg['action_horizon']}")
    print(f"  diffusion_steps:   {diffusion_cfg['num_steps']}")
    print(f"  epochs:           {train_cfg['num_epochs']}")
    print(f"  batch_size:       {train_cfg['batch_size']}")
    print(f"  output_dir:       {train_cfg['output_dir']}")

    best_loss = float("inf")
    best_path = os.path.join(train_cfg["output_dir"], "best_model.pt")

    for epoch in range(1, int(train_cfg["num_epochs"]) + 1):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            obs_batch = batch["obs"].to(device)
            action_batch = batch["action"].to(device)
            batch_size = action_batch.shape[0]

            timesteps = torch.randint(0, int(diffusion_cfg["num_steps"]), (batch_size,), device=device)
            noise = torch.randn_like(action_batch)
            noisy_actions = q_sample(action_batch, timesteps, noise, alphas_bar)
            predicted_noise = model(noisy_actions, timesteps, obs_batch)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            ema_model.update_parameters(model)

            epoch_loss += float(loss.item())
            n_batches += 1

        scheduler.step()

        if epoch % 500 == 0:
            print(f"Saving checkpoint at epoch {epoch}...")
            epoch_path = os.path.join(train_cfg["output_dir"], f"model_epoch_{epoch}.pt")
            torch.save({
                "model_state_dict": ema_model.module.state_dict(),
                "obs_dim": dataset.obs_dim,
                "action_dim": dataset.action_dim,
                "obs_horizon": int(model_cfg["obs_horizon"]),
                "action_horizon": int(model_cfg["action_horizon"]),
                "hidden_dim": int(model_cfg["hidden_dim"]),
                "time_emb_dim": int(model_cfg.get("time_emb_dim", 128)),
                "n_layers": int(model_cfg.get("n_layers", 6)),
                "diffusion_steps": int(diffusion_cfg["num_steps"]),
                "beta_start": float(diffusion_cfg.get("beta_start", 1e-4)),
                "beta_end": float(diffusion_cfg.get("beta_end", 2e-2)),
                "obs_keys": list(model_cfg.get("obs_keys", [])),
                "obs_mean": dataset.obs_mean,
                "obs_std": dataset.obs_std,
                "action_mean": dataset.action_mean,
                "action_std": dataset.action_std,
                "config": config,
                "epoch": epoch,
                "loss": epoch_loss,
            }, epoch_path)

        epoch_loss /= max(n_batches, 1)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(
                {
                    "model_state_dict": ema_model.module.state_dict(),
                    "obs_dim": dataset.obs_dim,
                    "action_dim": dataset.action_dim,
                    "obs_horizon": int(model_cfg["obs_horizon"]),
                    "action_horizon": int(model_cfg["action_horizon"]),
                    "hidden_dim": int(model_cfg["hidden_dim"]),
                    "time_emb_dim": int(model_cfg.get("time_emb_dim", 128)),
                    "diffusion_steps": int(diffusion_cfg["num_steps"]),
                    "beta_start": float(diffusion_cfg.get("beta_start", 1e-4)),
                    "beta_end": float(diffusion_cfg.get("beta_end", 2e-2)),
                    "obs_keys": list(model_cfg.get("obs_keys", [])),
                    "obs_mean": dataset.obs_mean,
                    "obs_std": dataset.obs_std,
                    "action_mean": dataset.action_mean,
                    "action_std": dataset.action_std,
                    "config": config,
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "n_layers": int(model_cfg.get("n_layers", 6)),
                },
                best_path,
            )

        if epoch == 1 or epoch % int(train_cfg["log_freq"]) == 0:
            print(f"Epoch {epoch:4d}/{int(train_cfg['num_epochs'])} | loss={epoch_loss:.6f} | best={best_loss:.6f}")

    final_path = os.path.join(train_cfg["output_dir"], "model_final.pt")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "obs_dim": dataset.obs_dim,
            "action_dim": dataset.action_dim,
            "obs_horizon": int(model_cfg["obs_horizon"]),
            "action_horizon": int(model_cfg["action_horizon"]),
            "hidden_dim": int(model_cfg["hidden_dim"]),
            "time_emb_dim": int(model_cfg.get("time_emb_dim", 128)),
            "diffusion_steps": int(diffusion_cfg["num_steps"]),
            "beta_start": float(diffusion_cfg.get("beta_start", 1e-4)),
            "beta_end": float(diffusion_cfg.get("beta_end", 2e-2)),
            "obs_keys": list(model_cfg.get("obs_keys", [])),
            "obs_mean": dataset.obs_mean,
            "obs_std": dataset.obs_std,
            "action_mean": dataset.action_mean,
            "action_std": dataset.action_std,
            "config": config,
            "loss": epoch_loss,
            "n_layers": int(model_cfg.get("n_layers", 6)),
        },
        final_path,
    )

    save_config(os.path.join(train_cfg["output_dir"], "config.json"), config)
    print(f"Saved best checkpoint: {best_path}")
    print(f"Saved final checkpoint: {final_path}")


def main() -> None:
    args = parse_args()

    if args.config:
        config = load_config(args.config)
    else:
        config = default_config()

    config = merge_overrides(config, args)

    if args.config is None:
        config_path = os.path.join(PROJECT_ROOT, "configs", "diffusion_lift.json")
        save_config(config_path, config)
        print(f"Saved config to {config_path}")

    train(config, seed=args.seed, device_arg=args.device)


if __name__ == "__main__":
    main()
