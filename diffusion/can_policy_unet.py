"""DiffusionPolicy-C: 1D Temporal UNet backbone for Robomimic Can.

Implements the convolutional (CNN) backbone from Chi et al. 2023.
Architecture is task-agnostic; only the checkpoint loader references Can.
"""

from __future__ import annotations

import math
from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = timesteps.device
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device).float()
            / max(half - 1, 1)
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class ConditioningEncoder(nn.Module):
    def __init__(self, obs_horizon: int, obs_dim: int, time_emb_dim: int, cond_dim: int):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.mlp = nn.Sequential(
            nn.Linear(obs_horizon * obs_dim + time_emb_dim, cond_dim * 2),
            nn.Mish(),
            nn.Linear(cond_dim * 2, cond_dim),
        )

    def forward(self, obs_history: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        B = obs_history.shape[0]
        obs_flat = obs_history.reshape(B, -1)
        t_emb = self.time_emb(timesteps)
        return self.mlp(torch.cat([obs_flat, t_emb], dim=-1))


def _valid_groups(n_groups: int, channels: int) -> int:
    """Return largest divisor of channels that is <= n_groups."""
    g = min(n_groups, channels)
    while channels % g != 0:
        g -= 1
    return g


class Conv1dBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(_valid_groups(n_groups, out_channels), out_channels),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConditionalResidualBlock1D(nn.Module):
    """Conv residual block with FiLM conditioning."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int,
                 kernel_size: int = 3, n_groups: int = 8):
        super().__init__()
        self.conv1 = Conv1dBlock(in_channels,  out_channels, kernel_size, n_groups)
        self.conv2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        self.cond_proj = nn.Linear(cond_dim, out_channels * 2)
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        scale, shift = self.cond_proj(cond).unsqueeze(-1).chunk(2, dim=1)
        out = out * (1.0 + scale) + shift
        out = self.conv2(out)
        return out + self.residual_conv(x)


# ---------------------------------------------------------------------------
# 1D UNet
# ---------------------------------------------------------------------------

class UNetDiffusionPolicy(nn.Module):
    """DiffusionPolicy-C: 1D temporal UNet for Can task.

    Channel layout example with down_dims=[256,512,1024], action_dim=7:

      all_dims = [7, 256, 512, 1024]
      in_out   = [(7,256), (256,512), (512,1024)]

      Encoder saves skips with channels: [256, 512, 1024]

      Decoder (processes skips in reverse: [1024, 512, 256]):
        step 0: cat(bottleneck_1024, skip_1024) = 2048 → block → 1024 → upsample
        step 1: cat(x_1024,         skip_512)  = 1536 → block → 512  → upsample
        step 2: cat(x_512,          skip_256)  = 768  → block → 256  → no upsample

      final_conv: 256 → 7 (action_dim)
    """

    def __init__(
        self,
        obs_dim:        int,
        action_dim:     int,
        obs_horizon:    int = 2,
        action_horizon: int = 8,
        down_dims:      List[int] = None,
        kernel_size:    int = 5,
        n_groups:       int = 8,
        time_emb_dim:   int = 256,
    ):
        super().__init__()
        if down_dims is None:
            down_dims = [256, 512, 1024]

        self.obs_dim        = obs_dim
        self.action_dim     = action_dim
        self.obs_horizon    = obs_horizon
        self.action_horizon = action_horizon
        self.down_dims      = down_dims

        cond_dim = down_dims[0]

        self.cond_encoder = ConditioningEncoder(
            obs_horizon=obs_horizon,
            obs_dim=obs_dim,
            time_emb_dim=time_emb_dim,
            cond_dim=cond_dim,
        )

        all_dims = [action_dim] + list(down_dims)
        in_out   = list(zip(all_dims[:-1], all_dims[1:]))
        n_levels = len(in_out)

        # ---- Encoder ----
        self.down_blocks  = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for idx, (dim_in, dim_out) in enumerate(in_out):
            self.down_blocks.append(nn.ModuleList([
                ConditionalResidualBlock1D(dim_in,  dim_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_out, dim_out, cond_dim, kernel_size, n_groups),
            ]))
            self.down_samples.append(
                nn.Identity() if idx == n_levels - 1
                else nn.Conv1d(dim_out, dim_out, 3, stride=2, padding=1)
            )

        # ---- Bottleneck ----
        mid_dim = down_dims[-1]
        self.mid_block1 = ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups)
        self.mid_block2 = ConditionalResidualBlock1D(mid_dim, mid_dim, cond_dim, kernel_size, n_groups)

        # ---- Decoder ----
        self.up_blocks  = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        reversed_in_out = list(reversed(in_out))
        for idx, (dim_in, dim_out) in enumerate(reversed_in_out):
            x_in_channels   = reversed_in_out[idx - 1][1] if idx > 0 else down_dims[-1]
            skip_channels   = dim_out
            concat_channels = x_in_channels + skip_channels
            self.up_blocks.append(nn.ModuleList([
                ConditionalResidualBlock1D(concat_channels, dim_out, cond_dim, kernel_size, n_groups),
                ConditionalResidualBlock1D(dim_out,         dim_out, cond_dim, kernel_size, n_groups),
            ]))
            self.up_samples.append(
                nn.Identity() if idx == n_levels - 1
                else nn.ConvTranspose1d(dim_out, dim_out, 4, stride=2, padding=1)
            )

        self.final_conv = nn.Sequential(
            Conv1dBlock(down_dims[0], down_dims[0], kernel_size, n_groups),
            nn.Conv1d(down_dims[0], action_dim, 1),
        )

    def forward(
        self,
        noisy_actions: torch.Tensor,   # (B, action_horizon, action_dim)
        timesteps:     torch.Tensor,   # (B,)
        obs_history:   torch.Tensor,   # (B, obs_horizon, obs_dim)
    ) -> torch.Tensor:                 # (B, action_horizon, action_dim)

        cond = self.cond_encoder(obs_history, timesteps)
        x    = noisy_actions.transpose(1, 2)               # (B, action_dim, T)

        skips = []
        for (block1, block2), downsample in zip(self.down_blocks, self.down_samples):
            x = block1(x, cond)
            x = block2(x, cond)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond)
        x = self.mid_block2(x, cond)

        for (block1, block2), upsample, skip in zip(
            self.up_blocks, self.up_samples, skips[::-1]
        ):
            x = x[..., :skip.shape[-1]]
            x = torch.cat([x, skip], dim=1)
            x = block1(x, cond)
            x = block2(x, cond)
            x = upsample(x)

        x = self.final_conv(x)
        return x.transpose(1, 2)                           # (B, action_horizon, action_dim)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_unet_checkpoint(
    path: str,
    model: UNetDiffusionPolicy,
    dataset,
    config: dict,
    epoch: int,
    loss: float,
) -> None:
    model_cfg     = config["model"]
    diffusion_cfg = config["diffusion"]
    torch.save(
        {
            "backbone":         "unet",
            "model_state_dict": model.state_dict(),
            "obs_dim":          model.obs_dim,
            "action_dim":       model.action_dim,
            "obs_horizon":      model.obs_horizon,
            "action_horizon":   model.action_horizon,
            "down_dims":        model.down_dims,
            "kernel_size":      int(model_cfg.get("kernel_size",   5)),
            "n_groups":         int(model_cfg.get("n_groups",       8)),
            "time_emb_dim":     int(model_cfg.get("time_emb_dim", 256)),
            "diffusion_steps":  int(diffusion_cfg["num_steps"]),
            "beta_start":       float(diffusion_cfg.get("beta_start", 1e-4)),
            "beta_end":         float(diffusion_cfg.get("beta_end",   2e-2)),
            "obs_keys":         list(model_cfg.get("obs_keys", [])),
            "obs_mean":         dataset.obs_mean,
            "obs_std":          dataset.obs_std,
            "action_mean":      dataset.action_mean,
            "action_std":       dataset.action_std,
            "config":           config,
            "epoch":            epoch,
            "loss":             loss,
        },
        path,
    )


@torch.no_grad()
def sample_action_sequence_x0(
    model,
    obs_history: np.ndarray,      # (obs_horizon, obs_dim) unnormalized
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
    """DDPM reverse process with x0 prediction for the UNet policy."""
    model_device = device or next(model.parameters()).device
    if t_start is None:
        t_start = diffusion_steps - 1

    alphas     = alphas.to(model_device)
    alphas_bar = alphas_bar.to(model_device)

    obs_norm   = (obs_history - obs_mean) / obs_std
    obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0).to(model_device)

    x = torch.randn((1, model.action_horizon, model.action_dim), device=model_device)

    for step in reversed(range(t_start + 1)):
        t_tensor       = torch.tensor([step], device=model_device)
        alpha          = alphas[step]
        alpha_bar      = alphas_bar[step]
        alpha_bar_prev = alphas_bar[step - 1] if step > 0 else torch.tensor(1.0, device=model_device)

        x0_hat = model(x, t_tensor, obs_tensor)

        if step > 0:
            coef_x0 = torch.sqrt(alpha_bar_prev) * (1.0 - alpha) / (1.0 - alpha_bar)
            coef_xt = torch.sqrt(alpha) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            mean    = coef_x0 * x0_hat + coef_xt * x
            variance = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * (1.0 - alpha)
            x = mean + torch.sqrt(variance.clamp(min=1e-20)) * torch.randn_like(x)
        else:
            x = x0_hat

    action = x[0].cpu().numpy() * action_std + action_mean
    return action.astype(np.float32)


def load_unet_checkpoint(checkpoint_path: str, device: torch.device):
    from diffusion.can_policy import make_beta_schedule

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    required = {
        "model_state_dict", "obs_dim", "action_dim",
        "obs_horizon", "action_horizon", "diffusion_steps",
        "obs_mean", "obs_std", "action_mean", "action_std",
    }
    missing = sorted(required.difference(ckpt.keys()))
    if missing:
        raise ValueError(f"Not a UNet checkpoint. Missing keys: {missing}.")

    model = UNetDiffusionPolicy(
        obs_dim        = int(ckpt["obs_dim"]),
        action_dim     = int(ckpt["action_dim"]),
        obs_horizon    = int(ckpt["obs_horizon"]),
        action_horizon = int(ckpt["action_horizon"]),
        down_dims      = list(ckpt.get("down_dims",    [256, 512, 1024])),
        kernel_size    = int(ckpt.get("kernel_size",   5)),
        n_groups       = int(ckpt.get("n_groups",      8)),
        time_emb_dim   = int(ckpt.get("time_emb_dim", 256)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, alphas, alphas_bar = make_beta_schedule(int(ckpt["diffusion_steps"]))
    return model, ckpt, alphas.to(device), alphas_bar.to(device)
