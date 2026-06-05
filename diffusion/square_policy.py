"""Custom diffusion policy components for Robomimic Square.

This module is intentionally self-contained so the Square diffusion-policy
trainer and evaluator do not depend on robomimic's diffusion-policy stack.

Key fixes over the original:
  1. Cosine beta schedule (Nichol & Dhariwal 2021) instead of linear.
  2. Correct DDPM posterior q(x_{t-1} | x_t, x0) in sample_action_sequence.
  3. Residual MLP with FiLM conditioning instead of additive obs+time fusion.
  4. Default obs_horizon reduced to 2 (matching Chi et al. 2023).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset


DEFAULT_OBS_KEYS = [
    "robot0_eef_pos",
    "robot0_eef_quat",
    "robot0_gripper_qpos",
    "object",
]


# ---------------------------------------------------------------------------
# Beta schedule
# ---------------------------------------------------------------------------

def make_beta_schedule(num_steps: int, beta_start: float = 1e-4, beta_end: float = 2e-2):
    """Cosine beta schedule (Nichol & Dhariwal 2021 / squaredcos_cap_v2).

    The linear schedule used previously distributed noise poorly and hurt
    action quality.  The cosine schedule is what Chi et al. 2023 use.
    beta_start / beta_end are ignored but kept for API compatibility.
    """
    steps = num_steps + 1
    t = torch.linspace(0, num_steps, steps)
    # f(t) = cos((t/T + s) / (1 + s) * pi/2)^2,  s = 0.008
    f = torch.cos((t / num_steps + 0.008) / 1.008 * math.pi / 2.0) ** 2
    alphas_bar_full = f / f[0]                          # shape: (num_steps+1,)
    betas = torch.clamp(
        1.0 - alphas_bar_full[1:] / alphas_bar_full[:-1], min=0.0, max=0.999
    )
    alphas = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)           # shape: (num_steps,)
    return betas, alphas, alphas_bar


# ---------------------------------------------------------------------------
# Forward diffusion (used during training only)
# ---------------------------------------------------------------------------

def q_sample(x0: torch.Tensor, t: torch.Tensor, eps: torch.Tensor, alphas_bar: torch.Tensor) -> torch.Tensor:
    """Sample x_t ~ q(x_t | x_0) = sqrt(alpha_bar_t)*x0 + sqrt(1-alpha_bar_t)*eps."""
    a_bar = alphas_bar[t].view(-1, 1, 1).to(x0.device)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half_dim = self.dim // 2
        device = timesteps.device
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half_dim, device=device).float()
            / max(half_dim - 1, 1)
        )
        args = timesteps.float().unsqueeze(-1) * freqs.unsqueeze(0)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = nn.functional.pad(emb, (0, 1))
        return emb


class ResidualBlock(nn.Module):
    """Residual MLP block with FiLM (scale + shift) conditioning."""

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1   = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2   = nn.Linear(dim, dim)
        self.cond_proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm1(x)
        h = torch.relu(self.fc1(h)) * (1.0 + scale) + shift
        h = self.norm2(h)
        h = self.fc2(h)
        return x + h


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SquareTrajectoryDataset(Dataset):
    """Sliding-window Square dataset for a diffusion policy."""

    def __init__(
        self,
        hdf5_path: str,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        obs_keys: Optional[Sequence[str]] = None,
        normalize: bool = True,
    ):
        self.hdf5_path = hdf5_path
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.obs_keys = list(obs_keys or DEFAULT_OBS_KEYS)
        self.normalize = normalize

        self._samples: List[Dict[str, np.ndarray]] = []
        self._load()
        if not self._samples:
            raise RuntimeError(
                f"No training windows found in {hdf5_path}. "
                f"Check obs_horizon={obs_horizon} and action_horizon={action_horizon}."
            )

        self.obs_mean: Optional[np.ndarray] = None
        self.obs_std:  Optional[np.ndarray] = None
        self.action_mean: Optional[np.ndarray] = None
        self.action_std:  Optional[np.ndarray] = None
        if normalize:
            self._fit_normalizer()

    def _load(self) -> None:
        with h5py.File(self.hdf5_path, "r") as handle:
            for demo_key in sorted(handle["data"].keys()):
                demo = handle["data"][demo_key]
                actions = demo["actions"][:].astype(np.float32)
                steps = actions.shape[0]
                if steps < max(self.obs_horizon, self.action_horizon):
                    continue

                obs_seq = np.concatenate(
                    [demo["obs"][key][:].reshape(steps, -1) for key in self.obs_keys],
                    axis=1,
                ).astype(np.float32)

                for start in range(0, steps - self.action_horizon + 1):
                    obs_end   = start + 1
                    obs_start = max(0, obs_end - self.obs_horizon)
                    obs_hist  = obs_seq[obs_start:obs_end]
                    if obs_hist.shape[0] < self.obs_horizon:
                        pad_count = self.obs_horizon - obs_hist.shape[0]
                        pad = np.repeat(obs_hist[:1], pad_count, axis=0)
                        obs_hist = np.concatenate([pad, obs_hist], axis=0)

                    action_seq = actions[start : start + self.action_horizon]
                    self._samples.append({"obs": obs_hist, "action": action_seq})

    def _fit_normalizer(self) -> None:
        obs     = np.stack([s["obs"]    for s in self._samples], axis=0)
        actions = np.stack([s["action"] for s in self._samples], axis=0)
        self.obs_mean    = obs.mean(axis=(0, 1))
        self.obs_std     = obs.std(axis=(0, 1)).clip(1e-6)
        self.action_mean = actions.mean(axis=(0, 1))
        self.action_std  = actions.std(axis=(0, 1)).clip(1e-6)

    @property
    def obs_dim(self) -> int:
        return int(self._samples[0]["obs"].shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self._samples[0]["action"].shape[-1])

    def get_normalization_stats(self) -> Dict[str, np.ndarray]:
        return {
            "obs_mean": self.obs_mean,
            "obs_std":  self.obs_std,
            "action_mean": self.action_mean,
            "action_std":  self.action_std,
        }

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        sample = self._samples[index]
        obs    = sample["obs"].copy()
        action = sample["action"].copy()

        if self.normalize:
            obs    = (obs    - self.obs_mean)    / self.obs_std
            action = (action - self.action_mean) / self.action_std

        return {
            "obs":    torch.from_numpy(obs).float(),
            "action": torch.from_numpy(action).float(),
        }
