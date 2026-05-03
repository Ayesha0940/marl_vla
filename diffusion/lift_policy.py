"""Custom diffusion policy components for Robomimic Lift.

This module is intentionally self-contained so the Lift diffusion-policy
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
    """Residual MLP block with FiLM (scale + shift) conditioning.

    FiLM lets the conditioning signal modulate every layer of the denoiser
    instead of being mixed in only at the input, which is the key structural
    feature of the UNet used in Chi et al. 2023.
    """

    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.fc1   = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.fc2   = nn.Linear(dim, dim)
        # Projects conditioning to (scale, shift) pair — FiLM
        self.cond_proj = nn.Linear(cond_dim, dim * 2)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        scale, shift = self.cond_proj(cond).chunk(2, dim=-1)
        h = self.norm1(x)
        h = torch.relu(self.fc1(h)) * (1.0 + scale) + shift
        h = self.norm2(h)
        h = self.fc2(h)
        return x + h


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class LiftDiffusionPolicy(nn.Module):
    """Noise-prediction network for the Lift diffusion policy.

    Architecture:
      - Sinusoidal time embedding.
      - Conditioning encoder: concat(obs_flat, t_emb) → cond vector.
      - Input projection: noisy_action_flat → hidden.
      - N residual blocks, each FiLM-conditioned on cond.
      - Output projection → predicted noise (same shape as action chunk).

    Parameters
    ----------
    obs_dim       : dimensionality of a single observation vector.
    action_dim    : dimensionality of a single action vector.
    obs_horizon   : number of past observations fed to the policy (default 2).
    action_horizon: number of actions predicted per chunk (default 8).
    hidden_dim    : width of all hidden layers (default 256).
    time_emb_dim  : dimensionality of the sinusoidal time embedding (default 128).
    n_layers      : number of residual blocks (default 4).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        obs_horizon: int = 2,
        action_horizon: int = 8,
        hidden_dim: int = 256,
        time_emb_dim: int = 128,
        n_layers: int = 4,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon

        cond_dim = hidden_dim

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)

        # Condition = obs history + time embedding fused by a small MLP
        self.cond_encoder = nn.Sequential(
            nn.Linear(obs_horizon * obs_dim + time_emb_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, cond_dim),
        )

        # Project noisy action sequence into hidden space
        self.input_proj = nn.Linear(action_horizon * action_dim, hidden_dim)

        # Residual blocks with FiLM conditioning
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim, cond_dim) for _ in range(n_layers)]
        )

        # Project back to action space
        self.output_proj = nn.Linear(hidden_dim, action_horizon * action_dim)

    def forward(
        self,
        noisy_actions: torch.Tensor,   # (B, action_horizon, action_dim)
        timesteps: torch.Tensor,        # (B,)
        obs_history: torch.Tensor,      # (B, obs_horizon, obs_dim)
    ) -> torch.Tensor:                  # (B, action_horizon, action_dim)
        B = noisy_actions.shape[0]

        obs_flat = obs_history.reshape(B, -1)                          # (B, obs_horizon*obs_dim)
        t_emb    = self.time_emb(timesteps)                            # (B, time_emb_dim)
        cond     = self.cond_encoder(torch.cat([obs_flat, t_emb], dim=-1))  # (B, cond_dim)

        x = self.input_proj(noisy_actions.reshape(B, -1))              # (B, hidden_dim)
        for block in self.blocks:
            x = block(x, cond)

        return self.output_proj(x).view(B, self.action_horizon, self.action_dim)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LiftTrajectoryDataset(Dataset):
    """Sliding-window Lift dataset for a diffusion policy."""

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


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_action_sequence(
    model: LiftDiffusionPolicy,
    obs_history: np.ndarray,          # (obs_horizon, obs_dim)  — unnormalised
    obs_mean: np.ndarray,
    obs_std: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
    alphas: torch.Tensor,             # (T,)
    alphas_bar: torch.Tensor,         # (T,)
    diffusion_steps: int,
    t_start: Optional[int] = None,
    device: Optional[torch.device] = None,
) -> np.ndarray:                      # (action_horizon, action_dim)
    """Run the DDPM reverse process to produce one action chunk.

    The key fix over the original:  uses the correct DDPM posterior
    q(x_{t-1} | x_t, x_0) instead of the wrong formula
        x = sqrt(alpha)*x0_hat + sqrt(1-alpha)*noise
    which was the primary cause of 0 % success rate.
    """
    model_device = device or next(model.parameters()).device
    if t_start is None:
        t_start = diffusion_steps - 1

    alphas     = alphas.to(model_device)
    alphas_bar = alphas_bar.to(model_device)

    # Normalise observation history
    obs_norm   = (obs_history - obs_mean) / obs_std
    obs_tensor = torch.from_numpy(obs_norm).float().unsqueeze(0).to(model_device)  # (1, T_o, obs_dim)

    # Start from pure Gaussian noise in action space
    x = torch.randn((1, model.action_horizon, model.action_dim), device=model_device)

    for step in reversed(range(t_start + 1)):
        t          = torch.tensor([step], device=model_device)
        alpha      = alphas[step]
        alpha_bar  = alphas_bar[step]
        alpha_bar_prev = alphas_bar[step - 1] if step > 0 else torch.tensor(1.0, device=model_device)

        # Predict noise, then compute x0 estimate
        eps_pred = model(x, t, obs_tensor)
        x0_hat   = (x - torch.sqrt(1.0 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
        x0_hat   = x0_hat.clamp(-1.0, 1.0)   # clip_sample=True (Chi et al. 2023)

        if step > 0:
            # Correct DDPM posterior mean: q(x_{t-1} | x_t, x_0)
            #   mu = sqrt(alpha_bar_{t-1}) * (1-alpha_t)/(1-alpha_bar_t) * x0_hat
            #      + sqrt(alpha_t) * (1-alpha_bar_{t-1})/(1-alpha_bar_t) * x_t
            coef_x0 = torch.sqrt(alpha_bar_prev) * (1.0 - alpha)     / (1.0 - alpha_bar)
            coef_xt = torch.sqrt(alpha)           * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
            mean    = coef_x0 * x0_hat + coef_xt * x

            # Posterior variance: beta_tilde_t = (1-alpha_bar_{t-1})/(1-alpha_bar_t) * beta_t
            variance = (1.0 - alpha_bar_prev) / (1.0 - alpha_bar) * (1.0 - alpha)
            x = mean + torch.sqrt(variance.clamp(min=1e-20)) * torch.randn_like(x)
        else:
            x = x0_hat

    # Denormalise and return
    action = x[0].cpu().numpy() * action_std + action_mean
    return action.astype(np.float32)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def load_lift_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a checkpoint saved by train_diffusion_lift.py."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    required_keys = {
        "model_state_dict", "obs_dim", "action_dim",
        "obs_horizon", "action_horizon", "diffusion_steps",
        "obs_mean", "obs_std", "action_mean", "action_std",
    }
    missing_keys = sorted(required_keys.difference(checkpoint.keys()))
    if missing_keys:
        raise ValueError(
            "Checkpoint is not a custom Lift diffusion-policy checkpoint. "
            f"Missing keys: {missing_keys}. "
            "Use a checkpoint produced by training/lift/train_diffusion_lift.py."
        )

    model = LiftDiffusionPolicy(
        obs_dim       = int(checkpoint["obs_dim"]),
        action_dim    = int(checkpoint["action_dim"]),
        obs_horizon   = int(checkpoint["obs_horizon"]),
        action_horizon= int(checkpoint["action_horizon"]),
        hidden_dim    = int(checkpoint.get("hidden_dim",    256)),
        time_emb_dim  = int(checkpoint.get("time_emb_dim", 128)),
        n_layers      = int(checkpoint.get("n_layers",       6)),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    _, alphas, alphas_bar = make_beta_schedule(int(checkpoint["diffusion_steps"]))
    return model, checkpoint, alphas.to(device), alphas_bar.to(device)