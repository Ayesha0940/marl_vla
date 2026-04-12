"""
Diffusion model for action denoising in robotic manipulation tasks.

Directly ported from MADDPG MPE testbed with the following changes:
- Added get_task_dims() for task-agnostic dim detection from robomimic checkpoints
- Added flatten_obs() for dict obs flattening
- TrajectoryDiffusion, make_beta_schedule, q_sample, diffusion_denoise_action unchanged

Supports: Lift, Can, Square, and any robomimic task automatically.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================
# TASK-AGNOSTIC UTILS
# =========================

def get_task_dims(ckpt_dict):
    """
    Auto-detect obs_keys, state_dim, action_dim from a robomimic checkpoint.
    Works for any task (Lift, Can, Square, etc.) without hardcoding.

    Args:
        ckpt_dict: dict loaded from torch.load(checkpoint_path)

    Returns:
        obs_keys:   list of str, e.g. ['object', 'robot0_eef_pos', ...]
        state_dim:  int, total flattened obs dimension
        action_dim: int, action dimension
    """
    shape_meta = ckpt_dict['shape_metadata']
    obs_keys   = shape_meta['all_obs_keys']
    state_dim  = sum(shape_meta['all_shapes'][k][0] for k in obs_keys)
    action_dim = shape_meta['ac_dim']
    return obs_keys, state_dim, action_dim


def flatten_obs(obs_dict, obs_keys):
    """
    Flatten a robomimic obs dict into a 1D numpy array.
    Only uses the keys the policy was trained on.

    Args:
        obs_dict: dict of {key: np.array}
        obs_keys: list of keys to include (from get_task_dims)

    Returns:
        np.array of shape (state_dim,)
    """
    return np.concatenate([obs_dict[k].flatten() for k in obs_keys]).astype(np.float32)


# =========================
# DIFFUSION MODEL
# =========================

class TrajectoryDiffusion(nn.Module):
    """
    DDPM-style diffusion model for action trajectories.

    Ported directly from MADDPG MPE testbed.
    x:    [B, H, Da]  — action trajectory
    cond: [B, Ds]     — flattened obs (state conditioning)

    For H=1 (single-step denoising), x is [B, 1, Da].
    """
    def __init__(self, horizon, action_dim, cond_dim, hidden_dim=256):
        super().__init__()
        self.horizon    = horizon
        self.action_dim = action_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.net = nn.Sequential(
            nn.Linear(horizon * action_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, horizon * action_dim),
        )

    def forward(self, x_noisy, t, cond):
        """
        Args:
            x_noisy: [B, H, Da]
            t:       [B]  (int, 0..T-1)
            cond:    [B, Ds]
        Returns:
            eps_pred: [B, H, Da]
        """
        B      = x_noisy.shape[0]
        x_flat = x_noisy.reshape(B, -1)

        t_norm  = t.float().unsqueeze(-1) / 1000.0
        t_emb   = self.time_mlp(t_norm)
        c_emb   = self.cond_mlp(cond)
        h       = t_emb + c_emb

        h_cat    = torch.cat([x_flat, h], dim=-1)
        eps_pred = self.net(h_cat)
        eps_pred = eps_pred.view(B, self.horizon, self.action_dim)
        return eps_pred


# =========================
# DIFFUSION SCHEDULE
# =========================

def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    """
    Linear beta schedule. Ported directly from MPE testbed.

    Returns:
        betas:      [T]
        alphas:     [T]
        alphas_bar: [T]  (cumulative product)
    """
    betas      = torch.linspace(beta_start, beta_end, T)
    alphas     = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


def q_sample(x0, t, eps, alphas_bar):
    """
    Forward diffusion q(x_t | x_0). Ported directly from MPE testbed.

    Args:
        x0:         [B, H, Da]
        t:          [B]
        eps:        [B, H, Da]
        alphas_bar: [T]
    Returns:
        x_t: [B, H, Da]
    """
    a_bar = alphas_bar[t].view(-1, 1, 1).to(x0.device)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps


# =========================
# MODEL LOADING / SAVING
# =========================

# Global state — mirrors MPE testbed pattern
DIFFUSION_MODEL  = None
DIFFUSION_CONSTS = {}


def load_diffusion_model(model_path):
    """
    Load a trained diffusion model from disk into global state.
    Ported from MPE testbed; extended to store obs_keys and task name.

    Args:
        model_path: path to .pt file saved by train_diffusion.py
    """
    global DIFFUSION_MODEL, DIFFUSION_CONSTS

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)

    model = TrajectoryDiffusion(
        horizon    = ckpt["horizon"],
        action_dim = ckpt["action_dim"],
        cond_dim   = ckpt["cond_dim"],
        hidden_dim = 256,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    betas, alphas, alphas_bar = make_beta_schedule(ckpt["diffusion_steps"])

    DIFFUSION_MODEL  = model
    DIFFUSION_CONSTS = {
        "betas":      betas,
        "alphas":     alphas,
        "alphas_bar": alphas_bar,
        "act_mean":   ckpt["act_mean"],
        "act_std":    ckpt["act_std"],
        "T":          ckpt["diffusion_steps"],
        "H":          ckpt["horizon"],
        "obs_keys":   ckpt["obs_keys"],   # added vs MPE version
        "task":       ckpt.get("task", "unknown"),
    }

    print(f"[Diffusion] Loaded model | task={DIFFUSION_CONSTS['task']} | "
          f"action_dim={ckpt['action_dim']} | cond_dim={ckpt['cond_dim']} | "
          f"obs_keys={ckpt['obs_keys']}")


def get_diffusion_obs_keys():
    """Return obs_keys from loaded diffusion model. Call after load_diffusion_model()."""
    return DIFFUSION_CONSTS["obs_keys"]


# =========================
# INFERENCE
# =========================

@torch.no_grad()
def diffusion_denoise_action(noisy_action_vec, state_vec, t_start=40):
    """
    Denoise a single action vector using reverse diffusion.
    Ported directly from MPE testbed. No changes needed.

    Args:
        noisy_action_vec: np.array [Da]   — action with noise applied
        state_vec:        np.array [Ds]   — flattened obs (use flatten_obs())
        t_start:          int             — reverse diffusion start step (controls denoising strength)

    Returns:
        clean_action_vec: np.array [Da]
    """
    model = DIFFUSION_MODEL
    C     = DIFFUSION_CONSTS

    H          = C["H"]
    alphas     = C["alphas"]
    alphas_bar = C["alphas_bar"]

    # Normalize noisy action
    a = torch.from_numpy(noisy_action_vec).float()
    a = (a - C["act_mean"][0, 0]) / C["act_std"][0, 0]

    # Build x_t — place noisy action at t=0 of trajectory
    x    = torch.zeros((1, H, a.shape[0]))
    x[0, 0] = a

    cond = torch.from_numpy(state_vec).float().unsqueeze(0)  # [1, Ds]

    # Reverse diffusion
    for t in reversed(range(t_start + 1)):
        t_tensor = torch.tensor([t])
        eps_pred = model(x, t_tensor, cond)

        alpha     = alphas[t]
        alpha_bar = alphas_bar[t]

        x0_hat = (x - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)

        if t > 0:
            noise = torch.randn_like(x)
            x     = torch.sqrt(alpha) * x0_hat + torch.sqrt(1 - alpha) * noise
        else:
            x = x0_hat

    # Unnormalize
    clean = x[0, 0] * C["act_std"][0, 0] + C["act_mean"][0, 0]
    return clean.numpy()