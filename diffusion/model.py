"""
Diffusion model for action denoising in robotic manipulation tasks.

Directly ported from MADDPG MPE testbed with the following changes:
- Added get_task_dims() for task-agnostic dim detection from robomimic checkpoints
- Added flatten_obs() for dict obs flattening
- Added ResNet18Encoder and encode_image() for vision conditioning
- Added build_cond_vec() to unify state/vision/state+vision conditioning
- TrajectoryDiffusion, make_beta_schedule, q_sample, diffusion_denoise_action unchanged

Conditioning modes (selected via cond_mode argument):
    'state':        cond = flatten_obs()           [Ds]
    'vision':       cond = encode_image()          [512]
    'state+vision': cond = concat(state, vision)   [Ds + 512]

Supports: Lift, Can, Square, and any robomimic task automatically.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

VISION_DIM = 512  # ResNet-18 output dim — constant


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


def get_cond_dim(cond_mode, state_dim):
    """
    Return the conditioning dimension for a given mode.

    Args:
        cond_mode:  'state' | 'vision' | 'state+vision'
        state_dim:  int, flattened state dimension

    Returns:
        int: cond_dim to pass to TrajectoryDiffusion
    """
    if cond_mode == 'state':
        return state_dim
    elif cond_mode == 'vision':
        return VISION_DIM
    elif cond_mode == 'state+vision':
        return state_dim + VISION_DIM
    else:
        raise ValueError(f"Unknown cond_mode: {cond_mode}. "
                         f"Choose from 'state', 'vision', 'state+vision'")


# =========================
# VISION ENCODER
# =========================

class ResNet18Encoder(nn.Module):
    """
    Frozen pretrained ResNet-18 image encoder.

    Input:  [B, 3, H, W] float32 in [0, 1]
    Output: [B, 512]

    Weights are frozen — no training needed.
    ImageNet pretrained features work well for manipulation tasks.
    """
    def __init__(self):
        super().__init__()
        import torchvision.models as tv_models
        import torchvision.transforms as T

        base = tv_models.resnet18(weights=tv_models.ResNet18_Weights.DEFAULT)
        # Remove final FC classification layer — keep up to global avgpool
        self.encoder = nn.Sequential(*list(base.children())[:-1])

        # Freeze all weights — no gradients needed
        for p in self.encoder.parameters():
            p.requires_grad = False

        # ImageNet normalisation
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )

    def forward(self, img):
        """
        Args:
            img: [B, 3, H, W] float in [0, 1]
        Returns:
            [B, 512]
        """
        img = self.normalize(img)
        return self.encoder(img).squeeze(-1).squeeze(-1)  # [B, 512]


# Global encoder instance — created once, reused across all calls
_ENCODER = None

def get_encoder():
    """Return the global ResNet-18 encoder, creating it on first call."""
    global _ENCODER
    if _ENCODER is None:
        _ENCODER = ResNet18Encoder().eval()
    return _ENCODER


def encode_image(img_uint8):
    """
    Encode a single HWC uint8 image to a 512-dim numpy vector.

    Args:
        img_uint8: np.array [H, W, 3] uint8 — raw camera frame from robosuite

    Returns:
        np.array [512] float32
    """
    encoder = get_encoder()
    img = torch.FloatTensor(img_uint8).permute(2, 0, 1).unsqueeze(0) / 255.0
    # [1, 3, H, W]
    with torch.no_grad():
        feat = encoder(img)
    return feat.squeeze(0).numpy().astype(np.float32)  # [512]


def build_cond_vec(obs_dict, obs_keys, cond_mode):
    """
    Build the conditioning vector for the diffusion model.
    Single entry point — handles all three modes uniformly.

    Args:
        obs_dict:  dict from robosuite env.reset() / env.step()
        obs_keys:  list of state obs keys (from get_task_dims)
        cond_mode: 'state' | 'vision' | 'state+vision'

    Returns:
        np.array — cond vector to pass to diffusion_denoise_action()
    """
    if cond_mode == 'state':
        return flatten_obs(obs_dict, obs_keys)

    elif cond_mode == 'vision':
        img = obs_dict['agentview_image']   # [H, W, 3] uint8
        return encode_image(img)

    elif cond_mode == 'state+vision':
        state = flatten_obs(obs_dict, obs_keys)
        img   = obs_dict['agentview_image']
        vis   = encode_image(img)
        return np.concatenate([state, vis]).astype(np.float32)

    else:
        raise ValueError(f"Unknown cond_mode: {cond_mode}")


# =========================
# DIFFUSION MODEL
# =========================

class TrajectoryDiffusion(nn.Module):
    """
    DDPM-style diffusion model for action trajectories.

    Ported directly from MADDPG MPE testbed. Unchanged.
    x:    [B, H, Da]  — action trajectory
    cond: [B, Dc]     — conditioning vector (state, vision, or state+vision)

    cond_dim varies by mode:
        state:        Ds  (e.g. 19 for Lift, 26 for Square)
        vision:       512
        state+vision: Ds + 512
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
            cond:    [B, Dc]
        Returns:
            eps_pred: [B, H, Da]
        """
        B      = x_noisy.shape[0]
        x_flat = x_noisy.reshape(B, -1)

        t_norm   = t.float().unsqueeze(-1) / 1000.0
        t_emb    = self.time_mlp(t_norm)
        c_emb    = self.cond_mlp(cond)
        h        = t_emb + c_emb

        h_cat    = torch.cat([x_flat, h], dim=-1)
        eps_pred = self.net(h_cat)
        eps_pred = eps_pred.view(B, self.horizon, self.action_dim)
        return eps_pred


# =========================
# DIFFUSION SCHEDULE
# =========================

def make_beta_schedule(T, beta_start=1e-4, beta_end=2e-2):
    """Linear beta schedule. Ported directly from MPE testbed."""
    betas      = torch.linspace(beta_start, beta_end, T)
    alphas     = 1.0 - betas
    alphas_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, alphas_bar


def q_sample(x0, t, eps, alphas_bar):
    """Forward diffusion q(x_t | x_0). Ported directly from MPE testbed."""
    a_bar = alphas_bar[t].view(-1, 1, 1).to(x0.device)
    return torch.sqrt(a_bar) * x0 + torch.sqrt(1.0 - a_bar) * eps


# =========================
# MODEL LOADING / SAVING
# =========================

DIFFUSION_MODEL  = None
DIFFUSION_CONSTS = {}


def load_diffusion_model(model_path):
    """
    Load a trained diffusion model from disk into global state.

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
        "obs_keys":   ckpt["obs_keys"],
        "cond_mode":  ckpt.get("cond_mode", "state"),  # backward compatible
        "task":       ckpt.get("task", "unknown"),
    }

    print(f"[Diffusion] Loaded | task={DIFFUSION_CONSTS['task']} | "
          f"cond_mode={DIFFUSION_CONSTS['cond_mode']} | "
          f"action_dim={ckpt['action_dim']} | cond_dim={ckpt['cond_dim']}")


def get_diffusion_obs_keys():
    """Return obs_keys from loaded diffusion model."""
    return DIFFUSION_CONSTS["obs_keys"]


def get_diffusion_cond_mode():
    """Return cond_mode from loaded diffusion model."""
    return DIFFUSION_CONSTS["cond_mode"]


# =========================
# INFERENCE
# =========================

@torch.no_grad()
def diffusion_denoise_action(noisy_action_vec, cond_vec, t_start=40):
    """
    Denoise a single action vector using reverse diffusion.
    Ported directly from MPE testbed.

    Args:
        noisy_action_vec: np.array [Da]   — action with noise applied
        cond_vec:         np.array [Dc]   — conditioning vector from build_cond_vec()
        t_start:          int             — reverse diffusion start step

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

    # Build x_t
    x       = torch.zeros((1, H, a.shape[0]))
    x[0, 0] = a

    cond = torch.from_numpy(cond_vec).float().unsqueeze(0)  # [1, Dc]

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

    clean = x[0, 0] * C["act_std"][0, 0] + C["act_mean"][0, 0]
    return clean.numpy()


@torch.no_grad()
def diffusion_denoise_action_window(noisy_action_vec, cond_vec, t_start=40):
    """
    Denoise using a sliding window (H>1).

    Args:
        noisy_action_vec: np.array [Da]
        cond_vec:         np.array [Dc]  — from build_cond_vec()
        t_start:          int

    Returns:
        clean_action: np.array [Da]
    """
    model = DIFFUSION_MODEL
    C     = DIFFUSION_CONSTS

    H          = C["H"]
    alphas     = C["alphas"]
    alphas_bar = C["alphas_bar"]

    a = torch.from_numpy(noisy_action_vec).float()
    a = (a - C["act_mean"][0, 0]) / C["act_std"][0, 0]

    x    = a.unsqueeze(0).unsqueeze(0).repeat(1, H, 1)
    cond = torch.from_numpy(cond_vec).float().unsqueeze(0)

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

    clean = x[0, 0] * C["act_std"][0, 0] + C["act_mean"][0, 0]
    return clean.numpy()