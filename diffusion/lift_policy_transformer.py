"""DiffusionPolicy-T: Transformer backbone for Robomimic Lift.

Implements the Transformer backbone from Chi et al. 2023.
The key idea is a standard decoder-only Transformer where:
  - Each action token in the noisy action sequence is one input token
  - Observation history + diffusion timestep form the conditioning prefix
  - The Transformer predicts the noise for each action token via cross-attention
    to the conditioning tokens

Shared utilities (make_beta_schedule, q_sample, LiftTrajectoryDataset,
sample_action_sequence) live in lift_policy.py.
This file only defines the network and a thin checkpoint loader.

Usage (training):
    from diffusion.lift_policy import (
        LiftTrajectoryDataset, make_beta_schedule, q_sample,
        sample_action_sequence,
    )
    from diffusion.lift_policy_transformer import (
        TransformerDiffusionPolicy, save_transformer_checkpoint,
        load_transformer_checkpoint,
    )

Usage (eval):
    model, ckpt, alphas, alphas_bar = load_transformer_checkpoint(path, device)
    # pass to sample_action_sequence exactly like the MLP/UNet versions
"""

from __future__ import annotations

import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Positional and timestep embeddings
# ---------------------------------------------------------------------------

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for either sequence positions or diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,) integer indices → (B, dim)  or  (B, T) → (B, T, dim)."""
        squeeze = x.dim() == 1
        if squeeze:
            x = x.unsqueeze(1)             # (B, 1)
        B, T = x.shape
        half = self.dim // 2
        device = x.device
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device).float()
            / max(half - 1, 1)
        )                                  # (half,)
        args = x.float().unsqueeze(-1) * freqs.unsqueeze(0).unsqueeze(0)  # (B,T,half)
        emb = torch.cat([args.sin(), args.cos()], dim=-1)                  # (B,T,dim)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        if squeeze:
            emb = emb.squeeze(1)           # (B, dim)
        return emb


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------

class TransformerEncoderBlock(nn.Module):
    """Standard pre-norm Transformer encoder block (self-attention + FFN)."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, key_padding_mask=None) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, key_padding_mask=key_padding_mask)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x


class TransformerDecoderBlock(nn.Module):
    """Pre-norm Transformer decoder block (self-attn + cross-attn + FFN).

    Action tokens attend to each other (causal mask optional) and
    cross-attend to the conditioning tokens (obs + timestep).
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.norm1      = nn.LayerNorm(d_model)
        self.self_attn  = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2      = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm3      = nn.LayerNorm(d_model)
        self.ff         = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x:    torch.Tensor,   # (B, T_action, d_model)  — action tokens
        cond: torch.Tensor,   # (B, T_cond,  d_model)  — conditioning tokens
    ) -> torch.Tensor:
        # Self-attention over action tokens (no causal mask — all tokens predict together)
        h = self.norm1(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h

        # Cross-attention: action tokens attend to conditioning tokens
        h = self.norm2(x)
        h, _ = self.cross_attn(h, cond, cond)
        x = x + h

        # FFN
        x = x + self.ff(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Transformer Diffusion Policy
# ---------------------------------------------------------------------------

class TransformerDiffusionPolicy(nn.Module):
    """DiffusionPolicy-T: Transformer noise predictor.

    Architecture (Chi et al. 2023 Figure 3, right):
      Conditioning stream:
        - obs tokens: each obs frame is projected to d_model, positional
          embedding added, then passed through n_enc_layers encoder blocks
        - timestep token: sinusoidal embedding of t, projected to d_model,
          prepended to obs tokens
      Action stream:
        - noisy action tokens: each action step projected to d_model,
          positional embedding added
        - n_dec_layers decoder blocks cross-attending to conditioning tokens
      Output:
        - linear head projects each action token back to action_dim

    Parameters
    ----------
    obs_dim        : dimensionality of a single obs vector
    action_dim     : dimensionality of a single action vector
    obs_horizon    : number of past obs frames (default 2)
    action_horizon : length of action chunk (default 16)
    d_model        : Transformer model width (default 256)
    n_heads        : number of attention heads (default 8)
    n_enc_layers   : encoder layers for conditioning stream (default 4)
    n_dec_layers   : decoder layers for action stream (default 4)
    d_ff           : FFN hidden dim (default 1024)
    dropout        : dropout probability (default 0.1)
    time_emb_dim   : sinusoidal time embedding dim (default 256)
    """

    def __init__(
        self,
        obs_dim:        int,
        action_dim:     int,
        obs_horizon:    int = 2,
        action_horizon: int = 16,
        d_model:        int = 256,
        n_heads:        int = 8,
        n_enc_layers:   int = 4,
        n_dec_layers:   int = 4,
        d_ff:           int = 1024,
        dropout:        float = 0.1,
        time_emb_dim:   int = 256,
    ):
        super().__init__()
        self.obs_dim        = obs_dim
        self.action_dim     = action_dim
        self.obs_horizon    = obs_horizon
        self.action_horizon = action_horizon
        self.d_model        = d_model

        # ---- Conditioning stream ----
        # Timestep embedding → d_model token
        self.time_emb  = SinusoidalEmbedding(time_emb_dim)
        self.time_proj = nn.Linear(time_emb_dim, d_model)

        # Obs frame projection + positional embedding
        self.obs_proj   = nn.Linear(obs_dim, d_model)
        self.obs_pos    = SinusoidalEmbedding(d_model)

        # Encoder blocks over conditioning tokens (timestep + obs)
        self.enc_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_enc_layers)
        ])
        self.enc_norm = nn.LayerNorm(d_model)

        # ---- Action stream ----
        # Noisy action projection + positional embedding
        self.action_proj = nn.Linear(action_dim, d_model)
        self.action_pos  = SinusoidalEmbedding(d_model)

        # Decoder blocks over action tokens cross-attending to conditioning
        self.dec_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_dec_layers)
        ])
        self.dec_norm = nn.LayerNorm(d_model)

        # Output head: d_model → action_dim per token
        self.output_head = nn.Linear(d_model, action_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(
        self,
        noisy_actions: torch.Tensor,   # (B, action_horizon, action_dim)
        timesteps:     torch.Tensor,   # (B,)
        obs_history:   torch.Tensor,   # (B, obs_horizon, obs_dim)
    ) -> torch.Tensor:                 # (B, action_horizon, action_dim)
        B = noisy_actions.shape[0]
        device = noisy_actions.device

        # ---- Build conditioning tokens ----
        # Timestep token: (B, 1, d_model)
        t_token = self.time_proj(self.time_emb(timesteps)).unsqueeze(1)

        # Obs tokens with positional embedding: (B, obs_horizon, d_model)
        obs_pos_ids = torch.arange(self.obs_horizon, device=device).unsqueeze(0).expand(B, -1)
        obs_tokens  = self.obs_proj(obs_history) + self.obs_pos(obs_pos_ids)

        # Concat timestep token (prepended) + obs tokens → (B, 1+obs_horizon, d_model)
        cond_tokens = torch.cat([t_token, obs_tokens], dim=1)

        # Encode conditioning tokens
        for block in self.enc_blocks:
            cond_tokens = block(cond_tokens)
        cond_tokens = self.enc_norm(cond_tokens)

        # ---- Build action tokens ----
        act_pos_ids  = torch.arange(self.action_horizon, device=device).unsqueeze(0).expand(B, -1)
        action_tokens = self.action_proj(noisy_actions) + self.action_pos(act_pos_ids)

        # Decode: action tokens cross-attend to conditioning tokens
        for block in self.dec_blocks:
            action_tokens = block(action_tokens, cond_tokens)
        action_tokens = self.dec_norm(action_tokens)

        # Project each action token back to action_dim
        return self.output_head(action_tokens)   # (B, action_horizon, action_dim)


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_transformer_checkpoint(
    path: str,
    model: TransformerDiffusionPolicy,
    dataset,
    config: dict,
    epoch: int,
    loss: float,
) -> None:
    """Save a Transformer checkpoint in the format expected by load_transformer_checkpoint."""
    model_cfg     = config["model"]
    diffusion_cfg = config["diffusion"]
    torch.save(
        {
            "backbone":        "transformer",
            "model_state_dict": model.state_dict(),
            "obs_dim":         model.obs_dim,
            "action_dim":      model.action_dim,
            "obs_horizon":     model.obs_horizon,
            "action_horizon":  model.action_horizon,
            "d_model":         model.d_model,
            "n_heads":         int(model_cfg.get("n_heads",       8)),
            "n_enc_layers":    int(model_cfg.get("n_enc_layers",  4)),
            "n_dec_layers":    int(model_cfg.get("n_dec_layers",  4)),
            "d_ff":            int(model_cfg.get("d_ff",       1024)),
            "dropout":         float(model_cfg.get("dropout",   0.1)),
            "time_emb_dim":    int(model_cfg.get("time_emb_dim", 256)),
            "diffusion_steps": int(diffusion_cfg["num_steps"]),
            "beta_start":      float(diffusion_cfg.get("beta_start", 1e-4)),
            "beta_end":        float(diffusion_cfg.get("beta_end",   2e-2)),
            "obs_keys":        list(model_cfg.get("obs_keys", [])),
            "obs_mean":        dataset.obs_mean,
            "obs_std":         dataset.obs_std,
            "action_mean":     dataset.action_mean,
            "action_std":      dataset.action_std,
            "config":          config,
            "epoch":           epoch,
            "loss":            loss,
        },
        path,
    )


def load_transformer_checkpoint(checkpoint_path: str, device: torch.device):
    """Load a Transformer checkpoint saved by save_transformer_checkpoint.

    Returns (model, checkpoint_dict, alphas, alphas_bar) — same signature
    as load_lift_checkpoint and load_unet_checkpoint so the eval script
    works without modification.
    """
    from diffusion.lift_policy import make_beta_schedule

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    required = {
        "model_state_dict", "obs_dim", "action_dim",
        "obs_horizon", "action_horizon", "diffusion_steps",
        "obs_mean", "obs_std", "action_mean", "action_std",
    }
    missing = sorted(required.difference(ckpt.keys()))
    if missing:
        raise ValueError(
            f"Not a Transformer checkpoint. Missing keys: {missing}. "
            "Use a checkpoint saved by save_transformer_checkpoint()."
        )

    model = TransformerDiffusionPolicy(
        obs_dim        = int(ckpt["obs_dim"]),
        action_dim     = int(ckpt["action_dim"]),
        obs_horizon    = int(ckpt["obs_horizon"]),
        action_horizon = int(ckpt["action_horizon"]),
        d_model        = int(ckpt.get("d_model",       256)),
        n_heads        = int(ckpt.get("n_heads",          8)),
        n_enc_layers   = int(ckpt.get("n_enc_layers",     4)),
        n_dec_layers   = int(ckpt.get("n_dec_layers",     4)),
        d_ff           = int(ckpt.get("d_ff",          1024)),
        dropout        = float(ckpt.get("dropout",      0.1)),
        time_emb_dim   = int(ckpt.get("time_emb_dim",  256)),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    _, alphas, alphas_bar = make_beta_schedule(int(ckpt["diffusion_steps"]))
    return model, ckpt, alphas.to(device), alphas_bar.to(device)