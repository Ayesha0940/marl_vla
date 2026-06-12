"""
Action-only denoiser: 1D temporal U-Net with state context and FiLM anchor conditioning.

Denoises only action, conditioned on the state (which may be noisy at deployment).
The noisy action and state are concatenated along the feature axis as the network input;
the network outputs only predicted action noise.

  Forward:  q(a_t | a_0) = N(√ᾱ_t · a_0, (1−ᾱ_t) I)
  Network:  ε̂^a = ε_θ([a_t; s], c, t)   — s is (noisy) state context
  Loss:     L = ‖ε^a − ε̂^a‖²

Warm-start mode (default): the eps target is computed to recover the clean a_0
even when x0_action is noise-augmented, teaching the model to pull off-manifold
actions back to the clean distribution.

Reuses building blocks from joint_unet.py and q_sample from model.py unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .joint_unet import (
    _groups,
    SinusoidalTimeEmbedding,
    FiLM,
    ResidualBlock1D,
    DownBlock1D,
    UpBlock1D,
    ANCHOR_DIM,
    TIME_EMB_DIM,
)
from .model import q_sample


# ── Model ──────────────────────────────────────────────────────────────────────

class ActionDenoisingUNet1D(nn.Module):
    """
    1D temporal U-Net for action-only noise prediction, conditioned on state.

    Noisy action and state context are concatenated along the feature axis as input
    (in_ch = D_a + D_s). Output is only the predicted action noise (out_ch = D_a),
    leaving the state untouched.

    Args:
        state_dim:     D_s — flattened state dimension
        action_dim:    D_a — action dimension (7 for Lift)
        anchor_dim:    D_c — projected anchor embedding (output of Anchor.compute)
        time_emb_dim:  sinusoidal time embedding dimension
        channel_sizes: (c0, c1, c2) — feature channels at each U-Net scale

    Shapes:
        x_t:        (B, H, D_a) — noisy action sequence
        state_ctx:  (B, H, D_s) — state context (may be noisy at deployment)
        anchor_emb: (B, D_c)    — from Anchor.compute(traj)
        t:          (B,)        — integer diffusion timestep in [0, T)

    Returns:
        eps_hat: (B, H, D_a) — predicted action noise
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        anchor_dim: int = ANCHOR_DIM,
        time_emb_dim: int = TIME_EMB_DIM,
        channel_sizes: tuple = (64, 128, 256),
    ):
        super().__init__()
        self.state_dim  = state_dim
        self.action_dim = action_dim
        in_ch    = action_dim + state_dim   # [noisy_action || state_ctx]
        c0, c1, c2 = channel_sizes
        cond_dim = time_emb_dim + anchor_dim

        self.time_emb   = SinusoidalTimeEmbedding(time_emb_dim)
        self.input_proj = nn.Conv1d(in_ch, c0, 1)

        # Encoder
        self.down1 = DownBlock1D(c0, c1, cond_dim, downsample=True)
        self.down2 = DownBlock1D(c1, c2, cond_dim, downsample=True)

        # Bottleneck
        self.mid = ResidualBlock1D(c2, c2, cond_dim)

        # Decoder
        self.up2 = UpBlock1D(c2, c2, c1, cond_dim, upsample=True)
        self.up1 = UpBlock1D(c1, c1, c0, cond_dim, upsample=True)

        # Output projection — action noise only (D_a channels)
        self.output_proj = nn.Sequential(
            nn.GroupNorm(_groups(c0), c0),
            nn.Mish(),
            nn.Conv1d(c0, action_dim, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        state_ctx: torch.Tensor,
        anchor_emb: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        B, H, Da = x_t.shape
        assert Da == self.action_dim, f"Expected action_dim {self.action_dim}, got {Da}"
        assert state_ctx.shape == (B, H, self.state_dim), (
            f"Expected state_ctx (B={B}, H={H}, D_s={self.state_dim}), got {state_ctx.shape}"
        )

        t_emb = self.time_emb(t)                          # (B, time_emb_dim)
        cond  = torch.cat([t_emb, anchor_emb], dim=-1)    # (B, cond_dim)

        x = torch.cat([x_t, state_ctx], dim=-1)           # (B, H, Da+Ds)
        x = x.permute(0, 2, 1)                            # (B, Da+Ds, H)
        x = self.input_proj(x)                            # (B, c0, H)

        x, skip1 = self.down1(x, cond)                    # (B, c1, H//2)
        x, skip2 = self.down2(x, cond)                    # (B, c2, H//4)
        x = self.mid(x, cond)                             # (B, c2, H//4)
        x = self.up2(x, skip2, cond)                      # (B, c1, H//2)
        x = self.up1(x, skip1, cond)                      # (B, c0, H)

        x = self.output_proj(x)                           # (B, Da, H)
        return x.permute(0, 2, 1)                         # (B, H, Da)


# ── Loss ───────────────────────────────────────────────────────────────────────

def action_denoising_loss(
    model: ActionDenoisingUNet1D,
    x0_action: torch.Tensor,
    state_ctx: torch.Tensor,
    anchor_emb: torch.Tensor,
    alphas_bar: torch.Tensor,
    x0_action_clean: torch.Tensor = None,
) -> torch.Tensor:
    """
    DDPM loss for action noise prediction only.

    When x0_action_clean is provided (warm-start mode):
      - Forward process runs from the noise-augmented x0_action
      - eps target is computed so the reverse process recovers the CLEAN action
      - Teaches the model to pull off-manifold (noisy) actions back to clean

    When x0_action_clean is None, falls back to standard DDPM.

    Args:
        model:           ActionDenoisingUNet1D
        x0_action:       (B, H, D_a) — forward process start (may be noise-augmented)
        state_ctx:       (B, H, D_s) — state context (noise-augmented at training)
        anchor_emb:      (B, D_c)
        alphas_bar:      (T,) — cumulative alphas from make_beta_schedule
        x0_action_clean: (B, H, D_a) — clean target; if None uses x0_action

    Returns:
        loss scalar
    """
    B      = x0_action.shape[0]
    T      = alphas_bar.shape[0]
    device = x0_action.device
    ab     = alphas_bar.to(device)

    t   = torch.randint(0, T, (B,), device=device)
    eps = torch.randn_like(x0_action)
    x_t = q_sample(x0_action, t, eps, ab)           # (B, H, D_a)

    if x0_action_clean is not None:
        # eps that takes x_t back to the CLEAN x0
        abar_t     = ab[t].reshape(B, 1, 1)
        eps_target = (x_t - abar_t.sqrt() * x0_action_clean) / (1.0 - abar_t).sqrt()
    else:
        eps_target = eps

    eps_hat = model(x_t, state_ctx, anchor_emb, t)
    return F.mse_loss(eps_hat, eps_target)


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def action_denoise(
    model: ActionDenoisingUNet1D,
    noisy_action: torch.Tensor,
    state_ctx: torch.Tensor,
    anchor_emb: torch.Tensor,
    alphas: torch.Tensor,
    alphas_bar: torch.Tensor,
    t_start: int = 20,
) -> torch.Tensor:
    """
    Reverse diffusion on action only; state_ctx is held fixed throughout.

    Args:
        model:        ActionDenoisingUNet1D
        noisy_action: (B, H, D_a) — already normalized
        state_ctx:    (B, H, D_s) — already normalized (may be deployment-noisy)
        anchor_emb:   (B, D_c)
        alphas:       (T,) from make_beta_schedule
        alphas_bar:   (T,) from make_beta_schedule
        t_start:      reverse diffusion start step (default 20)

    Returns:
        clean_action: (B, H, D_a) normalized
    """
    x          = noisy_action.clone()
    device     = x.device
    alphas     = alphas.to(device)
    alphas_bar = alphas_bar.to(device)

    for t in reversed(range(t_start + 1)):
        t_tensor = torch.full((x.shape[0],), t, dtype=torch.long, device=device)
        eps_pred = model(x, state_ctx, anchor_emb, t_tensor)

        a_bar  = alphas_bar[t]
        a      = alphas[t]
        x0_hat = (x - (1 - a_bar).sqrt() * eps_pred) / a_bar.sqrt()

        if t > 0:
            x = a.sqrt() * x0_hat + (1 - a).sqrt() * torch.randn_like(x)
        else:
            x = x0_hat

    return x
