"""
Joint (state, action) denoiser: 1D temporal U-Net with FiLM anchor conditioning.

Generalizes the action-only denoiser (Eq. 11-13 in the IROS paper) to corrupt
and denoise both s and a jointly, using a clean anchor c as the only
uncorrupted conditioning signal.

  Forward:  q(a_t, s_t | a_0, s_0) = N(√ᾱ_t · [a_0; s_0], (1−ᾱ_t) I)
  Network:  (ε̂^a, ε̂^s) = ε_θ([a_t; s_t], c, t)
  Loss:     L = ‖ε^a − ε̂^a‖² + λ · ‖ε^s − ε̂^s‖²

c (the anchor) replaces the corrupted state used in the original paper.
Reuses make_beta_schedule and q_sample from model.py unchanged.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import make_beta_schedule, q_sample  # noqa: reuse existing schedule

ANCHOR_DIM   = 128   # D_c — fixed projected anchor dimension (output of Anchor.compute)
TIME_EMB_DIM = 128   # sinusoidal time embedding dimension


# ── Helpers ────────────────────────────────────────────────────────────────────

def _groups(channels: int) -> int:
    """Largest power-of-two divisor of channels, capped at 8, for GroupNorm."""
    for g in (8, 4, 2, 1):
        if channels % g == 0:
            return g
    return 1


# ── Building blocks ────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """Fixed sinusoidal embedding for diffusion timestep t."""

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000)
            * torch.arange(half, dtype=torch.float32)
            / max(half - 1, 1)
        )
        self.register_buffer("freqs", freqs)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) int
        emb = t.float().unsqueeze(1) * self.freqs.unsqueeze(0)  # (B, half)
        return torch.cat([emb.sin(), emb.cos()], dim=-1)         # (B, dim)


class FiLM(nn.Module):
    """
    Feature-wise Linear Modulation.

    Applies per-channel scale and shift derived from a conditioning vector.
    Identity-initialized so the network behaves as if unconditioned at the
    start of training, improving stability.
    """

    def __init__(self, cond_dim: int, channels: int):
        super().__init__()
        self.scale_net = nn.Linear(cond_dim, channels)
        self.shift_net = nn.Linear(cond_dim, channels)
        nn.init.zeros_(self.scale_net.weight)
        nn.init.ones_(self.scale_net.bias)   # scale → 1 at init
        nn.init.zeros_(self.shift_net.weight)
        nn.init.zeros_(self.shift_net.bias)  # shift → 0 at init

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T), cond: (B, cond_dim)
        scale = self.scale_net(cond).unsqueeze(-1)  # (B, C, 1)
        shift = self.shift_net(cond).unsqueeze(-1)  # (B, C, 1)
        return x * scale + shift


class ResidualBlock1D(nn.Module):
    """
    1D residual conv block with GroupNorm + FiLM conditioning.

    Layout: conv → norm → FiLM → mish → conv → norm → mish → + residual
    """

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, kernel_size: int = 3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_ch,  out_ch, kernel_size, padding=pad)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=pad)
        self.norm1 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.norm2 = nn.GroupNorm(_groups(out_ch), out_ch)
        self.film  = FiLM(cond_dim, out_ch)
        self.skip  = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.film(self.norm1(self.conv1(x)), cond)  # (B, out_ch, T)
        h = F.mish(h)
        h = F.mish(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class DownBlock1D(nn.Module):
    """Encoder block: residual + optional stride-2 downsampling."""

    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, downsample: bool = True):
        super().__init__()
        self.res  = ResidualBlock1D(in_ch, out_ch, cond_dim)
        self.down = (
            nn.Conv1d(out_ch, out_ch, 3, stride=2, padding=1)
            if downsample else nn.Identity()
        )

    def forward(self, x: torch.Tensor, cond: torch.Tensor):
        h = self.res(x, cond)     # (B, out_ch, T) — saved as skip connection
        return self.down(h), h   # downsampled output, skip


class UpBlock1D(nn.Module):
    """Decoder block: optional upsample → concat skip → residual."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int, cond_dim: int, upsample: bool = True):
        super().__init__()
        self.up  = (
            nn.ConvTranspose1d(in_ch, in_ch, 4, stride=2, padding=1)
            if upsample else nn.Identity()
        )
        self.res = ResidualBlock1D(in_ch + skip_ch, out_ch, cond_dim)

    def forward(self, x: torch.Tensor, skip: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        if x.shape[-1] != skip.shape[-1]:          # handle odd-length rounding
            x = F.interpolate(x, size=skip.shape[-1], mode="nearest")
        x = torch.cat([x, skip], dim=1)            # (B, in_ch+skip_ch, T_large)
        return self.res(x, cond)                    # (B, out_ch, T_large)


# ── Joint U-Net ────────────────────────────────────────────────────────────────

class JointUNet1D(nn.Module):
    """
    1D temporal U-Net for joint (state, action) noise prediction.

    Args:
        state_dim:     D_s — flattened state dimension
        action_dim:    D_a — action dimension (7 for Lift)
        anchor_dim:    D_c — projected anchor embedding (output of Anchor.compute)
        time_emb_dim:  sinusoidal time embedding dimension
        channel_sizes: (c0, c1, c2) — feature channels at each U-Net scale

    Shapes:
        x_t:        (B, H, D_s + D_a) — noisy joint state-action sequence
        anchor_emb: (B, D_c)           — from Anchor.compute(traj)
        t:          (B,)               — integer diffusion timestep in [0, T)

    Returns:
        eps_hat: (B, H, D_s + D_a) — predicted joint noise
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
        in_ch    = state_dim + action_dim
        c0, c1, c2 = channel_sizes
        cond_dim = time_emb_dim + anchor_dim   # concat sinusoidal(t) ++ anchor_emb

        self.time_emb   = SinusoidalTimeEmbedding(time_emb_dim)
        self.input_proj = nn.Conv1d(in_ch, c0, 1)

        # Encoder
        self.down1 = DownBlock1D(c0, c1, cond_dim, downsample=True)
        self.down2 = DownBlock1D(c1, c2, cond_dim, downsample=True)

        # Bottleneck
        self.mid = ResidualBlock1D(c2, c2, cond_dim)

        # Decoder  (in_ch, skip_ch, out_ch)
        self.up2 = UpBlock1D(c2, c2, c1, cond_dim, upsample=True)
        self.up1 = UpBlock1D(c1, c1, c0, cond_dim, upsample=True)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.GroupNorm(_groups(c0), c0),
            nn.Mish(),
            nn.Conv1d(c0, in_ch, 1),
        )

    def forward(
        self,
        x_t: torch.Tensor,
        anchor_emb: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        B, H, D = x_t.shape
        assert D == self.state_dim + self.action_dim, (
            f"Expected input dim {self.state_dim + self.action_dim}, got {D}"
        )

        t_emb = self.time_emb(t)                          # (B, time_emb_dim)
        cond  = torch.cat([t_emb, anchor_emb], dim=-1)    # (B, cond_dim)

        x = x_t.permute(0, 2, 1)           # (B, D, H) for Conv1d
        x = self.input_proj(x)             # (B, c0, H)

        x, skip1 = self.down1(x, cond)     # (B, c1, H//2), skip1: (B, c1, H)
        x, skip2 = self.down2(x, cond)     # (B, c2, H//4), skip2: (B, c2, H//2)

        x = self.mid(x, cond)              # (B, c2, H//4)

        x = self.up2(x, skip2, cond)       # (B, c1, H//2)
        x = self.up1(x, skip1, cond)       # (B, c0, H)

        x = self.output_proj(x)            # (B, D, H)
        return x.permute(0, 2, 1)          # (B, H, D)

    def predict_noise(
        self,
        x_t: torch.Tensor,
        anchor_emb: torch.Tensor,
        t: torch.Tensor,
    ) -> tuple:
        """
        Returns (eps_s_hat, eps_a_hat) split along the feature axis.
        Used by joint_diffusion_loss to compute the two-component loss.
        """
        eps_hat = self(x_t, anchor_emb, t)
        eps_s = eps_hat[..., :self.state_dim]
        eps_a = eps_hat[..., self.state_dim:]
        return eps_s, eps_a


# ── Loss ───────────────────────────────────────────────────────────────────────

def joint_diffusion_loss(
    model: JointUNet1D,
    x0_state: torch.Tensor,
    x0_action: torch.Tensor,
    anchor_emb: torch.Tensor,
    alphas_bar: torch.Tensor,
    lam: float = None,
    x0_state_clean: torch.Tensor = None,
    x0_action_clean: torch.Tensor = None,
) -> tuple:
    """
    Two-component DDPM loss with optional noise-augmented warm-start training.

        L = ‖ε^a − ε̂^a‖² + λ · ‖ε^s − ε̂^s‖²

    When x0_state_clean / x0_action_clean are provided (noise-augmented mode):
      - Forward process runs from the NOISY x0 (x0_state / x0_action)
      - The eps TARGET is computed so that the reverse process recovers the CLEAN x0
      - This teaches the model to denoise deployment-corrupted inputs back to clean

    When clean tensors are omitted, falls back to standard DDPM (x0 = clean).

    Args:
        model:           JointUNet1D
        x0_state:        (B, H, D_s) — noisy state (forward process start)
        x0_action:       (B, H, D_a) — noisy action (forward process start)
        anchor_emb:      (B, D_c)    — from anchor.compute(traj_batch)
        alphas_bar:      (T,)        — cumulative alphas from make_beta_schedule
        lam:             loss weight for state term; None → D_a / D_s
        x0_state_clean:  (B, H, D_s) — clean target; if None uses x0_state
        x0_action_clean: (B, H, D_a) — clean target; if None uses x0_action

    Returns:
        (total_loss, loss_a, loss_s)
    """
    if lam is None:
        lam = model.action_dim / model.state_dim

    B      = x0_state.shape[0]
    T      = alphas_bar.shape[0]
    device = x0_state.device
    ab     = alphas_bar.to(device)

    x0_fwd = torch.cat([x0_state, x0_action], dim=-1)          # (B, H, D) — noisy forward start
    t      = torch.randint(0, T, (B,), device=device)
    eps    = torch.randn_like(x0_fwd)
    x_t    = q_sample(x0_fwd, t, eps, ab)                       # (B, H, D)

    if x0_state_clean is not None and x0_action_clean is not None:
        # Compute eps that takes x_t back to the CLEAN x0, not the noisy x0.
        # x_t = sqrt(abar_t)*x0_noisy + sqrt(1-abar_t)*eps_fwd
        # eps_clean = (x_t - sqrt(abar_t)*x0_clean) / sqrt(1-abar_t)
        x0_clean = torch.cat([x0_state_clean, x0_action_clean], dim=-1)
        abar_t   = ab[t].reshape(B, 1, 1)                       # (B, 1, 1)
        eps_target = (x_t - abar_t.sqrt() * x0_clean) / (1.0 - abar_t).sqrt()
    else:
        eps_target = eps

    eps_s_hat, eps_a_hat = model.predict_noise(x_t, anchor_emb, t)

    loss_a = F.mse_loss(eps_a_hat, eps_target[..., model.state_dim:])
    loss_s = F.mse_loss(eps_s_hat, eps_target[..., :model.state_dim])
    total  = loss_a + lam * loss_s

    return total, loss_a, loss_s


# ── Inference ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def joint_denoise(
    model: JointUNet1D,
    noisy_state: torch.Tensor,
    noisy_action: torch.Tensor,
    anchor_emb: torch.Tensor,
    alphas: torch.Tensor,
    alphas_bar: torch.Tensor,
    t_start: int = 20,
) -> tuple:
    """
    Reverse diffusion starting from corrupted (s̃, ã) at diffusion step t_start.

    Args:
        model:        JointUNet1D
        noisy_state:  (B, H, D_s) — already normalized
        noisy_action: (B, H, D_a) — already normalized
        anchor_emb:   (B, D_c)
        alphas:       (T,) from make_beta_schedule
        alphas_bar:   (T,) from make_beta_schedule
        t_start:      reverse diffusion start step (default 20)

    Returns:
        (clean_state, clean_action) both (B, H, D) normalized
    """
    x = torch.cat([noisy_state, noisy_action], dim=-1)  # (B, H, D)
    device = x.device
    alphas     = alphas.to(device)
    alphas_bar = alphas_bar.to(device)

    for t in reversed(range(t_start + 1)):
        t_tensor = torch.full((x.shape[0],), t, dtype=torch.long, device=device)
        eps_pred = model(x, anchor_emb, t_tensor)

        a_bar = alphas_bar[t]
        a     = alphas[t]
        x0_hat = (x - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)

        if t > 0:
            x = torch.sqrt(a) * x0_hat + torch.sqrt(1 - a) * torch.randn_like(x)
        else:
            x = x0_hat

    clean_state  = x[..., :model.state_dim]
    clean_action = x[..., model.state_dim:]
    return clean_state, clean_action
