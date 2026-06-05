"""
Joint (state, action) denoiser with dual-stream cross-attention and
auxiliary forward- and inverse-dynamics supervision.

Drop-in replacement for JointUNet1D. Forward signature is identical:
    eps_hat = model(x_t, anchor_emb, t)         # (B, H, D_s+D_a)
    eps_s, eps_a = model.predict_noise(x_t, anchor_emb, t)

New auxiliary heads:
    model.predict_dynamics(clean_s, clean_a, anchor_emb)
        → Δs_{t+1} predictions, shape (B, H-1, D_s)   [R²_fwd = 0.57]
    model.predict_inverse_dynamics(clean_s, clean_a, anchor_emb)
        → â_t predictions, shape (B, H-1, D_a)         [R²_inv = 0.89]

Why inverse dynamics is weighted higher
----------------------------------------
On OSC-Pose Lift the action is essentially a desired EEF displacement, so
a_t is almost directly encoded in (s_t, s_{t+1}).  R²_inv = 0.89 vs
R²_fwd = 0.57 means the inverse-dynamics head provides a much cleaner
gradient signal.  Every gradient step on loss_inv shapes the cross-attention
to route action-relevant info through the state stream — which is exactly
the inductive bias needed at deployment when actions are corrupted.

See joint_unet.py for the original 1D U-Net implementation this replaces.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import make_beta_schedule, q_sample  # reuse existing schedule

ANCHOR_DIM   = 128
TIME_EMB_DIM = 128


# ── Embeddings ─────────────────────────────────────────────────────────────────

class SinusoidalEmbedding(nn.Module):
    """Sinusoidal embedding for timesteps OR sequence positions."""

    def __init__(self, dim: int):
        super().__init__()
        half = dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, dtype=torch.float32)
            / max(half - 1, 1)
        )
        self.register_buffer("freqs", freqs)
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (...) integer indices → (..., dim)
        emb = x.float().unsqueeze(-1) * self.freqs            # (..., half)
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)        # (..., 2*half)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb


# ── FiLM (per-token, identity-initialised) ────────────────────────────────────

class FiLM(nn.Module):
    """
    Token-wise FiLM modulation. cond: (B, cond_dim) → scale/shift broadcast
    over H. Identity-init so untrained network is unconditioned.
    """

    def __init__(self, cond_dim: int, d_model: int):
        super().__init__()
        self.to_ss = nn.Linear(cond_dim, 2 * d_model)
        nn.init.zeros_(self.to_ss.weight)
        nn.init.zeros_(self.to_ss.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: (B, H, d_model), cond: (B, cond_dim)
        ss = self.to_ss(cond).unsqueeze(1)                     # (B, 1, 2d)
        scale, shift = ss.chunk(2, dim=-1)
        return x * (1.0 + scale) + shift


# ── Dual-stream block ─────────────────────────────────────────────────────────

class CrossStreamBlock(nn.Module):
    """
    One block of:
        self-attn on each stream
        cross-attn s←a  and  a←s   (parallel; updates computed before commit)
        FFN on each stream
        FiLM(cond) on each stream
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 cond_dim: int, dropout: float = 0.1):
        super().__init__()

        # Self-attention (pre-norm)
        self.s_self_norm = nn.LayerNorm(d_model)
        self.a_self_norm = nn.LayerNorm(d_model)
        self.s_self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.a_self_attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # Cross-attention (pre-norm). Keys/values normalised separately
        # from queries so each stream has its own scale.
        self.s_xq_norm = nn.LayerNorm(d_model)
        self.s_xk_norm = nn.LayerNorm(d_model)
        self.a_xq_norm = nn.LayerNorm(d_model)
        self.a_xk_norm = nn.LayerNorm(d_model)
        self.s_cross   = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.a_cross   = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )

        # FFN
        self.s_ff_norm = nn.LayerNorm(d_model)
        self.a_ff_norm = nn.LayerNorm(d_model)
        self.s_ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )
        self.a_ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_ff, d_model),
        )

        # Per-stream FiLM (cond = time ⊕ anchor)
        self.s_film = FiLM(cond_dim, d_model)
        self.a_film = FiLM(cond_dim, d_model)

    def forward(self, s: torch.Tensor, a: torch.Tensor,
                cond: torch.Tensor) -> tuple:
        # Self-attention
        sh = self.s_self_norm(s); s = s + self.s_self_attn(sh, sh, sh)[0]
        ah = self.a_self_norm(a); a = a + self.a_self_attn(ah, ah, ah)[0]

        # Cross-attention — compute both new states BEFORE committing
        sq, sk = self.s_xq_norm(s), self.s_xk_norm(a)
        aq, ak = self.a_xq_norm(a), self.a_xk_norm(s)
        s_new = s + self.s_cross(sq, sk, sk)[0]
        a_new = a + self.a_cross(aq, ak, ak)[0]
        s, a = s_new, a_new

        # FFN
        s = s + self.s_ff(self.s_ff_norm(s))
        a = a + self.a_ff(self.a_ff_norm(a))

        # FiLM(cond) — last so it shapes the residual stream the next block sees
        s = self.s_film(s, cond)
        a = self.a_film(a, cond)
        return s, a


# ── Cross-attention joint denoiser ─────────────────────────────────────────────

class CrossAttnJointDenoiser(nn.Module):
    """
    Dual-stream cross-attention joint denoiser.

    Shapes:
        x_t        : (B, H, D_s + D_a)  noisy concatenated state-action
        anchor_emb : (B, D_c)
        t          : (B,) int
    Returns:
        eps_hat    : (B, H, D_s + D_a)  predicted joint noise (compat. with
                                         JointUNet1D loss path)
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        anchor_dim: int = ANCHOR_DIM,
        time_emb_dim: int = TIME_EMB_DIM,
        d_model: int = 192,
        n_heads: int = 6,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_horizon: int = 64,
    ):
        super().__init__()
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.d_model     = d_model
        self.n_layers    = n_layers

        # Stream tokenisers
        self.state_in   = nn.Linear(state_dim,  d_model)
        self.action_in  = nn.Linear(action_dim, d_model)

        # Positional embedding (shared between streams; type embedding
        # adds the s-vs-a distinction)
        self.pos_emb     = SinusoidalEmbedding(d_model)
        self.type_emb_s  = nn.Parameter(torch.zeros(1, 1, d_model))
        self.type_emb_a  = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.type_emb_s, std=0.02)
        nn.init.normal_(self.type_emb_a, std=0.02)

        # Conditioning vector = time ⊕ anchor → cond_dim, projected
        self.time_emb  = SinusoidalEmbedding(time_emb_dim)
        cond_in        = time_emb_dim + anchor_dim
        self.cond_proj = nn.Sequential(
            nn.Linear(cond_in, cond_in), nn.GELU(),
            nn.Linear(cond_in, cond_in),
        )
        self.cond_dim  = cond_in

        # Stack of cross-stream blocks
        self.blocks = nn.ModuleList([
            CrossStreamBlock(d_model, n_heads, d_ff, self.cond_dim, dropout)
            for _ in range(n_layers)
        ])

        # Output heads
        self.s_out_norm = nn.LayerNorm(d_model)
        self.a_out_norm = nn.LayerNorm(d_model)
        self.state_out  = nn.Linear(d_model, state_dim)
        self.action_out = nn.Linear(d_model, action_dim)

        # Forward-dynamics auxiliary: (s_tok, a_tok) at t → Δs_{t+1}  (R²=0.57)
        self.dyn_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, state_dim),
        )

        # Inverse-dynamics auxiliary: consecutive state features → a_t  (R²=0.89)
        # Targets the much stronger signal; weighted higher in the loss.
        self.inv_dyn_head = nn.Sequential(
            nn.Linear(2 * d_model, d_model), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, action_dim),
        )

        # Cache positional ids (registered as buffer so .to(device) works)
        self.register_buffer(
            "_pos_ids", torch.arange(max_horizon, dtype=torch.float32),
            persistent=False,
        )

    # ── Shared forward path ────────────────────────────────────────────────────

    def _streams(self, x_t: torch.Tensor, anchor_emb: torch.Tensor,
                 t: torch.Tensor) -> tuple:
        """Run the dual-stream stack. Returns (s_tokens, a_tokens, cond)."""
        B, H, D = x_t.shape
        assert D == self.state_dim + self.action_dim, (
            f"Expected input dim {self.state_dim + self.action_dim}, got {D}"
        )

        s_raw = x_t[..., :self.state_dim]
        a_raw = x_t[..., self.state_dim:]

        # Tokenise
        s_tok = self.state_in(s_raw)                       # (B, H, d)
        a_tok = self.action_in(a_raw)

        # Positional + type embeddings
        pos = self.pos_emb(self._pos_ids[:H]).unsqueeze(0)  # (1, H, d)
        s_tok = s_tok + pos + self.type_emb_s
        a_tok = a_tok + pos + self.type_emb_a

        # Conditioning
        cond = self.cond_proj(
            torch.cat([self.time_emb(t), anchor_emb], dim=-1)
        )                                                   # (B, cond_dim)

        # Stack of cross-stream blocks
        for block in self.blocks:
            s_tok, a_tok = block(s_tok, a_tok, cond)

        return s_tok, a_tok, cond

    # ── Main interface ─────────────────────────────────────────────────────────

    def forward(self, x_t: torch.Tensor, anchor_emb: torch.Tensor,
                t: torch.Tensor) -> torch.Tensor:
        s_tok, a_tok, _ = self._streams(x_t, anchor_emb, t)
        eps_s = self.state_out(self.s_out_norm(s_tok))      # (B, H, D_s)
        eps_a = self.action_out(self.a_out_norm(a_tok))     # (B, H, D_a)
        return torch.cat([eps_s, eps_a], dim=-1)            # (B, H, D_s+D_a)

    def predict_noise(self, x_t: torch.Tensor, anchor_emb: torch.Tensor,
                      t: torch.Tensor) -> tuple:
        """For compatibility with joint_unet.JointUNet1D.predict_noise."""
        eps = self(x_t, anchor_emb, t)
        return eps[..., :self.state_dim], eps[..., self.state_dim:]

    # ── Auxiliary dynamics ─────────────────────────────────────────────────────

    def predict_dynamics(self, clean_state: torch.Tensor,
                         clean_action: torch.Tensor,
                         anchor_emb: torch.Tensor) -> torch.Tensor:
        """
        Forward-dynamics prediction on CLEAN inputs.

        Returns Δŝ_{t+1} for t = 0..H-2, shape (B, H-1, D_s).
        Compares against (s_{t+1} - s_t) to force stream features to encode
        dynamics-relevant information.
        """
        B = clean_state.shape[0]
        t0 = clean_state.new_zeros(B, dtype=torch.long)
        x_t = torch.cat([clean_state, clean_action], dim=-1)
        s_tok, a_tok, _ = self._streams(x_t, anchor_emb, t0)
        joint = torch.cat([s_tok[:, :-1], a_tok[:, :-1]], dim=-1)
        return self.dyn_head(joint)                          # (B, H-1, D_s)

    def predict_inverse_dynamics(self, clean_state: torch.Tensor,
                                 anchor_emb: torch.Tensor) -> torch.Tensor:
        """
        Inverse dynamics: predict a_t from state-stream features of (s_t, s_{t+1}).

        Returns predicted actions for t = 0..H-2, shape (B, H-1, D_a).
        Action channel is zeroed so the state stream cannot read actions via
        cross-attention — genuine inverse-dynamics signal, no leak.
        """
        B, H, _ = clean_state.shape
        t0 = clean_state.new_zeros(B, dtype=torch.long)
        zero_a = clean_state.new_zeros(B, H, self.action_dim)
        x_t = torch.cat([clean_state, zero_a], dim=-1)              # action masked out
        s_tok, _, _ = self._streams(x_t, anchor_emb, t0)
        s_pairs = torch.cat([s_tok[:, :-1], s_tok[:, 1:]], dim=-1)  # (B, H-1, 2d)
        return self.inv_dyn_head(s_pairs)                            # (B, H-1, D_a)


# ── Loss ───────────────────────────────────────────────────────────────────────

def cross_attn_joint_loss(
    model: CrossAttnJointDenoiser,
    x0_state: torch.Tensor,
    x0_action: torch.Tensor,
    anchor_emb: torch.Tensor,
    alphas_bar: torch.Tensor,
    lam: float = None,
    lam_dyn: float = 0.05,   # forward dynamics — low weight (R²=0.57)
    lam_inv: float = 0.30,   # inverse dynamics — primary auxiliary (R²=0.89)
    x0_state_clean: torch.Tensor = None,
    x0_action_clean: torch.Tensor = None,
) -> tuple:
    """
    Three-component loss: denoising + forward dynamics + inverse dynamics.

        L = ‖ε^a − ε̂^a‖² + λ·‖ε^s − ε̂^s‖² + λ_dyn·‖Δŝ − Δs‖² + λ_inv·‖â − a‖²

    Warm-start separation (x0_*_clean provided) is supported identically to
    joint_diffusion_loss(): the eps target is computed so the reverse process
    recovers the CLEAN x0 even when the forward process starts from a noisy x0.
    Both dynamics losses are always computed on the clean tensors.

    Args:
        model:           CrossAttnJointDenoiser
        x0_state:        (B, H, D_s)  forward-process start (typically noisy)
        x0_action:       (B, H, D_a)
        anchor_emb:      (B, D_c)
        alphas_bar:      (T,) from make_beta_schedule
        lam:             state denoising weight; None → D_a/D_s
        lam_dyn:         forward-dynamics weight (default 0.05)
        lam_inv:         inverse-dynamics weight (default 0.30)
        x0_state_clean:  optional clean state target
        x0_action_clean: optional clean action target

    Returns:
        (total, loss_a, loss_s, loss_dyn, loss_inv)
    """
    if lam is None:
        lam = model.action_dim / model.state_dim

    B      = x0_state.shape[0]
    T      = alphas_bar.shape[0]
    device = x0_state.device
    ab     = alphas_bar.to(device)

    x0_fwd = torch.cat([x0_state, x0_action], dim=-1)
    t      = torch.randint(0, T, (B,), device=device)
    eps    = torch.randn_like(x0_fwd)
    x_t    = q_sample(x0_fwd, t, eps, ab)

    if x0_state_clean is not None and x0_action_clean is not None:
        x0_clean   = torch.cat([x0_state_clean, x0_action_clean], dim=-1)
        abar_t     = ab[t].reshape(B, 1, 1)
        eps_target = (x_t - abar_t.sqrt() * x0_clean) / (1.0 - abar_t).sqrt()
        clean_s    = x0_state_clean
        clean_a    = x0_action_clean
    else:
        eps_target = eps
        clean_s    = x0_state
        clean_a    = x0_action

    eps_s_hat, eps_a_hat = model.predict_noise(x_t, anchor_emb, t)
    loss_a = F.mse_loss(eps_a_hat, eps_target[..., model.state_dim:])
    loss_s = F.mse_loss(eps_s_hat, eps_target[..., :model.state_dim])

    # Forward-dynamics auxiliary (low weight, R²=0.57)
    delta_s_pred = model.predict_dynamics(clean_s, clean_a, anchor_emb)
    delta_s_true = clean_s[:, 1:] - clean_s[:, :-1]
    loss_dyn     = F.mse_loss(delta_s_pred, delta_s_true)

    # Inverse-dynamics auxiliary (primary, R²=0.89)
    a_pred   = model.predict_inverse_dynamics(clean_s, anchor_emb)
    a_true   = clean_a[:, :-1]
    loss_inv = F.mse_loss(a_pred, a_true)

    total = loss_a + lam * loss_s + lam_dyn * loss_dyn + lam_inv * loss_inv
    return total, loss_a, loss_s, loss_dyn, loss_inv


# ── Inference (mirrors joint_unet.joint_denoise signature) ─────────────────────

@torch.no_grad()
def cross_attn_joint_denoise(
    model: CrossAttnJointDenoiser,
    noisy_state: torch.Tensor,
    noisy_action: torch.Tensor,
    anchor_emb: torch.Tensor,
    alphas: torch.Tensor,
    alphas_bar: torch.Tensor,
    t_start: int = 20,
) -> tuple:
    """Reverse diffusion starting from corrupted (s̃, ã) at step t_start."""
    x = torch.cat([noisy_state, noisy_action], dim=-1)
    device     = x.device
    alphas     = alphas.to(device)
    alphas_bar = alphas_bar.to(device)

    for t in reversed(range(t_start + 1)):
        t_tensor = torch.full((x.shape[0],), t, dtype=torch.long, device=device)
        eps_pred = model(x, anchor_emb, t_tensor)

        a_bar  = alphas_bar[t]
        a      = alphas[t]
        x0_hat = (x - torch.sqrt(1 - a_bar) * eps_pred) / torch.sqrt(a_bar)

        if t > 0:
            x = torch.sqrt(a) * x0_hat + torch.sqrt(1 - a) * torch.randn_like(x)
        else:
            x = x0_hat

    return x[..., :model.state_dim], x[..., model.state_dim:]
