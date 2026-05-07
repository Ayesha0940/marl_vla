# Diffusion Policy Architecture Comparison

Three noise-prediction networks for the Lift task, each implementing the same DDPM loop
(shared via `lift_policy.py`) but with a different backbone.

---

## At a Glance

| Property | MLP (`lift_policy.py`) | UNet (`lift_policy_unet.py`) | Transformer (`lift_policy_transformer.py`) |
|---|---|---|---|
| Class | `LiftDiffusionPolicy` | `UNetDiffusionPolicy` | `TransformerDiffusionPolicy` |
| Backbone family | Residual MLP | 1D Temporal Conv (UNet) | Encoder-Decoder Transformer |
| Chi et al. 2023 name | — (custom) | DiffusionPolicy-C | DiffusionPolicy-T |
| Action representation | Flat vector `(B, T_a*D_a)` | Channels over time `(B, D_a, T_a)` | Token sequence `(B, T_a, D_a)` |
| Temporal inductive bias | None | Local convolutions over time axis | Global self-attention over action tokens |
| Conditioning mechanism | FiLM (scale+shift) per residual block | FiLM per conv residual block | Cross-attention to conditioning tokens |
| How obs+time are fused | Single MLP → scalar `cond` vector | Single MLP → scalar `cond` vector | Separate token streams, concatenated |
| Default `action_horizon` | 8 | 16 | 16 |
| Default `obs_horizon` | 2 | 2 | 2 |
| Default hidden width | `hidden_dim=256` | `down_dims=[256,512,1024]` | `d_model=256` |
| Weight init | PyTorch default | PyTorch default | Xavier uniform |
| Dropout | None | None | Yes (`dropout=0.1`) |
| Normalization | LayerNorm | GroupNorm (conv blocks) | LayerNorm |
| Activation | ReLU in blocks, Mish in cond encoder | Mish throughout | GELU in FFN |

---

## 1. MLP — `LiftDiffusionPolicy`

**File:** [lift_policy.py](lift_policy.py)

### Data flow

```
obs_history (B, T_o, D_o)  +  timesteps (B,)
        │                           │
  flatten → (B, T_o*D_o)    sinusoidal emb (B, time_emb_dim)
        └──────────── cat ──────────┘
                      │
               cond_encoder MLP
                      │
               cond (B, hidden_dim)
                      │
noisy_actions (B, T_a, D_a) ──flatten──► input_proj ──► x (B, hidden_dim)
                                                          │
                                              ┌─ ResidualBlock(x, cond) ─┐ × N
                                              └──────────────────────────┘
                                                          │
                                                   output_proj
                                                          │
                                          (B, T_a, D_a)  ← reshape
```

### Key building block: `ResidualBlock`

Each block receives the same global `cond` vector and applies FiLM (feature-wise linear
modulation) after the first linear layer:

```
scale, shift = cond_proj(cond).chunk(2)   # one projection per block
h = relu(fc1(norm1(x))) * (1 + scale) + shift
h = fc2(norm2(h))
out = x + h
```

The entire action sequence is treated as a **single flat vector** — there is no explicit
temporal structure inside the network.

### Strengths / Weaknesses
- Simplest and fastest to train; fewest parameters at default settings.
- No temporal inductive bias: the network must learn action-step ordering from scratch.
- FiLM conditioning is applied at every layer, which is expressive for a global cond signal.

---

## 2. UNet — `UNetDiffusionPolicy`

**File:** [lift_policy_unet.py](lift_policy_unet.py)

### Data flow

```
obs_history + timesteps ──► ConditioningEncoder MLP ──► cond (B, cond_dim=256)

noisy_actions (B, T_a, D_a) ──transpose──► x (B, D_a, T_a)   [actions as channels]
        │
        ▼  Encoder (down path)
  ConvResBlock × 2  →  FiLM(cond)  →  save skip  →  Conv1d stride-2 downsample
  ... × n_levels
        │
        ▼  Bottleneck
  ConvResBlock × 2  →  FiLM(cond)
        │
        ▼  Decoder (up path)
  cat(x, skip)  →  ConvResBlock × 2  →  FiLM(cond)  →  ConvTranspose1d upsample
  ... × n_levels
        │
        ▼  final_conv
  (B, action_dim, T_a) ──transpose──► (B, T_a, D_a)
```

### Encoder / Decoder channel layout (default `down_dims=[256,512,1024]`)

| Level | Encoder in→out | Skip channels | Decoder concat in | Decoder out |
|---|---|---|---|---|
| 0 | 7 → 256 | 256 | 512 + 256 = 768 | 256 |
| 1 | 256 → 512 | 512 | 1024 + 512 = 1536 | 512 |
| 2 | 512 → 1024 | 1024 | 1024 + 1024 = 2048 | 1024 |

Downsampling: `Conv1d(stride=2)`. Upsampling: `ConvTranspose1d(stride=2)`.
The bottleneck level uses `nn.Identity()` for both (no spatial reduction at deepest level).

### Key building block: `ConditionalResidualBlock1D`

```
out = Conv1dBlock(x)                       # GroupNorm + Mish
scale, shift = cond_proj(cond).unsqueeze(-1).chunk(2)
out = out * (1 + scale) + shift            # FiLM applied to first conv output
out = Conv1dBlock(out)
return out + residual_conv(x)              # skip connection, 1×1 conv if channels differ
```

FiLM is applied between the two convolutions (after conv1, before conv2), broadcasting
over the time dimension via `.unsqueeze(-1)`.

### Strengths / Weaknesses
- Local convolutions give strong **temporal inductive bias** — nearby action steps share
  receptive fields.
- Multi-scale skip connections allow the network to predict noise at different temporal
  resolutions.
- Larger parameter count and more memory due to hierarchical channel expansion.
- Conditioning is still a single global vector (same as MLP); no per-token conditioning.

---

## 3. Transformer — `TransformerDiffusionPolicy`

**File:** [lift_policy_transformer.py](lift_policy_transformer.py)

### Data flow

```
── Conditioning stream ──────────────────────────────────────────
timesteps (B,)  ──► SinusoidalEmbedding ──► time_proj ──► t_token (B, 1, d_model)
obs_history (B, T_o, D_o) ──► obs_proj + obs_pos_emb ──► obs_tokens (B, T_o, d_model)
cond_tokens = cat([t_token, obs_tokens], dim=1)   # (B, 1+T_o, d_model)
cond_tokens ──► TransformerEncoderBlock × n_enc_layers ──► enc_norm ──► cond_tokens

── Action stream ────────────────────────────────────────────────
noisy_actions (B, T_a, D_a) ──► action_proj + action_pos_emb ──► action_tokens (B, T_a, d_model)
action_tokens ──► TransformerDecoderBlock(action_tokens, cond_tokens) × n_dec_layers ──► dec_norm
──► output_head ──► (B, T_a, D_a)
```

### Key building blocks

**`TransformerEncoderBlock`** (conditioning stream):
- Pre-norm self-attention over all conditioning tokens (timestep + obs frames).
- Standard FFN (GELU, dropout).

**`TransformerDecoderBlock`** (action stream):
- Pre-norm **self-attention** over action tokens (no causal mask — all positions predict
  together, unlike autoregressive decoders).
- Pre-norm **cross-attention**: action tokens (Q) attend to conditioning tokens (K, V).
- Standard FFN (GELU, dropout).

### Why cross-attention instead of FiLM

In the MLP and UNet, the conditioning signal is a **single vector** broadcast to every
layer/position. The Transformer instead lets each action token **selectively attend** to
different conditioning tokens — the timestep token and individual obs frames can have
different influence on different action steps.

### Strengths / Weaknesses
- **Global temporal context**: each action token can attend to every other action token,
  regardless of distance — better for capturing long-range dependencies in the action chunk.
- **Per-token conditioning**: cross-attention lets each action step focus on the most
  relevant part of the observation history and diffusion timestep.
- More expensive: O(T²) self-attention in both streams; `n_enc_layers + n_dec_layers`
  stacks of multi-head attention.
- Xavier weight init and dropout included — regularisation not present in the other two.
- Same shared `make_beta_schedule`, `q_sample`, `LiftTrajectoryDataset`, and
  `sample_action_sequence` from `lift_policy.py`.

---

## Conditioning: FiLM vs Cross-Attention

| | MLP | UNet | Transformer |
|---|---|---|---|
| Conditioning token count | 1 (global vector) | 1 (global vector) | `1 + obs_horizon` (token sequence) |
| Per-layer conditioning | Same `cond` reused in every block | Same `cond` reused in every block | Dedicated cross-attention per decoder layer |
| Action-to-obs interaction | Implicit (obs baked into scalar) | Implicit (obs baked into scalar) | Explicit (each action token attends to each obs token) |
| Time-to-obs interaction | Fused before network via MLP | Fused before network via MLP | Fused inside encoder via self-attention |

---

## Shared Infrastructure

All three backbones reuse:

- **`make_beta_schedule`** — cosine schedule (Nichol & Dhariwal 2021) in `lift_policy.py`
- **`q_sample`** — forward diffusion noising in `lift_policy.py`
- **`LiftTrajectoryDataset`** — sliding-window HDF5 dataset with normalisation in `lift_policy.py`
- **`sample_action_sequence`** — DDPM reverse process with correct posterior in `lift_policy.py`

The three `load_*_checkpoint` functions all return `(model, ckpt, alphas, alphas_bar)` so
the eval script can swap backbones without modification.

---

## Parameter Count Estimates (default hyperparameters, obs_dim=23, action_dim=7)

| Model | Approximate params |
|---|---|
| MLP (`hidden=256, n_layers=4`) | ~500 K |
| UNet (`down_dims=[256,512,1024]`) | ~25–30 M |
| Transformer (`d_model=256, 4+4 layers`) | ~10–12 M |

The UNet's large parameter count comes from the hierarchical channel expansion in the
encoder/decoder; the Transformer's from multi-head attention projections across many layers.
