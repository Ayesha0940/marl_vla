# Joint (State, Action) Denoiser — Robomimic Lift

A temporal U-Net that jointly denoises corrupted state and action sequences at deployment time, conditioned on a clean anchor embedding. Designed to recover clean robot behaviour when both the state observations and executed actions are subject to Gaussian noise (communication faults, vision corruption, actuator jitter).

---

## Overview

At deployment the policy sees noisy observations and produces noisy actions. The joint denoiser receives a sliding window of corrupted `(state, action)` pairs, an uncorrupted anchor embedding, and a diffusion timestep, and outputs cleaned `(state, action)` pairs. Only the cleaned action is executed; the cleaned state is discarded.

```
corrupted (s̃_t, ã_t) ──┐
                         ├─► JointUNet1D ──► (ŝ_t, â_t)   execute â_t
anchor embedding c ──────┘
diffusion step t ────────┘
```

---

## Repository Layout

```
diffusion/
    joint_unet.py          # JointUNet1D model, joint_diffusion_loss, joint_denoise
    anchors.py             # Anchor modules A0–A8 and build_anchor factory
    dataset.py             # JointDenoiserDataset — sliding-window HDF5 loader
    train_joint_denoiser.py  # Training entry point
    model.py               # make_beta_schedule, q_sample (shared with action denoiser)

evaluation/
    eval_joint_denoiser.py   # Rollout harness: sweeps (α_s × α_a) noise grid

training/
    train_eval_joint_denoiser_anchors.sh   # Train + eval anchors A0, A2, A3, A7
    ablation_noise_loss.sh                 # Ablation study for 3 training improvements
```

---

## Architecture

### JointUNet1D (`diffusion/joint_unet.py`)

A 1D temporal U-Net operating over windows of length H (default 16).

| Component | Detail |
|---|---|
| Input | `(B, H, D_s + D_a)` — noisy joint state-action window |
| Conditioning | Sinusoidal time embedding (128-d) ++ anchor embedding (128-d) → FiLM |
| Encoder | `DownBlock1D × 2` with stride-2 downsampling |
| Bottleneck | `ResidualBlock1D` |
| Decoder | `UpBlock1D × 2` with skip connections |
| Output | `(B, H, D_s + D_a)` — predicted noise ε̂ |
| Channel sizes | `(64, 128, 256)` default; configurable via `--channel_sizes` |

**FiLM conditioning** — each residual block applies per-channel scale and shift derived from the concatenated `[t_emb; anchor_emb]` vector. FiLM weights are zero-initialised (scale → 1, shift → 0) so the network behaves as unconditioned at the start of training.

**Output split** — the last dimension is split at `state_dim`:
- `ε̂^s = eps_hat[..., :state_dim]`
- `ε̂^a = eps_hat[..., state_dim:]`

### Anchor Modules (`diffusion/anchors.py`)

Each anchor provides a clean 128-d conditioning vector. The anchor is jointly trained with the denoiser.

| ID | Name | Input keys | Input dim |
|---|---|---|---|
| A0 | No anchor | — | zero vector |
| A1 | Initial object pose | `object_pose_t0` | 14 |
| A2 | Gripper history | `gripper_history` | K (default 5) |
| A3 | Clean proprioception | `proprio` | 14 (joint_pos + eef_pos + eef_quat) |
| A4 | Phase indicator | `phase` | 3 (one-hot: approach / grasp / lift) |
| A5 | Task ID | — | 5 (one-hot, single task) |
| A6 | A1 + A2 | `object_pose_t0`, `gripper_history` | 19 |
| A7 | A1 + A3 | `object_pose_t0`, `proprio` | 28 |
| A8 | A1 + A2 + A3 + A4 | all keys | 36 |

All anchors project through a two-layer MLP (`in_dim → 256 → 128`).

**A3 threat-model note:** `proprio` uses clean encoder-level readings (joint angles via FK, EE pose). This is defensible against comm/vision faults but not against joint encoder attacks. State explicitly when reporting.

### Diffusion Schedule

Linear beta schedule via `make_beta_schedule(T=100)`. Shared with the original action-only denoiser. At inference, reverse diffusion starts from step `t_start` (5, 10, or 20) rather than T, trading off denoising strength against latency.

---

## Training

### Loss

```
L = ‖ε^a − ε̂^a‖² + λ · ‖ε^s − ε̂^s‖²
```

Default `λ = D_a / D_s`. Only `ε^a` matters at deployment; `ε^s` is a training auxiliary.

### Noise Augmentation

At each training step a fresh `(α_s, α_a)` pair is sampled and used to corrupt the clean `(state, action)` window before the diffusion forward process. This simulates deployment-level corruption during training.

### Warm-Start Separation

The forward process runs from the **corrupted** `(s̃, ã)` as the starting point, but the epsilon target is computed to recover the **clean** `(s_0, a_0)`:

```
x_t   = √ᾱ_t · x̃_0 + √(1−ᾱ_t) · ε           ← forward from noisy
ε_tgt = (x_t − √ᾱ_t · x_clean) / √(1−ᾱ_t)   ← target pulls back to clean
```

This teaches the model that "the starting point is already off-manifold; the reverse process must pull it back to clean."

### Running a Single Training

```bash
python -m diffusion.train_joint_denoiser \
    --bc_rnn_ckpt checkpoints/bc_rnn_lift/.../model_epoch_600.pth \
    --hdf5_path   datasets/lift/ph/low_dim_v141.hdf5 \
    --anchor      A7 \
    --output_path diffusion_models/joint_a7_lift.pt
```

Key flags:

| Flag | Default | Description |
|---|---|---|
| `--anchor` | A1 | Anchor variant A0–A8 |
| `--epochs` | 200 | Training epochs |
| `--lam` | D_a/D_s | State loss weight λ |
| `--noise_schedule` | `uniform` | `uniform` or `asymmetric` (see below) |
| `--no_warm_start` | off | Disable warm-start separation (standard DDPM eps target) |
| `--aug_alpha_s_max` | 0.05 | Max state noise std for augmentation |
| `--aug_alpha_a_max` | 0.20 | Max action noise std for augmentation |
| `--diffusion_steps` | 100 | DDPM schedule length T |
| `--horizon` | 16 | Temporal window H |

---

## Ablation Study — Three Training Improvements

Evaluate the following targeted improvements against a uniform-noise, warm-start-on, auto-λ baseline. Run all anchors `{A0, A2, A3, A7}` × all variants in one shot:

```bash
bash training/ablation_noise_loss.sh 2>&1 | tee ablation.log
```

Checkpoints land in `diffusion_models/ablation/joint_<variant>_<anchor>.pt`.
Eval CSVs land in `results/lift/joint_denoiser/`.

### Approach 1 — Asymmetric Noise Schedule (`--noise_schedule asymmetric`)

**Motivation.** Uniform sampling of `α_s ∈ [0, 0.05]` wastes gradient budget on the low-noise regime `[0, 0.02]` that rarely occurs at deployment. Biasing toward `[0.03, 0.05]` gives the model more signal in the operating regime.

**Implementation (`diffusion/dataset.py`, `__getitem__`).** When `noise_schedule="asymmetric"`:

```python
alpha_s = np.random.beta(3, 1) * aug_alpha_s_max   # Beta(3,1): mean = 0.75 * max ≈ 0.0375
alpha_a = np.random.uniform(0.0, aug_alpha_a_max)  # unchanged
```

`Beta(3, 1)` has mean 0.75 so at `max=0.05` the average sampled `α_s ≈ 0.0375`, with most mass in `[0.03, 0.05]`. `α_a` stays uniform because action noise is already variable at deployment and no specific sub-range is targeted.

**Flag:** `--noise_schedule asymmetric`

---

### Approach 2 — Warm-Start Separation (on by default; disable with `--no_warm_start`)

**Motivation.** Standard DDPM training targets the noisy `x̃_0` with the epsilon target derived from it. For robustness the model must learn to pull corrupted inputs *all the way back to clean*, not just back to the corrupted starting point.

**Implementation (`diffusion/joint_unet.py`, `joint_diffusion_loss`).** When `x0_state_clean` and `x0_action_clean` are supplied (default):

```python
# Forward from corrupted x̃_0
x_t = q_sample(x_noisy, t, eps, alphas_bar)

# Epsilon that maps x_t → clean x_0 (not → noisy x̃_0)
eps_target = (x_t - sqrt(abar_t) * x_clean) / sqrt(1 - abar_t)
```

The ablation `no_warmstart` disables this by passing `None` for the clean tensors, reverting to standard DDPM where eps_target targets the noisy x̃_0.

**Flag:** `--no_warm_start` to disable (compare separation ON vs OFF)

---

### Approach 3 — Loss Reweighting (`--lam`)

**Motivation.** At deployment only `â` is executed; `ŝ` is discarded. The default `λ = D_a / D_s ≈ 0.17` (7/42 for Lift) is already low, but the model may still over-invest capacity in perfectly reconstructing the state. Reducing λ further frees model capacity for the action head.

**Variants tested:**

| Variant flag | λ value | Notes |
|---|---|---|
| *(none)* | `D_a / D_s` ≈ 0.17 | Baseline auto |
| `--lam 0.25` | 0.25 | Slight upward from auto (for Lift actually larger than auto) |
| `--lam 0.1` | 0.1 | Strong downweight of state term |

For Lift `D_a = 7, D_s = 42`, so auto `λ ≈ 0.167`. Both 0.1 and 0.25 bracket this; 0.1 is the more aggressive downweight.

---

### Ablation Variants

| Variant name | Noise schedule | Warm-start sep | λ |
|---|---|---|---|
| `baseline` | uniform | ON | auto |
| `asym_noise` | **asymmetric** | ON | auto |
| `no_warmstart` | uniform | **OFF** | auto |
| `lam01` | uniform | ON | **0.1** |
| `lam025` | uniform | ON | **0.25** |
| `asym_lam01` | **asymmetric** | ON | **0.1** |
| `all_three` | **asymmetric** | ON | **0.1** |

Anchors evaluated: **A0, A2, A3, A7** — 7 variants × 4 anchors = **28 training runs** total.

Each variant is evaluated at `t_start ∈ {5, 10}` over 25 rollouts per noise cell on the grid:

| | α_a = 0.05 | α_a = 0.10 | α_a = 0.20 |
|---|---|---|---|
| **α_s = 0.01** | · | · | · |
| **α_s = 0.05** | · | · | · |

---

## Evaluation

```bash
python evaluation/eval_joint_denoiser.py \
    --joint_ckpt diffusion_models/ablation/joint_asym_lam01_a7.pt \
    --anchor     A7 \
    --n_rollouts 25 \
    --t_start    10
```

The harness runs three conditions per noise cell automatically:

| Condition | Description |
|---|---|
| `BASE-clean` | No noise, no denoiser (performance ceiling) |
| `BASE-noisy` | Noise injected, no denoiser (performance floor) |
| `JOINT-Ax` | Joint denoiser with anchor Ax |

Results are written as CSV to `results/lift/joint_denoiser/`.

---

## Checkpoint Format

```python
torch.save({
    "model_state_dict":  ...,   # JointUNet1D weights
    "anchor_state_dict": ...,   # Anchor MLP weights
    "state_dim":         int,
    "action_dim":        int,
    "horizon":           int,   # H
    "diffusion_steps":   int,   # T
    "channel_sizes":     tuple,
    "anchor_dim":        128,
    "time_emb_dim":      128,
    "anchor_id":         str,   # "A0"–"A8"
    "object_dim":        int,
    "proprio_dim":       int,
    "gripper_k":         int,
    "state_mean":        np.ndarray,  # (D_s,)
    "state_std":         np.ndarray,
    "action_mean":       np.ndarray,  # (D_a,)
    "action_std":        np.ndarray,
    "obs_keys":          list[str],
    "best_loss":         float,
    "epochs":            int,
}, path)
```

---

## Dependencies

- Python 3.8+, PyTorch, NumPy, h5py
- `robomimic` (for environment + BC-RNN policy loading)
- MuJoCo 2.1 (`~/.mujoco/mujoco210`)
