# SA-MDP Regularized Diffusion Policy — Robomimic Lift

## Lift Task State Space

- **Task**: Pick up a red cube from a table and lift it above a height threshold.
- **Observation keys** (concatenated into a flat vector per timestep):
  - `robot0_eef_pos` — end-effector position (3D)
  - `robot0_eef_quat` — end-effector orientation as quaternion (4D)
  - `robot0_gripper_qpos` — gripper joint positions (2D)
  - `object` — object pose: position + quaternion (7D)
- **Total obs dim**: 16 per timestep
- **Obs horizon**: 2 timesteps → policy input is `(2, 16)` = 32-dim flattened context
- **Action space**: 7D delta end-effector control (position 3D + rotation 3D + gripper 1D)
- **Action horizon**: 16 steps predicted per chunk
- **Normalization**: per-dimension z-score (mean/std fitted over the full dataset), applied to both observations and actions before any model input

---

## SA-MDP Regularization

SA-MDP (Zhang et al. 2020) trains the policy to produce consistent actions under small observation perturbations, improving robustness to sensor noise or state estimation errors at test time.

### Loss

```
loss = loss_bc + kappa * loss_samdp

loss_bc    = MSE( model(x_t, t, obs),       x0 )
loss_samdp = MSE( model(x_t, t, obs_tilde), x0_hat.detach() )
```

- `loss_bc`: standard x0-prediction diffusion loss on clean observations
- `loss_samdp`: consistency loss — the model's output on a perturbed observation must match the clean-obs prediction (treated as a fixed target, so gradients only flow through the perturbed branch)

### Observation Perturbation (Option C — per-dim, per-sample)

```
sigma_max_norm = sigma_max / obs_std          # (obs_dim,) — one scale per feature
alpha          ~ Uniform(0, sigma_max_norm)   # (B, 1, obs_dim)
obs_tilde      = obs + alpha * randn_like(obs)
```

- `sigma_max = 0.05` in raw state space (matches `aug_alpha_s_max` used in joint-denoiser dataset)
- Dividing by `obs_std` per dimension converts the budget into normalized space, keeping the perturbation isotropic in raw space regardless of feature scale
- `alpha` is sampled fresh each batch, so the model sees the full range `[0, sigma_max_norm]` rather than a fixed noise level

---

## Training Implementation

### Data pipeline
- `LiftTrajectoryDataset`: sliding-window extraction from `datasets/lift/ph/low_dim_v141.hdf5`
- Each sample: `obs (2, 16)` + `action (16, 7)`, both z-score normalized
- `DataLoader`: batch size 256, 2 workers, shuffled, pin memory on CUDA

### Model
- **Backbone**: 1D Temporal UNet (`UNetDiffusionPolicy`) — Chi et al. 2023 architecture
- Channels: `[256, 512, 1024]`, kernel size 5, 8 group-norm groups
- Conditioning encoder: `concat(obs_flat, sinusoidal_time_emb) → cond` via 2-layer MLP with Mish, then FiLM-injected into every UNet residual block
- Prediction type: **x0** (clean action predicted directly, not noise)

### Diffusion schedule
- Cosine beta schedule (Nichol & Dhariwal 2021 / `squaredcos_cap_v2`), T = 100 steps
- Forward process: `x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps`
- Timestep `t` sampled uniformly each batch

### Optimization
- Optimizer: AdamW, lr = 1e-4, weight decay = 1e-5
- Gradient clipping: max norm 1.0
- LR schedule: cosine annealing from 1e-4 to 1e-6 over 4500 epochs
- EMA: decay = 0.9999 via `AveragedModel`; checkpoints save EMA weights

### Per-batch forward pass (two model calls)
1. **Clean branch**: `x0_hat = model(x_t, t, obs)` → computes `loss_bc`, result detached as SA-MDP target
2. **Perturbed branch**: `x0_hat_tilde = model(x_t, t, obs_tilde)` → computes `loss_samdp`
3. `total = loss_bc + kappa * loss_samdp` → single `backward()`

### Checkpointing
- `best_model.pt`: saved whenever epoch loss improves (EMA weights)
- `model_epoch_{N}.pt`: saved every 500 epochs (EMA weights)
- `model_final.pt`: raw (non-EMA) weights at end of training
- Checkpoint stores SA-MDP metadata: `kappa`, `sigma_max`, `samdp_sigma_space = "normalized"`

### Hyperparameter sweep
| Run | `kappa` | output dir |
|-----|---------|------------|
| k03 | 0.3 | `checkpoints/lift_samdp_k03` |
| k10 | 1.0 | `checkpoints/lift_samdp_unet` (default) |
| k30 | 3.0 | `checkpoints/lift_samdp_k30` |
