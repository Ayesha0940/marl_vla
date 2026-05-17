# SA-MDP Diffusion Policy: Implementation Details

## Background

State-Adversarial MDP (SA-MDP) is a robustness regularization technique from Zhang et al. (2020) originally developed for RL policies. The core idea: a robust policy should produce **consistent outputs under small perturbations of its input observations**. During training, you add a second forward pass with corrupted observations and penalize any divergence from the clean prediction.

This implementation applies SA-MDP to a **DiffusionPolicy-C (1D temporal UNet)** trained on the Robomimic Lift task. The architecture and training infrastructure are identical to the vanilla UNet trainer — only the loss function changes.

---

## Loss Function

### Vanilla x0 prediction loss

The vanilla trainer uses x0 (sample) prediction rather than epsilon prediction. The network directly predicts the clean action sequence from a noisy version:

```
x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * eps      # forward diffusion
x0_hat = model(x_t, t, obs)                                       # network predicts clean x0
loss_bc = MSE(x0_hat, x0)
```

### SA-MDP augmented loss

```
x_t = q_sample(x0, t, eps, alphas_bar)

# Clean branch — one forward pass
x0_hat = model(x_t, t, obs)
loss_bc = MSE(x0_hat, x0)

# Perturbed branch — second forward pass, same x_t and t
obs_tilde = obs + alpha * randn_like(obs)
x0_hat_tilde = model(x_t, t, obs_tilde)
loss_samdp = MSE(x0_hat_tilde, x0_hat.detach())

loss = loss_bc + kappa * loss_samdp
```

Key details:
- **One backward pass.** `x0_hat` is computed once and used by both terms. `.detach()` is applied only when `x0_hat` is used as the *target* for the perturbed branch, not when computing `loss_bc`. `loss.backward()` is called once.
- **`x_t` and `t` are shared** across both branches. The regularizer only perturbs obs — not the diffusion trajectory. This isolates the obs-conditioning path as the target of robustness training.
- **Target is the clean prediction, not the ground-truth x0.** `loss_samdp` measures consistency between the two predictions, not accuracy. This is deliberate: at high noise timesteps the network's clean prediction may itself be imprecise, and penalizing divergence from ground truth would double-penalize.

---

## Observation Perturbation: Option C

Three perturbation strategies exist for SA-MDP. This implementation uses **Option C: random sigma per batch**.

### Why Option C

The denoiser dataset (`dataset.py`) already trains with `aug_alpha_s_max=0.05` in raw state space using a uniform draw: `alpha_s ~ U(0, 0.05)`. Option C mirrors this exactly for the diffusion policy, making the noise distribution identical across both methods — removing a confound when comparing robustness results.

### Formulation

The perturbation is applied in **normalized observation space** (obs has already been z-score normalized by `LiftTrajectoryDataset.__getitem__` before reaching the loss function). To match the raw-space magnitude of `sigma_max=0.05`, we convert per-dimension:

```python
# Computed once before the training loop
obs_std_tensor = torch.from_numpy(dataset.obs_std).float().to(device)  # (obs_dim,)
sigma_max_norm = sigma_max / obs_std_tensor                              # (obs_dim,)
```

Per-batch perturbation:

```python
# alpha: (B, 1, obs_dim) — one scale per sample, per dim; broadcasts over obs_horizon
alpha     = torch.rand(B, 1, 1, device=device) * sigma_max_norm
obs_tilde = obs + alpha * torch.randn_like(obs)
```

`alpha` has shape `(B, 1, obs_dim)` and broadcasts correctly over obs of shape `(B, obs_horizon, obs_dim)`. All timesteps in the obs history receive the same per-dim scale for a given sample, but the noise itself `randn_like(obs)` is i.i.d. per element.

### Why per-dim normalization matters

`obs_std` varies significantly across observation keys (e.g., `robot0_eef_pos` has std ~0.1, `robot0_eef_quat` has std ~0.4, `object` pose components differ further). A scalar sigma applied in normalized space would be 4x larger for some keys than others in raw space. The per-dim conversion `0.05 / obs_std` ensures that `alpha * randn` in normalized space corresponds to exactly `0.05 * randn` in raw space for every key independently — apples-to-apples with the eval perturbation.

---

## Hyperparameters

| Parameter | Default | Notes |
|-----------|---------|-------|
| `kappa` | 1.0 | Weight of SA-MDP term. Sweep: 0.3 / 1.0 / 3.0 |
| `sigma_max` | 0.05 | Max perturbation in raw state space. Matches `aug_alpha_s_max` in dataset.py |

### kappa sweep guidance

| kappa | Expected behavior |
|-------|-----------------|
| 0.3 | Light regularization — clean performance preserved, modest robustness gain |
| 1.0 | Standard SA-MDP setting — balanced |
| 3.0 | Heavy regularization — strong robustness, possible clean performance hit |

Report the kappa with the best clean success rate that is also competitive at `alpha_s=0.05`. Do not select on robustness-only — that discards clean performance and is not a fair comparison to the vanilla.

---

## Training Cost

Two forward passes per batch (clean + perturbed), one backward pass. Approximately **1.9x wall-clock time** versus the vanilla trainer. Everything else — EMA, cosine LR scheduler, AdamW, gradient clipping at 1.0, `num_epochs=4500` — is identical to `train_unet_diffusion_lift.py`.

---

## Checkpoint Format

The SA-MDP checkpoint is a strict superset of the vanilla UNet checkpoint. The eval script loads it through the same `load_unet_checkpoint` path — no architecture changes, no eval-time branching required.

Additional fields written by `save_fn`:

| Key | Value |
|-----|-------|
| `backbone` | `"unet_samdp"` (overrides `"unet"` from `save_unet_checkpoint`) |
| `prediction_type` | `"x0"` |
| `samdp_kappa` | float, e.g. `1.0` |
| `samdp_sigma_max` | float, e.g. `0.05` (raw space, for human readability) |

---

## File Map

| File | Role |
|------|------|
| `training/lift/train_diffusion_samdp.py` | SA-MDP trainer (this implementation) |
| `training/lift/train_unet_diffusion_lift.py` | Vanilla control — unchanged |
| `diffusion/lift_policy_unet.py` | `UNetDiffusionPolicy`, `save_unet_checkpoint` — imported unchanged |
| `diffusion/lift_policy.py` | `LiftTrajectoryDataset`, `make_beta_schedule`, `q_sample` |
| `evaluation/eval_unet_diffusion_policy.py` | Clean rollout eval — accepts `backbone="unet_samdp"` |
| `evaluation/eval_noise_modes.py` | Robustness sweep over `alpha_s` / `alpha_a` — backbone-aware |

---

## Running the kappa sweep

```bash
# Vanilla baseline
conda run -n vla_marl python training/lift/train_unet_diffusion_lift.py

# SA-MDP kappa sweep
conda run -n vla_marl python training/lift/train_diffusion_samdp.py \
    --kappa 0.3 --output_dir checkpoints/lift_samdp_k03
conda run -n vla_marl python training/lift/train_diffusion_samdp.py \
    --kappa 1.0 --output_dir checkpoints/lift_samdp_k10
conda run -n vla_marl python training/lift/train_diffusion_samdp.py \
    --kappa 3.0 --output_dir checkpoints/lift_samdp_k30
```

## Evaluating

```bash
# Clean success rate
conda run -n vla_marl python evaluation/eval_unet_diffusion_policy.py \
    --agent checkpoints/lift_samdp_k10/best_model.pt --n_rollouts 50

# Robustness sweep (alpha_s=0.05 is the primary comparison point)
conda run -n vla_marl python evaluation/eval_noise_modes.py \
    --checkpoint checkpoints/lift_samdp_k10/best_model.pt \
    --alpha_s 0.0 0.01 0.02 0.03 0.04 0.05 \
    --alpha_a 0.0 \
    --n_rollouts 50 \
    --output_csv results/lift/samdp_k10_sweep.csv
```

---

## Reference

Zhang, H., Chen, H., Xiao, C., Li, B., Liu, M., Boning, D., & Hsieh, C.-J. (2020).
**Robust Deep Reinforcement Learning against Adversarial Perturbations on State Observations.**
NeurIPS 2020. https://arxiv.org/abs/2003.08938
