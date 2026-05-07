# Diffusion Policy Debugging: How We Got from 10% → 91% Success Rate

## Overview

| Checkpoint | Script | Fixes Applied | Success Rate |
|-----------|--------|--------------|-------------|
| `lift_diffusion_policy` | v1 | none | 10% |
| `lift_diffusion_policy_v2` | v1 | none | 24% |
| `lift_diffusion_policy_v3` | v2 | Bugs 1–3 | 52% → **84%** (after eval fix) |
| `lift_diffusion_policy_v4` | v2 | Bugs 1–5 | **91%** |
| `lift_diffusion_policy_v5` | v2 | Bugs 1–5 + longer training | TBD |

---

## Bug 1 — Eval script imported from a non-existent file

**File:** `evaluation/eval_diffusion_policy.py`

The eval script tried to import `sample_action_sequence_x0` from `train_diffusion_lift_v3.py`, which does not exist. The `except ImportError` silently set it to `None`. Any x0-trained checkpoint then crashed at runtime with a misleading error message.

```python
# Broken:
from train_diffusion_lift_v3 import sample_action_sequence_x0

# Fixed:
from train_diffusion_lift_v2 import sample_action_sequence_x0
```

---

## Bug 2 — v1 training script used epsilon prediction loss under a cosine noise schedule

**File:** `training/lift/train_diffusion_lift.py`

The v1 script predicted and regressed on the noise (`eps`) added during the forward diffusion process. Under the cosine schedule, `alphas_bar[t] ≈ 1.0` at low timesteps (t ≈ 0–10), meaning almost no noise is added and `eps ≈ 0`. The MSE gradient on near-zero targets is negligible, so the model never learned to denoise at the final steps where action precision matters.

```python
# Broken — epsilon prediction:
predicted_noise = model(noisy_actions, timesteps, obs)
loss = mse_loss(predicted_noise, noise)

# Fixed — x0 prediction (in train_diffusion_lift_v2.py):
x0_hat = model(noisy_actions, timesteps, obs)
loss = mse_loss(x0_hat, action_batch)
```

x0 prediction gives uniform gradient signal at all timesteps because the target (`x0`) has consistent magnitude regardless of the noise level.

**Note:** With z-score normalised actions the targets are NOT bounded to `[-1, 1]`, so no clamping is applied to `x0_hat` during either training or inference.

---

## Bug 3 — EMA weights not properly extracted before saving (v1 script)

**File:** `training/lift/train_diffusion_lift.py`

PyTorch's `AveragedModel` stores EMA-averaged weights in its own `state_dict()` with `module.`-prefixed keys. The v1 script passed `ema_model.module` directly to `torch.save`, which saved the raw unwrapped model — bypassing all averaging.

```python
# Broken — saves raw model, not EMA weights:
torch.save({"model_state_dict": ema_model.module.state_dict(), ...}, path)

# Fixed (in train_diffusion_lift_v2.py) — strip the module. prefix from EMA state dict:
def get_ema_state_dict(ema_model):
    return {k.replace("module.", "", 1): v
            for k, v in ema_model.state_dict().items()
            if k.startswith("module.")}
```

---

## Bug 4 — Observation history not updated after every environment step

**File:** `evaluation/eval_diffusion_policy.py`

The model uses an observation horizon of 2 (it expects the two most recent consecutive observations). During eval, the policy executes 8 actions per inference call (`exec_horizon=8`). The history deque was only updated once per inference call — at the top of the while loop using the observation from the *start* of the 8-step block. This meant the model received `obs_hist = [obs_t, obs_{t+8}]` (an 8-step gap) instead of `[obs_{t+7}, obs_{t+8}]` (consecutive), completely mismatching the training distribution.

```python
# Broken — history updated once per 8-step block:
while step < horizon:
    obs_vec = _flatten_obs(obs, obs_keys)
    history.append(obs_vec)          # obs is 8 steps stale
    obs_hist = np.stack(history)
    action_chunk = sample(...)
    for action in action_chunk[:8]:
        obs, _, done, _ = env.step(action)
        step += 1

# Fixed — history updated after every single env.step:
obs_vec = _flatten_obs(obs, obs_keys)
history = deque([obs_vec] * obs_horizon, maxlen=obs_horizon)
while step < horizon:
    obs_hist = np.stack(history)
    action_chunk = sample(...)
    for action in action_chunk[:8]:
        obs, _, done, _ = env.step(action)
        step += 1
        obs_vec = _flatten_obs(obs, obs_keys)
        history.append(obs_vec)      # updated every step
```

This single fix alone took the v3 checkpoint from 52% to **84%**.

---

## Bug 5 — LR warm restarts prevented convergence

**File:** `training/lift/train_diffusion_lift_v2.py`

`CosineAnnealingWarmRestarts(T_0=1000)` resets the learning rate back to its initial value (1e-4) at epochs 1001, 2001, 3001, and 4001. The model was therefore trained at high LR throughout all 4500 epochs and never had a sustained low-LR fine-tuning phase. The paper uses a simple monotonic cosine decay.

```python
# Broken — LR resets every 1000 epochs:
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=1, eta_min=1e-6)

# Fixed — LR decays smoothly from 1e-4 → 1e-6 over the full run:
scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
```

---

## Bug 6 — EMA decay too fast

**File:** `training/lift/train_diffusion_lift_v2.py`

EMA decay of `0.999` gives a half-life of ~693 gradient steps. Over ~1.35M total steps (4500 epochs × ~300 batches), this makes the EMA extremely sensitive to recent weights — almost equivalent to using no averaging. The paper uses `0.9999` (half-life ~6,931 steps), which provides a much smoother model.

```python
# Broken:
get_ema_multi_avg_fn(0.999)

# Fixed:
get_ema_multi_avg_fn(0.9999)
```

---

## Summary Table

| Bug | File | Root cause | Impact |
|-----|------|-----------|--------|
| 1 — Wrong import filename | `eval_diffusion_policy.py` | Import from non-existent `v3` file | x0 checkpoints crash at eval |
| 2 — Epsilon loss + cosine schedule | `train_diffusion_lift.py` | Gradient vanishes at low timesteps | Undertrained denoiser at final steps |
| 3 — EMA not saved correctly | `train_diffusion_lift.py` | `ema_model.module` bypasses averaging | No benefit from EMA |
| 4 — Obs history stale during exec | `eval_diffusion_policy.py` | History updated once per 8-step block | 8-step observation gap at inference |
| 5 — LR warm restarts | `train_diffusion_lift_v2.py` | LR resets every 1000 epochs | No fine-tuning phase |
| 6 — EMA decay too fast | `train_diffusion_lift_v2.py` | Decay 0.999 ≈ no averaging over long run | Noisy model weights |
