# Diffusion Architecture Comparison

## Overview

This document compares the single-agent diffusion implementation in the `diffusion/` folder with the multi-agent MPE diffusion integration in your MADDPG-based code.

## Core similarity

- Both use the same core denoiser architecture: `TrajectoryDiffusion`.
- Both are DDPM-style models that predict noise added to action trajectories.
- Both use the same diffusion functions:
  - `make_beta_schedule(T)`
  - `q_sample(x0, t, eps, alphas_bar)`
- Both perform reverse diffusion in a loop to compute a clean action from a noisy input.

## Main architectural differences

### 1. Input / output representation

- Single-agent implementation:
  - `state_vec` is a single-agent flattened observation.
  - `action_vec` is a single-agent action vector.
  - `cond_dim` is the single-agent observation dimension.
  - `action_dim` is the single-agent action dimension.

- Multi-agent MPE implementation:
  - `state_vec` is the concatenation of all agent observations.
  - `action_vec` is the concatenation of all agent actions.
  - `cond_dim` is the global state dimension across all agents.
  - `action_dim` is the joint action dimension across all agents.
  - Requires `concat_actions()` and `split_actions()` to convert between joint and per-agent actions.

### 2. Data pipeline

- Single-agent `collect_diffusion_data.py`:
  - Rolls out a trained robomimic policy.
  - Supports sliding windows with `H > 1`.
  - Saves `states: [N, H, Ds]` and `actions: [N, H, Da]`.
  - Can collect overlapping windows across episodes.

- Multi-agent MPE `collect_diffusion_data(arglist)`:
  - Rolls out a trained MADDPG policy in a multi-agent environment.
  - Concatenates all agent observations and actions into global vectors.
  - Keeps only episodes with at least `H` timesteps.
  - Does not use overlapping sliding windows.

### 3. Inference integration

- Single-agent:
  - Has both `diffusion_denoise_action()` and `diffusion_denoise_action_window()`.
  - Supports denoising single-step and windowed trajectories.

- Multi-agent MPE:
  - Uses `diffusion_denoise_action()` as a robustness filter for noisy MADDPG actions.
  - Denoises the joint action vector and then splits it back into per-agent actions.

### 4. Environment and policy framework

- Single-agent:
  - Built around `robomimic` tasks.
  - Uses PyTorch for diffusion and robomimic policy loading.

- Multi-agent MPE:
  - Built around TensorFlow MADDPG and Multiagent Particle Environments.
  - Diffusion is added as a separate PyTorch robustness module on top of TensorFlow-trained policies.

### 5. Script organization

- Single-agent:
  - Separate scripts for collection and training:
    - `collect_diffusion_data.py`
    - `train_diffusion.py`

- Multi-agent MPE:
  - Unified script with a `--mode` argument:
    - `train`
    - `test`
    - `collect_diffusion`
    - `train_diffusion`

## Slide-ready summary

1. `Same denoiser core`: Both implementations use the same `TrajectoryDiffusion` MLP architecture.
2. `Different framing`: Single-agent uses one policy; MPE uses a joint multi-agent state/action vector.
3. `Data pipeline`: Single-agent supports sliding windows; MPE collects fixed-length episodes.
4. `Inference role`: Single-agent denoises directly; MPE uses diffusion as a robustness filter for noisy joint actions.
5. `Frameworks`: Single-agent is robomimic + PyTorch; MPE is MultiAgent Particle Environment + TensorFlow policy + PyTorch diffusion.

## Comparison table

| Aspect | Single-agent diffusion | Multi-agent MPE diffusion |
|---|---|---|
| State input | single-agent obs | concatenated all-agent obs |
| Action input | single-agent action | concatenated joint action |
| Condition dimension | single-agent `Ds` | global `Ds` |
| Action dimension | single-agent `Da` | joint `Da` |
| Sliding windows | supported | not supported |
| Agent split/merge | not needed | required |
| Environment | robomimic | multi-agent particle env |
| Policy training | robomimic policy checkpoint | MADDPG TensorFlow agents |
| Training script layout | separate collection/train scripts | unified mode-based script |

## Practical note

The main difference is not the denoiser network architecture itself, but the surrounding data representation and integration with the control policy.

---

## Vision Conditioning Update

### What changed

The single-agent diffusion pipeline was extended to support three conditioning modes via a `--cond_mode` flag. The `TrajectoryDiffusion` model architecture is unchanged — only the conditioning vector fed into it changes.

### New conditioning modes

| Mode | `cond_dim` | Source | Use case |
|---|---|---|---|
| `state` | `Ds` (e.g. 26 for Square) | `flatten_obs()` | Original behaviour |
| `vision` | `512` | ResNet-18 agentview image | Vision-only ablation |
| `state+vision` | `Ds + 512` | concat(state, vision) | Full conditioning |

### New components in `model.py`

`ResNet18Encoder` — frozen pretrained ResNet-18. Takes `[B, 3, H, W]` float in `[0,1]`, returns `[B, 512]`. Weights are frozen — no training needed. ImageNet pretrained features transfer well to manipulation tasks.

`encode_image(img_uint8)` — encodes a single `[H, W, 3]` uint8 camera frame to a `[512]` numpy vector. Handles uint8→float conversion and ImageNet normalisation internally.

`build_cond_vec(obs_dict, obs_keys, cond_mode)` — single entry point for all three modes. Replaces the previous `flatten_obs()` call in all eval and collection scripts.

`get_cond_dim(cond_mode, state_dim)` — returns the correct `cond_dim` integer to pass to `TrajectoryDiffusion`.

`get_diffusion_cond_mode()` — returns the `cond_mode` stored in a loaded checkpoint so eval scripts can reconstruct conditioning automatically.

`get_encoder()` — global singleton `ResNet18Encoder`, created on first call and reused across all subsequent calls.

### Changes in `collect_diffusion_data.py`

`--cond_mode` argument added. When `vision` or `state+vision`, env is created with `use_camera_obs=True`, `has_offscreen_renderer=True`, `camera_names=['agentview']` at `84x84`. Output `.npz` now saves `conds [N, H, Dc]` instead of `states [N, H, Ds]`, and metadata includes `cond_mode` and `cond_dim`.

### Changes in `train_diffusion.py`

`cond_mode` and `cond_dim` read directly from `.npz` — no manual flag needed. `cond_mode` saved into checkpoint for eval scripts to read back automatically.

### What did NOT change

- `TrajectoryDiffusion` forward pass
- `make_beta_schedule`, `q_sample`
- `diffusion_denoise_action`, `diffusion_denoise_action_window` — second argument renamed `state_vec` → `cond_vec`, behaviour identical
- Training loop, normalisation, checkpoint format structure

### Eval script update (one line change)

Replace:
```python
state_vec = flatten_obs(obs, obs_keys)
denoised  = diffusion_denoise_action(noisy_action, state_vec, t_start)
```
With:
```python
cond_vec = build_cond_vec(obs, obs_keys, cond_mode)
denoised = diffusion_denoise_action(noisy_action, cond_vec, t_start)
```
Where `cond_mode = get_diffusion_cond_mode()` after loading the model.

### Collection commands

```bash
export CUDA_VISIBLE_DEVICES=0
cd ~/marl_vla
CKPT=checkpoints/bc_rnn_lift/bc_rnn_lift/*/models/model_epoch_600.pth

python -m diffusion.collect_diffusion_data --checkpoint $CKPT --task lift --cond_mode state        --n_episodes 200
python -m diffusion.collect_diffusion_data --checkpoint $CKPT --task lift --cond_mode vision       --n_episodes 200
python -m diffusion.collect_diffusion_data --checkpoint $CKPT --task lift --cond_mode state+vision --n_episodes 200
```

### Training commands

```bash
python -m diffusion.train_diffusion --data_path diffusion_data/lift_diffusion_data_H1_state.npz             --task lift
python -m diffusion.train_diffusion --data_path diffusion_data/lift_diffusion_data_H1_vision.npz            --task lift
python -m diffusion.train_diffusion --data_path diffusion_data/lift_diffusion_data_H1_state_plus_vision.npz --task lift
```

### Paper ablation table this enables

```
Table: Robustness Recovery under Action Noise
──────────────────────────────────────────────────────────────────
Denoiser conditioning    t_start=20   t_start=40   t_start=60
──────────────────────────────────────────────────────────────────
None (noisy baseline)       x%           x%           x%
State only                  x%           x%           x%
Vision only                 x%           x%           x%
State + Vision              x%           x%           x%
──────────────────────────────────────────────────────────────────
```