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
