# Implementation Spec: Joint State-Action Diffusion Denoiser on Robomimic Lift

## 0. Context for the agent

You are extending an existing diffusion-based action denoiser (IROS submission, attached in `/docs/iros_paper.pdf`) from action-only denoising to **joint (state, action) manifold denoising** on the Robomimic Lift task. The original architecture is Eq. 11–13 of the paper and assumes the conditioning state `s` is clean. We are removing that assumption.

Please read the paper's Section IV (AR-MGD formulation) and Section V.B (Diffusion-Based Action Manifold Alignment) before writing code. All equation references below are to that paper.

**Do not** attempt to implement everything at once. Follow the milestones in Section 7. Pause and report at each gate.

---

## 1. Goal

Train and evaluate a conditional DDPM that denoises corrupted `(s̃, ã)` pairs back toward the clean manifold `p(s, a)`, with a **clean anchor `c`** supplying the only uncorrupted information. Run a full ablation over anchor choices to identify which invariants the denoiser actually needs.

---

## 2. Environment

- **Task:** Robomimic Lift (single task, single robot)
- **Dataset:** Robomimic `lift/ph/low_dim_v141.hdf5` (proficient-human demos)
- **State:** low-dim state (object pose, gripper state, joint pos/vel, EE pose) — concatenated vector
- **Action:** 7-dim (6 DoF EE delta + gripper)
- **Policy for rollouts:** Robomimic's pretrained BC-RNN on Lift (download from the official model zoo; do not retrain)

Use the `robomimic` library's standard data loading and env wrappers. Do not modify the env.

---

## 3. Architectural changes from the paper

### 3.1 Forward process (generalizing Eq. 11)

Corrupt joint (s, a) sequence of horizon H jointly:

```
q(a_t, s_t | a_0, s_0) = N(√ᾱ_t · [a_0; s_0], (1 - ᾱ_t) I)
```

Concatenate along the feature axis so the noise tensor has shape `(B, H, D_s + D_a)`.

### 3.2 Noise predictor (generalizing Eq. 12)

```
(ε̂^a_t, ε̂^s_t) = ε_θ([a_t; s_t], c, t)
```

Use the same 1D temporal U-Net backbone as the paper, but:
- Input channels: `D_s + D_a` (was `D_a`)
- Output channels: `D_s + D_a` (was `D_a`)
- Conditioning `c` is injected via FiLM layers at each resolution (same mechanism the paper uses for state conditioning; `c` now replaces `s_t` as the conditioning input)

### 3.3 Loss (generalizing Eq. 13)

```
L = E[ ‖ε^a - ε̂^a_θ‖² + λ · ‖ε^s - ε̂^s_θ‖² ]
```

Tune `λ` on a validation split. Start with `λ = D_a / D_s` to roughly equalize per-dimension contribution. Log both loss components separately.

### 3.4 Inference

Given corrupted `(s̃_{t-H+1:t}, ã_{t-H+1:t})` and anchor `c`:
1. Treat the corrupted sequence as a sample at diffusion timestep `t_start`
2. Run reverse diffusion for `t_start` steps, conditioned on `c`
3. Execute `â_t` (most recent action in the cleaned sequence); discard `ŝ`

Default `t_start = 20` to match the paper's best setting. Expose it as a CLI flag.

---

## 4. Anchor variants (ablation matrix)

Implement each anchor as a pluggable module with a unified interface:

```python
class Anchor(nn.Module):
    def compute(self, traj: dict) -> Tensor:  # returns (B, D_c)
        ...
```

Then the training config selects which anchor(s) to use. Implement every row below.

| ID | Anchor name | What it is | Source |
|----|-------------|------------|--------|
| A0 | None | Empty tensor (unanchored joint denoising) | — |
| A1 | Initial object pose | Object pose at t=0 of the episode, held constant for all H-windows in that episode | `obs['object']` at reset |
| A2 | Gripper history | Last K=5 gripper commands (binary, commanded not sensed) | Action dim 6 of previous K actions |
| A3 | Clean proprioception | Joint angles + EE pose (assumed uncorrupted by the noise process) | `obs['robot0_joint_pos']`, `obs['robot0_eef_pos']`, `obs['robot0_eef_quat']` |
| A4 | Phase indicator | One-hot over {approach, grasp, lift} derived from demo labels via a simple rule-based labeler (gripper closed + object z > threshold → lift; etc.) | Computed from clean demo trajectories; at inference use a lightweight classifier trained on clean data |
| A5 | Task ID | One-hot task embedding (trivial for Lift alone; included to preserve multi-task extensibility) | Constant |
| A6 | A1 + A2 | Initial object pose + gripper history | Concat |
| A7 | A1 + A3 | Initial object pose + clean proprioception | Concat |
| A8 | All combined | A1 + A2 + A3 + A4 | Concat |

Each anchor's output is projected through a small MLP to a fixed dimension `D_c = 128` before being fed to the U-Net's FiLM conditioning.

**Critical for A3:** the "clean proprioception" assumption is where the threat model is encoded. Document this explicitly in the logs: A3 assumes encoder-level sensing (joint angles) is not subject to the same corruption as the state channel being denoised. This is defensible for modeling comm/vision faults but must be stated.

---

## 5. Threat model grid (for evaluation)

Sweep over a 2D grid of independent state and action noise magnitudes:

- `α_a ∈ {0.0, 0.5, 1.0, 1.5, 2.0}` (action noise std, matching paper's scale)
- `α_s ∈ {0.0, 0.05, 0.1, 0.2}` (state noise std; smaller because state has larger magnitude variance — tune these by inspecting the observation histogram first, do not hardcode)

Noise is zero-mean Gaussian, injected at every timestep, applied to the deployed state observation and the policy's output action before they enter the denoiser. The underlying env is uncorrupted.

**Do not** use the same noise seed across anchors — fix seeds per (α_s, α_a, anchor, rollout_seed) tuple so comparisons are paired.

---

## 6. Baselines to include in the same evaluation harness

Run all of these through the same noise grid and same evaluation protocol:

- **BASE-noisy:** Deployed policy with noise, no denoiser. Floor.
- **BASE-clean:** Deployed policy, no noise, no denoiser. Ceiling.
- **PAPER:** Original action-only denoiser from the IROS paper (conditioned on corrupted `s̃`). This is the direct comparison.
- **CASCADE (Option B from the discussion):** Separate state denoiser → action denoiser. State denoiser is a smaller 1D temporal U-Net trained on state sequences only. Use this as the "modular" alternative.
- **JOINT-Ax:** Each anchor variant from the table above.

---

## 7. Milestones

Execute in order. **Report to the user at each gate before proceeding.** Do not skip gates.

### Milestone 1 — Data pipeline
- Load Lift demos via robomimic
- Roll out the pretrained BC-RNN policy to collect additional on-policy trajectories (target: 200 episodes, ~500 transitions each)
- Build a horizon-H=16 windowed dataset of `(s_{0:H-1}, a_{0:H-1}, anchor_features)` tuples
- Write a unit test that verifies window indexing is correct (no cross-episode leakage)
- **Gate 1:** show dataset statistics (N windows, feature dims, anchor dims) and 3 sample visualizations of joint trajectories

### Milestone 2 — Joint denoiser model
- Implement the modified U-Net with `D_s + D_a` in/out channels and FiLM-based anchor conditioning
- Implement the training loop with the two-component loss
- Train one model with anchor A7 (object pose + proprioception) as a quick sanity run for 50 epochs
- **Gate 2:** show train/val loss curves for both `L_a` and `L_s`, and a visualization of denoiser reconstruction quality on a held-out noisy trajectory. If reconstruction looks broken, stop and debug.

### Milestone 3 — Evaluation harness
- Implement the noise-injection wrapper (both state and action)
- Implement rollout evaluation: per config, run 50 episodes, report success rate (binary lift success in Robomimic's standard definition), action reconstruction error `‖â - a*‖`, state reconstruction error `‖ŝ - s*‖` (where `a*`, `s*` come from a paired clean rollout with same seed)
- Implement wall-clock latency measurement per action
- **Gate 3:** run BASE-noisy, BASE-clean, and JOINT-A7 on a single cell of the noise grid (`α_s=0.1, α_a=1.0`) and show a results table. Verify the numbers are sane before scaling.

### Milestone 4 — Full anchor ablation
- Train all 9 anchor variants (A0–A8) with the same hyperparameters, same random seed for init, same training data
- Evaluate each on the full 5×4 noise grid, 50 episodes per cell, with **5 rollout seeds** per cell (so 250 episodes per cell total for statistical reporting)
- **Gate 4:** report full results table and IQM + 95% stratified bootstrap CIs using the `rliable` library (Agarwal et al. 2021). Highlight which anchors matter most.

### Milestone 5 — Baselines for context
- Train and evaluate PAPER and CASCADE on the same noise grid
- **Gate 5:** produce the final comparison table: BASE-noisy, BASE-clean, PAPER, CASCADE, JOINT-A0 (no anchor), JOINT-best-anchor

---

## 8. Deliverables

- `src/data/` — dataset, windowing, anchor computation modules (one file per anchor)
- `src/models/joint_unet.py` — modified U-Net
- `src/models/anchors/` — pluggable anchor modules
- `src/train.py` — training entry point (config-driven via hydra or argparse, one config per anchor variant)
- `src/eval.py` — evaluation harness over the noise grid
- `configs/` — one YAML per experimental cell
- `scripts/run_ablation.sh` — launches all anchor variants
- `results/` — pickled per-cell results + auto-generated summary CSV
- `notebooks/analysis.ipynb` — generates the final plots and tables
- `README.md` — reproduction instructions

---

## 9. Constraints and conventions

- **PyTorch**, not TF (the paper uses TF 1.10 but that's not worth preserving for new work)
- Use `robomimic==0.3.0` or latest stable; pin in `requirements.txt`
- Log everything to Weights & Biases; if not available, log to TensorBoard
- Seed everything (`torch`, `numpy`, `random`, `robomimic`) explicitly
- No silent failures — if an anchor is missing a required observation key, raise a clear error

---

## 10. Things to flag to the user rather than guess

Before finalizing any of the following, stop and ask:

1. The exact state noise magnitudes — these depend on the observation scale and should be set by inspecting the data, not assumed
2. Whether "clean proprioception" (A3) should include joint velocities or just positions/poses — this affects how much information the anchor leaks
3. Whether to use the pretrained BC-RNN or train a fresh policy for rollouts — the paper trains its own, but Robomimic's pretrained is well-validated
4. Horizon length H — paper uses H=25 for MPE, but Lift episodes are shorter; suggest H=16 but confirm
5. Whether to include adversarial PGD evaluation in this spec or defer to a follow-up — it's in the reviewer weakness list but doubles the scope

---

## 11. What NOT to do

- Do not modify the Robomimic env to "make denoising easier"
- Do not use information from the clean trajectory inside the anchor computation unless the anchor spec explicitly says so (A1, A3, A4 have defensible clean sources; A2 is commanded-not-sensed)
- Do not report mean±std with 2 seeds. Use IQM with 5+ seeds and bootstrap CIs.
- Do not collapse the noise grid into a single "noise level" scalar for reporting. The independent (α_s, α_a) structure is the whole point.
