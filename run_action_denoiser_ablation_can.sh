#!/usr/bin/env bash
# 4-way ablation: base policy vs action-denoiser (no condition) vs action-denoiser
# (conditioned on state) vs joint-denoiser — Can task, anchor A0.
#
# Conditioning difference:
#   no condition  : denoiser input = noisy action only   (D_a channels, --no_state_ctx)
#   state cond    : denoiser input = [noisy action | state_ctx] (D_a+D_s channels)
#
# Steps:
#   1. Train state-conditioned action denoiser  (skipped if checkpoint already exists)
#   2. Train no-condition action denoiser       (skipped if checkpoint already exists)
#   3. Evaluate: base policy / no-cond / state-cond / joint denoiser
#
# Run from repo root:
#   bash run_action_denoiser_ablation_can.sh
#
# Optional overrides via env:
#   EPOCHS=2 N_ROLLOUTS=2 bash run_action_denoiser_ablation_can.sh   # smoke test

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BC_RNN_CKPT="$REPO_ROOT/checkpoints/bc_rnn_can/bc_rnn_can/20260420182529/models/model_epoch_600.pth"
HDF5_PATH="$REPO_ROOT/datasets/can/ph/low_dim_v141.hdf5"
DIFFUSION_CKPT="$REPO_ROOT/checkpoints/can_diffusion_policy_unet/best_model.pt"
JOINT_CKPT="$REPO_ROOT/diffusion_models/ablation_can/joint_all_three_a0.pt"
ACTION_CKPT_COND="$REPO_ROOT/diffusion_models/action_a0_can.pt"         # state-conditioned (A0)
ACTION_CKPT_NOCOND="$REPO_ROOT/diffusion_models/action_a0_nocond_can.pt" # no-condition (A0)
RESULTS_DIR="$REPO_ROOT/results/can/action_denoiser"
RESULTS_CSV="$RESULTS_DIR/ablation_nocond_vs_state.csv"

# ── Hyperparameters (override via env) ───────────────────────────────────────
ANCHOR="${ANCHOR:-A0}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-4}"
HORIZON="${HORIZON:-8}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
AUG_ALPHA_S_MAX="${AUG_ALPHA_S_MAX:-0.02}"
AUG_ALPHA_A_MAX="${AUG_ALPHA_A_MAX:-0.20}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-0}"

# Evaluation grid
N_ROLLOUTS="${N_ROLLOUTS:-30}"
ALPHA_S_LIST="${ALPHA_S_LIST:-0.0 0.01 0.02}"
ALPHA_A_LIST="${ALPHA_A_LIST:-0.0 0.2 0.3}"
T_START_LIST="${T_START_LIST:-5}"

mkdir -p "$RESULTS_DIR"

# ── Helpers ──────────────────────────────────────────────────────────────────
log() { echo -e "\n\033[1;36m==> $*\033[0m"; }
die() { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

# ── Preflight checks ─────────────────────────────────────────────────────────
log "Preflight checks"
[[ -f "$BC_RNN_CKPT" ]]    || die "BC-RNN checkpoint not found: $BC_RNN_CKPT"
[[ -f "$HDF5_PATH" ]]      || die "HDF5 dataset not found: $HDF5_PATH"
[[ -f "$DIFFUSION_CKPT" ]] || die "Diffusion policy checkpoint not found: $DIFFUSION_CKPT"
echo "  BC-RNN:    $BC_RNN_CKPT"
echo "  HDF5:      $HDF5_PATH"
echo "  Diffusion: $DIFFUSION_CKPT"

if [[ -f "$JOINT_CKPT" ]]; then
    echo "  Joint A0:  $JOINT_CKPT  (will be included in eval)"
else
    echo "  Joint A0:  NOT FOUND — eval will skip joint denoiser"
    JOINT_CKPT=""
fi

# ── Step 1: Train state-conditioned action denoiser ──────────────────────────
log "Step 1: State-conditioned action denoiser (anchor=${ANCHOR})"

if [[ -f "$ACTION_CKPT_COND" ]]; then
    echo "  Skipping — checkpoint already exists: $ACTION_CKPT_COND"
else
    python -m diffusion.train_action_denoiser_can \
        --bc_rnn_ckpt     "$BC_RNN_CKPT" \
        --hdf5_path       "$HDF5_PATH" \
        --anchor          "$ANCHOR" \
        --horizon         "$HORIZON" \
        --diffusion_steps "$DIFFUSION_STEPS" \
        --epochs          "$EPOCHS" \
        --batch_size      "$BATCH_SIZE" \
        --lr              "$LR" \
        --aug_alpha_s_max "$AUG_ALPHA_S_MAX" \
        --aug_alpha_a_max "$AUG_ALPHA_A_MAX" \
        --output_path     "$ACTION_CKPT_COND" \
        --device          "$DEVICE" \
        --seed            "$SEED" \
        --log_every       10
    echo "  Saved: $ACTION_CKPT_COND"
fi

# ── Step 2: Train no-condition action denoiser ───────────────────────────────
log "Step 2: No-condition action denoiser (anchor=${ANCHOR}, --no_state_ctx)"

if [[ -f "$ACTION_CKPT_NOCOND" ]]; then
    echo "  Skipping — checkpoint already exists: $ACTION_CKPT_NOCOND"
else
    python -m diffusion.train_action_denoiser_can \
        --bc_rnn_ckpt     "$BC_RNN_CKPT" \
        --hdf5_path       "$HDF5_PATH" \
        --anchor          "$ANCHOR" \
        --no_state_ctx \
        --horizon         "$HORIZON" \
        --diffusion_steps "$DIFFUSION_STEPS" \
        --epochs          "$EPOCHS" \
        --batch_size      "$BATCH_SIZE" \
        --lr              "$LR" \
        --aug_alpha_s_max "$AUG_ALPHA_S_MAX" \
        --aug_alpha_a_max "$AUG_ALPHA_A_MAX" \
        --output_path     "$ACTION_CKPT_NOCOND" \
        --device          "$DEVICE" \
        --seed            "$SEED" \
        --log_every       10
    echo "  Saved: $ACTION_CKPT_NOCOND"
fi

# ── Step 3: Evaluate all 4 conditions ────────────────────────────────────────
log "Step 3: Evaluating (${N_ROLLOUTS} rollouts per cell)"
# Checkpoint order matters for labels in CSV:
#   action_denoiser_A0_0 = no-condition (first)
#   action_denoiser_A0_1 = state-conditioned (second)

ACTION_CKPT_COND_FLAG=""
if [[ -f "$ACTION_CKPT_COND" ]]; then
    ACTION_CKPT_COND_FLAG="$ACTION_CKPT_COND"
fi

JOINT_FLAG=""
if [[ -n "$JOINT_CKPT" ]]; then
    JOINT_FLAG="--joint_denoiser $JOINT_CKPT"
fi

python evaluation/eval_action_denoiser_can.py \
    --diffusion_checkpoint "$DIFFUSION_CKPT" \
    --action_denoiser      "$ACTION_CKPT_NOCOND" $ACTION_CKPT_COND_FLAG \
    $JOINT_FLAG \
    --alpha_s              $ALPHA_S_LIST \
    --alpha_a              $ALPHA_A_LIST \
    --t_start              $T_START_LIST \
    --n_rollouts           "$N_ROLLOUTS" \
    --output_csv           "$RESULTS_CSV"

log "Done — results saved to:"
echo "  $RESULTS_CSV"
echo ""
echo "  CSV model_type legend:"
echo "    baseline             — base policy (no denoiser)"
echo "    action_denoiser_A0_0 — action denoiser, no condition (D_a input only)"
echo "    action_denoiser_A0_1 — action denoiser, conditioned on state (D_a+D_s input)"
echo "    joint_denoiser_A0    — joint denoiser A0"
