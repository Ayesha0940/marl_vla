#!/usr/bin/env bash
# Action-only denoiser pipeline — Lift task, anchor A0 (no anchor)
#
# Steps:
#   1. Train ActionDenoisingUNet1D with A0 (zero anchor embedding)
#   2. Evaluate: baseline vs. action-denoiser vs. existing joint-denoiser (A0)
#
# Run from repo root:
#   bash run_action_denoiser_lift_a0.sh
#
# Optional overrides via env:
#   EPOCHS=50 bash run_action_denoiser_lift_a0.sh    # quick smoke test
#   DEVICE=cpu bash run_action_denoiser_lift_a0.sh

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BC_RNN_CKPT="$REPO_ROOT/checkpoints/bc_rnn_lift/bc_rnn_lift/20260405174006/models/model_epoch_600.pth"
HDF5_PATH="$REPO_ROOT/datasets/lift/ph/low_dim_v141.hdf5"
DIFFUSION_CKPT="$REPO_ROOT/checkpoints/lift_diffusion_policy_v5/best_model.pt"
JOINT_CKPT="$REPO_ROOT/diffusion_models/joint_a0_lift.pt"        # existing A0 joint denoiser for comparison
ACTION_CKPT="$REPO_ROOT/diffusion_models/action_a0_lift.pt"
RESULTS_DIR="$REPO_ROOT/results/lift/action_denoiser"
RESULTS_CSV="$RESULTS_DIR/action_a0_vs_joint_a0.csv"

# ── Hyperparameters (override via env) ───────────────────────────────────────
ANCHOR="${ANCHOR:-A0}"
EPOCHS="${EPOCHS:-200}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-4}"
HORIZON="${HORIZON:-16}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
AUG_ALPHA_S_MAX="${AUG_ALPHA_S_MAX:-0.05}"
AUG_ALPHA_A_MAX="${AUG_ALPHA_A_MAX:-0.20}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-0}"

# Evaluation grid
N_ROLLOUTS="${N_ROLLOUTS:-30}"
ALPHA_S_LIST="${ALPHA_S_LIST:-0.0 0.02 0.05}"
ALPHA_A_LIST="${ALPHA_A_LIST:-0.0 0.3}"
T_START_LIST="${T_START_LIST:-5 10}"

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
    echo "  Joint A0:  NOT FOUND — eval will compare baseline vs action-denoiser only"
    JOINT_CKPT=""
fi

# # ── Step 1: Train action-only denoiser ───────────────────────────────────────
# log "Step 1: Training action-only denoiser (anchor=${ANCHOR}, epochs=${EPOCHS})"

# cd "$REPO_ROOT"
# python -m diffusion.train_action_denoiser_lift \
#     --bc_rnn_ckpt     "$BC_RNN_CKPT" \
#     --hdf5_path       "$HDF5_PATH" \
#     --anchor          "$ANCHOR" \
#     --horizon         "$HORIZON" \
#     --diffusion_steps "$DIFFUSION_STEPS" \
#     --epochs          "$EPOCHS" \
#     --batch_size      "$BATCH_SIZE" \
#     --lr              "$LR" \
#     --aug_alpha_s_max "$AUG_ALPHA_S_MAX" \
#     --aug_alpha_a_max "$AUG_ALPHA_A_MAX" \
#     --output_path     "$ACTION_CKPT" \
#     --device          "$DEVICE" \
#     --seed            "$SEED" \
#     --log_every       10

# echo "  Saved: $ACTION_CKPT"

# ── Step 2: Evaluate ─────────────────────────────────────────────────────────
log "Step 2: Evaluating (${N_ROLLOUTS} rollouts per cell)"

# Build --joint_denoiser flag only if checkpoint exists
JOINT_FLAG=""
if [[ -n "$JOINT_CKPT" ]]; then
    JOINT_FLAG="--joint_denoiser $JOINT_CKPT"
fi

python evaluation/eval_action_denoiser_lift.py \
    --diffusion_checkpoint "$DIFFUSION_CKPT" \
    --action_denoiser      "$ACTION_CKPT" \
    $JOINT_FLAG \
    --alpha_s              $ALPHA_S_LIST \
    --alpha_a              $ALPHA_A_LIST \
    --t_start              $T_START_LIST \
    --n_rollouts           "$N_ROLLOUTS" \
    --output_csv           "$RESULTS_CSV"

log "Done — results saved to:"
echo "  $RESULTS_CSV"
