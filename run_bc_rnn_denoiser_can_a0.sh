#!/usr/bin/env bash
# BC RNN baseline evaluation with action and joint denoisers — Can task, anchor A0
#
# Steps:
#   1. Train ActionDenoisingUNet1D (400 epochs) if checkpoint is missing
#   2. Train JointUNet1D          (400 epochs) if checkpoint is missing
#   3. Evaluate: baseline vs. action-denoiser vs. joint-denoiser
#
# Run from repo root:
#   bash run_bc_rnn_denoiser_can_a0.sh
#
# Optional overrides via env:
#   EPOCHS=50    bash run_bc_rnn_denoiser_can_a0.sh    # quick smoke test
#   N_ROLLOUTS=5 bash run_bc_rnn_denoiser_can_a0.sh
#   DEVICE=cpu   bash run_bc_rnn_denoiser_can_a0.sh

set -euo pipefail

# ── Paths ────────────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BC_RNN_CKPT="$REPO_ROOT/checkpoints/bc_rnn_can/bc_rnn_can/20260420182529/models/model_epoch_600.pth"
HDF5_PATH="$REPO_ROOT/datasets/can/ph/low_dim_v141.hdf5"
ACTION_CKPT="$REPO_ROOT/diffusion_models/action_a0_can.pt"
JOINT_CKPT="$REPO_ROOT/diffusion_models/joint_a0_can.pt"
RESULTS_DIR="$REPO_ROOT/results/can/bc_rnn_denoiser"
RESULTS_CSV="$RESULTS_DIR/bc_rnn_a0_results.csv"

# ── Hyperparameters (override via env) ───────────────────────────────────────
ANCHOR="${ANCHOR:-A0}"
EPOCHS="${EPOCHS:-400}"
BATCH_SIZE="${BATCH_SIZE:-256}"
LR="${LR:-1e-4}"
HORIZON="${HORIZON:-8}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
AUG_ALPHA_S_MAX="${AUG_ALPHA_S_MAX:-0.05}"
AUG_ALPHA_A_MAX="${AUG_ALPHA_A_MAX:-0.20}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-0}"

# Evaluation grid
N_ROLLOUTS="${N_ROLLOUTS:-30}"
ALPHA_S_LIST="${ALPHA_S_LIST:-0.0 0.01 0.02}"
ALPHA_A_LIST="${ALPHA_A_LIST:-0.0 0.2 0.3}"
T_START_LIST="${T_START_LIST:-5}"
EPISODE_HORIZON="${EPISODE_HORIZON:-400}"

mkdir -p "$RESULTS_DIR"

# ── Helpers ──────────────────────────────────────────────────────────────────
log() { echo -e "\n\033[1;36m==> $*\033[0m"; }
die() { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }

# ── Preflight checks ─────────────────────────────────────────────────────────
log "Preflight checks"
[[ -f "$BC_RNN_CKPT" ]] || die "BC-RNN checkpoint not found: $BC_RNN_CKPT"
[[ -f "$HDF5_PATH" ]]   || die "HDF5 dataset not found: $HDF5_PATH"
echo "  BC-RNN:  $BC_RNN_CKPT"
echo "  HDF5:    $HDF5_PATH"

# ── Step 1: Train action denoiser (skip if checkpoint already exists) ─────────
if [[ -f "$ACTION_CKPT" ]]; then
    log "Step 1: Action denoiser already exists — skipping training"
    echo "  $ACTION_CKPT"
else
    log "Step 1: Training action denoiser (anchor=${ANCHOR}, epochs=${EPOCHS})"
    cd "$REPO_ROOT"
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
        --output_path     "$ACTION_CKPT" \
        --device          "$DEVICE" \
        --seed            "$SEED" \
        --log_every       10
    echo "  Saved: $ACTION_CKPT"
fi

# ── Step 2: Train joint denoiser (skip if checkpoint already exists) ──────────
if [[ -f "$JOINT_CKPT" ]]; then
    log "Step 2: Joint denoiser already exists — skipping training"
    echo "  $JOINT_CKPT"
else
    log "Step 2: Training joint denoiser (anchor=${ANCHOR}, epochs=${EPOCHS})"
    cd "$REPO_ROOT"
    python -m diffusion.train_joint_denoiser \
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
        --output_path     "$JOINT_CKPT" \
        --device          "$DEVICE" \
        --seed            "$SEED" \
        --log_every       10
    echo "  Saved: $JOINT_CKPT"
fi

# ── Step 3: Evaluate ──────────────────────────────────────────────────────────
log "Step 3: Evaluating BC RNN baseline + denoisers (${N_ROLLOUTS} rollouts per cell)"

cd "$REPO_ROOT"
python evaluation/eval_bc_rnn_denoiser_can.py \
    --bc_rnn_checkpoint  "$BC_RNN_CKPT" \
    --action_denoiser    "$ACTION_CKPT" \
    --joint_denoiser     "$JOINT_CKPT" \
    --alpha_s            $ALPHA_S_LIST \
    --alpha_a            $ALPHA_A_LIST \
    --t_start            $T_START_LIST \
    --n_rollouts         "$N_ROLLOUTS" \
    --episode_horizon    "$EPISODE_HORIZON" \
    --base_seed          "$SEED" \
    --device             "$DEVICE" \
    --output_csv         "$RESULTS_CSV"

log "Done — results saved to:"
echo "  $RESULTS_CSV"
