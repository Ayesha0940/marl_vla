#!/usr/bin/env bash
# BC RNN baseline evaluation with action and joint denoisers — Square task (NutAssemblySquare)
#
# For each anchor (A0, A7):
#   1. Train JointUNet1D          (400 epochs) if checkpoint is missing
#   2. Train ActionDenoisingUNet1D (400 epochs) if checkpoint is missing
#   3. Evaluate: baseline vs. action-denoiser vs. joint-denoiser
#
# Run from repo root:
#   bash run_bc_rnn_denoiser_square_all.sh
#
# Optional overrides via env:
#   EPOCHS=50    bash run_bc_rnn_denoiser_square_all.sh    # quick smoke test
#   N_ROLLOUTS=5 bash run_bc_rnn_denoiser_square_all.sh
#   DEVICE=cpu   bash run_bc_rnn_denoiser_square_all.sh

set -euo pipefail

# ── Shared paths ──────────────────────────────────────────────────────────────
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BC_RNN_CKPT="$REPO_ROOT/checkpoints/bc_rnn_square/seed3/bc_rnn_square_v3_seed3/20260418050524/models/model_epoch_800_NutAssemblySquare_success_0.8.pth"
HDF5_PATH="$REPO_ROOT/datasets/square/ph/low_dim_v141.hdf5"
RESULTS_DIR="$REPO_ROOT/results/square/bc_rnn_denoiser"

# ── Shared hyperparameters (override via env) ─────────────────────────────────
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
ALPHA_A_LIST="${ALPHA_A_LIST:-0.0 0.1 0.2}"
T_START_LIST="${T_START_LIST:-5}"
EPISODE_HORIZON="${EPISODE_HORIZON:-400}"

mkdir -p "$RESULTS_DIR"

# ── Helpers ───────────────────────────────────────────────────────────────────
log()  { echo -e "\n\033[1;36m==> $*\033[0m"; }
die()  { echo -e "\033[1;31mERROR: $*\033[0m" >&2; exit 1; }
hdr()  { echo -e "\n\033[1;35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"; \
         echo -e "\033[1;35m  $*\033[0m"; \
         echo -e "\033[1;35m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\033[0m"; }

# ── Preflight checks ──────────────────────────────────────────────────────────
log "Preflight checks"
[[ -f "$BC_RNN_CKPT" ]] || die "BC-RNN checkpoint not found: $BC_RNN_CKPT"
[[ -f "$HDF5_PATH" ]]   || die "HDF5 dataset not found: $HDF5_PATH"
echo "  BC-RNN: $BC_RNN_CKPT"
echo "  HDF5:   $HDF5_PATH"

# ── Anchor configs ────────────────────────────────────────────────────────────
# Format: "ANCHOR:JOINT_CKPT:ACTION_CKPT:RESULTS_CSV"
declare -a ANCHORS=(
    "A0:$REPO_ROOT/diffusion_models/joint_unet_square_a0.pt:$REPO_ROOT/diffusion_models/action_a0_square.pt:$RESULTS_DIR/bc_rnn_a0_vs_joint_a0.csv"
    "A7:$REPO_ROOT/diffusion_models/joint_unet_square_a7.pt:$REPO_ROOT/diffusion_models/action_a7_square.pt:$RESULTS_DIR/bc_rnn_a7_vs_joint_a7.csv"
)

# ── Main loop ─────────────────────────────────────────────────────────────────
for entry in "${ANCHORS[@]}"; do
    IFS=':' read -r ANCHOR JOINT_CKPT ACTION_CKPT RESULTS_CSV <<< "$entry"

    hdr "Square (BC RNN) — anchor ${ANCHOR}"

    # Step 1: Train joint UNet denoiser
    if [[ -f "$JOINT_CKPT" ]]; then
        log "Skipping joint UNet training — checkpoint exists: $JOINT_CKPT"
    else
        log "Training joint UNet denoiser (anchor=${ANCHOR}, epochs=${EPOCHS})"
        cd "$REPO_ROOT"
        python -m diffusion.train_joint_denoiser_unet \
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

    # Step 2: Train action-only denoiser
    if [[ -f "$ACTION_CKPT" ]]; then
        log "Skipping action denoiser training — checkpoint exists: $ACTION_CKPT"
    else
        log "Training action-only denoiser (anchor=${ANCHOR}, epochs=${EPOCHS})"
        cd "$REPO_ROOT"
        python -m diffusion.train_action_denoiser_square \
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

    # Step 3: Evaluate
    log "Evaluating (${N_ROLLOUTS} rollouts per cell)"
    cd "$REPO_ROOT"
    python evaluation/eval_bc_rnn_denoiser_square.py \
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
    echo "  Results: $RESULTS_CSV"
done

# ── Summary ───────────────────────────────────────────────────────────────────
hdr "All done"
echo "  Results written to:"
for entry in "${ANCHORS[@]}"; do
    IFS=':' read -r ANCHOR _ _ RESULTS_CSV <<< "$entry"
    echo "    [${ANCHOR}] $RESULTS_CSV"
done
