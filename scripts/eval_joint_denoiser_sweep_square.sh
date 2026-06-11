#!/usr/bin/env bash
# Evaluate vanilla UNet vs joint denoiser variants under joint noise — Square task.
# Run from the project root inside the vla_marl conda env:
#   bash scripts/eval_joint_denoiser_sweep_square.sh

set -euo pipefail

PYTHON="${PYTHON:-/home/axs0940/miniconda3/envs/vla_marl/bin/python}"

DIFFUSION_CKPT="checkpoints/square_diffusion_policy_unet/best_model.pt"
A0_CKPT="diffusion_models/ablation_square/joint_no_warmstart_a0.pt"
RESULTS_DIR="results/square/joint_denoiser_comparison"

ALPHA_S="0.0 0.01 0.02"
ALPHA_A="0.05"
N_ROLLOUTS=25

mkdir -p "$RESULTS_DIR"

if [[ ! -f "$DIFFUSION_CKPT" ]]; then
    echo "ERROR: Vanilla diffusion checkpoint not found: $DIFFUSION_CKPT"
    echo "  Train first: python training/square/train_unet_diffusion_square.py"
    exit 1
fi

echo "============================================================"
echo "Joint denoiser sweep — Square task"
echo "Diffusion ckpt: $DIFFUSION_CKPT"
echo "alpha_s:        $ALPHA_S"
echo "alpha_a:        $ALPHA_A"
echo "n_rollouts:     $N_ROLLOUTS"
echo "results dir:    $RESULTS_DIR"
echo "============================================================"

# Build --joint_ckpts list from whichever checkpoints exist
JOINT_CKPTS=""
for CKPT in "$A0_CKPT"; do
    if [[ -f "$CKPT" ]]; then
        JOINT_CKPTS="$JOINT_CKPTS $CKPT"
    else
        echo "  [SKIP] denoiser checkpoint not found: $CKPT"
    fi
done

if [[ -z "$JOINT_CKPTS" ]]; then
    echo "  No denoiser checkpoints found — running baseline only."
fi

"$PYTHON" -u evaluation/eval_diffusion_joint_denoiser_square.py \
    --diffusion_checkpoint "$DIFFUSION_CKPT" \
    ${JOINT_CKPTS:+--joint_ckpts $JOINT_CKPTS} \
    --alpha_s $ALPHA_S \
    --alpha_a $ALPHA_A \
    --n_rollouts "$N_ROLLOUTS" \
    --horizon 500 \
    --output_csv "$RESULTS_DIR/joint_sweep_v2.csv"

echo ""
echo "============================================================"
echo "Done. Results in $RESULTS_DIR/"
ls "$RESULTS_DIR/" 2>/dev/null || echo "(no results yet)"
echo "============================================================"
