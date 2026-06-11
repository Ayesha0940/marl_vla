#!/usr/bin/env bash
# Evaluate vanilla UNet vs SA-MDP variants under obs noise — Square task.
# Run from the project root inside the vla_marl conda env:
#   bash scripts/eval_samdp_sweep_square.sh

set -euo pipefail

PYTHON="${PYTHON:-/home/axs0940/miniconda3/envs/vla_marl/bin/python}"

RESULTS_DIR="results/square/samdp_comparison"
mkdir -p "$RESULTS_DIR"

ALPHA_S="0.0 0.01 0.02"
ALPHA_A="0.0 0.1 0.2 0.3"
N_ROLLOUTS=50

# Which mode(s) to run: state_only | action_only | joint | all
MODE="joint"

declare -A CHECKPOINTS=(
    ["vanilla"]="checkpoints/square_diffusion_policy_unet/best_model.pt"
    ["samdp_k03"]="checkpoints/square_samdp_k03/best_model.pt"
    # ["samdp_k10"]="checkpoints/square_samdp_k10/best_model.pt"   # train first
    # ["samdp_k30"]="checkpoints/square_samdp_k30/best_model.pt"   # train first
)

echo "============================================================"
echo "SA-MDP evaluation sweep — Square task"
echo "mode:       $MODE"
echo "alpha_s:    $ALPHA_S"
echo "alpha_a:    $ALPHA_A"
echo "n_rollouts: $N_ROLLOUTS"
echo "results dir: $RESULTS_DIR"
echo "============================================================"

for TAG in "${!CHECKPOINTS[@]}"; do
    CKPT="${CHECKPOINTS[$TAG]}"

    if [[ ! -f "$CKPT" ]]; then
        echo ""
        echo "  [SKIP] $TAG — checkpoint not found: $CKPT"
        continue
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "  Evaluating: $TAG"
    echo "  Checkpoint: $CKPT"
    echo "------------------------------------------------------------"

    "$PYTHON" -u evaluation/eval_noise_modes_square.py \
        --checkpoint "$CKPT" \
        --alpha_s $ALPHA_S \
        --alpha_a $ALPHA_A \
        --n_rollouts "$N_ROLLOUTS" \
        --mode "$MODE" \
        --output_csv "$RESULTS_DIR/${TAG}_sweep"
done

echo ""
echo "============================================================"
echo "All done. Results in $RESULTS_DIR/"
ls "$RESULTS_DIR/" 2>/dev/null || echo "(no results yet)"
echo "============================================================"
