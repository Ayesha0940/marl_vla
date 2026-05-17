#!/usr/bin/env bash
# Evaluate vanilla UNet vs SA-MDP variants under obs noise.
# Run from the project root inside the vla_marl conda env:
#   bash scripts/eval_samdp_sweep.sh

set -euo pipefail

RESULTS_DIR="results/lift/samdp_comparison"
mkdir -p "$RESULTS_DIR"

# Noise levels for the sweep (space-separated, passed as --alpha_s / --alpha_a).
# The dominant eval point is 0.05, matching aug_alpha_s_max and SA-MDP sigma_max.
ALPHA_S="0.0 0.01 0.02 0.03 0.04 0.05 0.1 0.2"
ALPHA_A="0.0 0.05 0.1 0.2"
N_ROLLOUTS=25

# Which mode(s) to run: state_only | action_only | joint | all
# Set to "state_only" for a quick state-noise-only sweep; "all" for the full grid.
MODE="state_only"

declare -A CHECKPOINTS=(
    ["vanilla"]="checkpoints/lift_diffusion_policy_v5/best_model.pt"
    ["samdp_k03"]="checkpoints/lift_samdp_k03/best_model.pt"
    # ["samdp_k10"]="checkpoints/lift_samdp_k10/best_model.pt"   # train first
    # ["samdp_k30"]="checkpoints/lift_samdp_k30/best_model.pt"   # train first
)

echo "============================================================"
echo "SA-MDP evaluation sweep"
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

    python -u evaluation/eval_noise_modes.py \
        --checkpoint "$CKPT" \
        --alpha_s $ALPHA_S \
        --alpha_a $ALPHA_A \
        --n_rollouts "$N_ROLLOUTS" \
        --mode "$MODE" \
        --output_csv "$RESULTS_DIR/${TAG}_sweep.csv"
done

echo ""
echo "============================================================"
echo "All done. Results in $RESULTS_DIR/"
ls "$RESULTS_DIR/"
echo "============================================================"
