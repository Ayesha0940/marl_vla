#!/usr/bin/env bash
# Ablation study evaluation for the Can task.
#
# Assumptions:
#   - Baselines have already been run and saved to results/can/joint_denoiser/baselines.csv
#   - All variant checkpoints have already been trained into diffusion_models/ablation_can/
#
# This script only performs the evaluation sweep for each variant.
#
# Usage:
#   bash training/ablation_noise_loss_can.sh
#   bash training/ablation_noise_loss_can.sh 2>&1 | tee ablation_can.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BC_RNN_CKPT="checkpoints/bc_rnn_can/bc_rnn_can/20260405211805/models/model_epoch_600.pth"

ANCHORS=(A0 A2 A3 A7)
N_ROLLOUTS=25
T_STARTS=(10)

RESULTS_DIR="results/can/joint_denoiser"
CKPT_DIR="diffusion_models/ablation_can"
mkdir -p "${RESULTS_DIR}"

export MUJOCO_GL=egl
PYTHON="python -u"

# ---------------------------------------------------------------------------
# Helper: build checkpoint list for all anchors in a variant
# ---------------------------------------------------------------------------
ckpts_for_variant() {
    local name="$1"
    local anchor

    for anchor in "${ANCHORS[@]}"; do
        printf '%s\n' "${CKPT_DIR}/joint_${name}_${anchor,,}.pt"
    done
}

# ---------------------------------------------------------------------------
# Eval each variant
# ---------------------------------------------------------------------------
run_variant() {
    local name="$1"

    echo ""
    echo "============================================================"
    echo "[VARIANT] ${name}"
    echo "============================================================"

    local ckpts=()
    mapfile -t ckpts < <(ckpts_for_variant "${name}")

    for ckpt in "${ckpts[@]}"; do
        if [[ ! -f "${ckpt}" ]]; then
            echo "[ERROR] Missing checkpoint: ${ckpt}" >&2
            echo "        Run ablation_noise_loss_setup_can.sh first." >&2
            return 1
        fi
    done

    echo "[EVAL ] ${name}  anchors=${ANCHORS[*]}  t_starts=${T_STARTS[*]}"
    $PYTHON evaluation/eval_joint_denoiser.py \
        --bc_rnn_ckpt  "${BC_RNN_CKPT}" \
        --joint_ckpts  "${ckpts[@]}" \
        --t_starts     "${T_STARTS[@]}" \
        --n_rollouts   "${N_ROLLOUTS}" \
        --output_csv   "${RESULTS_DIR}/${name}_results.csv"
}

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
run_variant "baseline"
run_variant "lam01"
run_variant "all_three"

echo ""
echo "All variants completed."
echo "Baselines:   ${RESULTS_DIR}/baselines.csv"
echo "Per-variant: ${RESULTS_DIR}/{variant}_results.csv"
