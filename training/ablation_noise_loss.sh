#!/usr/bin/env bash
# Ablation study for three training improvements.
#
# Assumptions:
#   - Baselines have already been run and saved to results/lift/joint_denoiser/baselines.csv
#   - All variant checkpoints have already been trained into diffusion_models/ablation/
#
# This script only performs the evaluation sweep for each variant.
#
# Usage:
#   bash training/ablation_noise_loss.sh
#   bash training/ablation_noise_loss.sh 2>&1 | tee ablation.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BC_RNN_CKPT="checkpoints/bc_rnn_lift/bc_rnn_lift/20260405174006/models/model_epoch_600.pth"
HDF5_PATH="datasets/lift/ph/low_dim_v141.hdf5"

ANCHORS=(A0 A7)
HORIZON=16
DIFFUSION_STEPS=100
EPOCHS=200
BATCH_SIZE=256
LR=1e-4
N_ROLLOUTS=50
T_STARTS=(10)

RESULTS_DIR="results/lift/joint_denoiser"
CKPT_DIR="diffusion_models/ablation"
mkdir -p "${RESULTS_DIR}" "${CKPT_DIR}"

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
    local name="$1"; shift   # remaining args are extra train flags

    echo ""
    echo "============================================================"
    echo "[VARIANT] ${name}"
    echo "============================================================"

    # Build checkpoint list for all anchors in this variant
    local ckpts=()
    mapfile -t ckpts < <(ckpts_for_variant "${name}")

    for ckpt in "${ckpts[@]}"; do
        if [[ ! -f "${ckpt}" ]]; then
            echo "[ERROR] Missing checkpoint: ${ckpt}" >&2
            echo "        Train the variant first or update CKPT_DIR." >&2
            return 1
        fi
    done

    # Single eval call: all anchors × all t_starts → one pivot CSV
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

# run_variant "asym_noise" \
#     --noise_schedule asymmetric

# run_variant "no_warmstart" \
#     --no_warm_start

run_variant "lam01" \
    --lam 0.1

# run_variant "lam025" \
#     --lam 0.25

# run_variant "asym_lam01" \
#     --noise_schedule asymmetric \
#     --lam 0.1

run_variant "all_three" \
    --noise_schedule asymmetric \
    --lam 0.1

echo ""
echo "All variants completed."
echo "Baselines:   ${RESULTS_DIR}/baselines.csv"
echo "Per-variant: ${RESULTS_DIR}/{variant}_results.csv"
