#!/usr/bin/env bash
# One-time setup for the ablation study.
#
# This script runs the baselines once and trains all anchor checkpoints for
# each variant. The evaluation-only script can then be run multiple times
# without repeating these expensive steps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BC_RNN_CKPT="checkpoints/bc_rnn_lift/bc_rnn_lift/20260405174006/models/model_epoch_600.pth"
HDF5_PATH="datasets/lift/ph/low_dim_v141.hdf5"

ANCHORS=(A0 A2 A7)
HORIZON=16
DIFFUSION_STEPS=100
EPOCHS=200
BATCH_SIZE=256
LR=1e-4
N_ROLLOUTS=25

RESULTS_DIR="results/lift/joint_denoiser"
CKPT_DIR="diffusion_models/ablation"
mkdir -p "${RESULTS_DIR}" "${CKPT_DIR}"

export MUJOCO_GL=egl
PYTHON="python -u"

echo "============================================================"
echo "[BASELINES]"
echo "============================================================"
$PYTHON evaluation/eval_baselines.py \
    --bc_rnn_ckpt "${BC_RNN_CKPT}" \
    --n_rollouts  "${N_ROLLOUTS}" \
    --output_csv  "${RESULTS_DIR}/baselines.csv"

run_variant() {
    local name="$1"; shift

    echo ""
    echo "============================================================"
    echo "[TRAIN] ${name}"
    echo "============================================================"

    local anchor
    for anchor in "${ANCHORS[@]}"; do
        local ckpt="${CKPT_DIR}/joint_${name}_${anchor,,}.pt"
        echo "[TRAIN] ${name}  anchor=${anchor}"
        $PYTHON -m diffusion.train_joint_denoiser \
            --bc_rnn_ckpt     "${BC_RNN_CKPT}" \
            --hdf5_path       "${HDF5_PATH}" \
            --anchor          "${anchor}" \
            --horizon         "${HORIZON}" \
            --diffusion_steps "${DIFFUSION_STEPS}" \
            --epochs          "${EPOCHS}" \
            --batch_size      "${BATCH_SIZE}" \
            --lr              "${LR}" \
            --output_path     "${ckpt}" \
            "$@"
    done
}

# run_variant "baseline"

# run_variant "asym_noise" \
#     --noise_schedule asymmetric

# run_variant "no_warmstart" \
#     --no_warm_start

# run_variant "lam01" \
#     --lam 0.1

# run_variant "lam025" \
#     --lam 0.25

# run_variant "asym_lam01" \
#     --noise_schedule asymmetric \
#     --lam 0.1

# run_variant "all_three" \
#     --noise_schedule asymmetric \
#     --lam 0.1

echo ""
echo "Setup completed."
echo "Baselines:   ${RESULTS_DIR}/baselines.csv"
echo "Checkpoints: ${CKPT_DIR}/joint_{variant}_{anchor}.pt"