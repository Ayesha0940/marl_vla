#!/usr/bin/env bash
# Ablation study for three training improvements — Square task.
#
# Assumptions:
#   - Baselines have already been run and saved to results/square/joint_denoiser/baselines.csv
#   - All variant checkpoints have already been trained into diffusion_models/ablation_square/
#
# This script only performs the evaluation sweep for each variant.
#
# Usage:
#   bash training/square/ablation_noise_loss_square.sh
#   bash training/square/ablation_noise_loss_square.sh 2>&1 | tee ablation_square.log

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Activate the correct conda environment
CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate vla_marl

BC_RNN_CKPT="checkpoints/bc_rnn_square/seed1/bc_rnn_square_v3_seed1/20260418035142/models/model_epoch_800_NutAssemblySquare_success_0.8.pth"
DIFFUSION_CKPT="checkpoints/square_diffusion_policy_unet/best_model.pt"

ANCHORS=(A0 A7)
HORIZON=16
DIFFUSION_STEPS=100
EPOCHS=400
BATCH_SIZE=256
LR=1e-4
N_ROLLOUTS=25
T_STARTS=(5 10 15)

RESULTS_DIR="results/square/joint_denoiser"
CKPT_DIR="diffusion_models/ablation_square"
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
    local name="$1"; shift   # remaining args are extra eval flags

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

    # Eval each t_start separately (square eval takes --t_start singular)
    for t_start in "${T_STARTS[@]}"; do
        echo "[EVAL ] ${name}  anchors=${ANCHORS[*]}  t_start=${t_start}"
        $PYTHON evaluation/eval_diffusion_joint_denoiser_square.py \
            --env_ckpt             "${BC_RNN_CKPT}" \
            --diffusion_checkpoint "${DIFFUSION_CKPT}" \
            --joint_ckpts          "${ckpts[@]}" \
            --t_start              "${t_start}" \
            --n_rollouts           "${N_ROLLOUTS}" \
            --output_csv           "${RESULTS_DIR}/${name}_t${t_start}_results.csv"
    done
}

# ---------------------------------------------------------------------------
# Variant definitions
# ---------------------------------------------------------------------------
run_variant "baseline"

# run_variant "asym_noise" \
#     --noise_schedule asymmetric

run_variant "no_warmstart" \
    --no_warm_start

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
echo "Per-variant: ${RESULTS_DIR}/{variant}_t{t_start}_results.csv"
