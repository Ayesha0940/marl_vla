#!/usr/bin/env bash
# Train joint (state, action) denoiser variants for Square/NutAssemblySquare.
# Run from the project root inside the vla_marl conda env:
#   bash scripts/train_joint_denoiser_square.sh

set -euo pipefail

PYTHON="${PYTHON:-/home/axs0940/miniconda3/envs/vla_marl/bin/python}"

BC_RNN_CKPT="checkpoints/bc_rnn_square/seed3/bc_rnn_square_v3_seed3/20260418050524/models/model_epoch_800_NutAssemblySquare_success_0.8.pth"
HDF5="datasets/square/ph/low_dim_v141.hdf5"
OUT="diffusion_models/ablation_square"

mkdir -p "$OUT"

if [[ ! -f "$BC_RNN_CKPT" ]]; then
    echo "ERROR: BC-RNN checkpoint not found: $BC_RNN_CKPT"
    exit 1
fi

if [[ ! -f "$HDF5" ]]; then
    echo "ERROR: HDF5 not found: $HDF5"
    exit 1
fi

echo "============================================================"
echo "Joint denoiser training — Square task"
echo "BC-RNN ckpt: $BC_RNN_CKPT"
echo "HDF5:        $HDF5"
echo "Output dir:  $OUT"
echo "============================================================"

echo ""
echo "------------------------------------------------------------"
echo "  Training: joint_no_warmstart_a0  (anchor=A0, no warm-start)"
echo "------------------------------------------------------------"
"$PYTHON" -m diffusion.train_joint_denoiser \
    --bc_rnn_ckpt "$BC_RNN_CKPT" \
    --hdf5_path   "$HDF5" \
    --anchor      A0 \
    --no_warm_start \
    --epochs      400 \
    --output_path "$OUT/joint_no_warmstart_a0.pt"

echo ""
echo "------------------------------------------------------------"
echo "  Training: joint_no_warmstart_a7  (anchor=A7, no warm-start)"
echo "------------------------------------------------------------"
"$PYTHON" -m diffusion.train_joint_denoiser \
    --bc_rnn_ckpt "$BC_RNN_CKPT" \
    --hdf5_path   "$HDF5" \
    --anchor      A7 \
    --no_warm_start \
    --epochs      400 \
    --output_path "$OUT/joint_no_warmstart_a7.pt"

echo ""
echo "============================================================"
echo "Done. Checkpoints:"
ls "$OUT/"
echo "============================================================"
