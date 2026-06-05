#!/usr/bin/env bash
# Train the joint (state, action) denoiser for Square — anchor A7.
#
# Uses the best Square BC-RNN checkpoint (seed1, 80% success) for obs_key
# auto-detection, and trains directly on the demo HDF5 (no rollout collection
# needed).  Saves checkpoint to diffusion_models/joint_a7_square.pt.
#
# Usage:
#   bash training/square/train_joint_denoiser_square.sh
#   bash training/square/train_joint_denoiser_square.sh --epochs 400

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

BC_RNN_CKPT="checkpoints/bc_rnn_square/seed1/bc_rnn_square_v3_seed1/20260418035142/models/model_epoch_800_NutAssemblySquare_success_0.8.pth"
HDF5_PATH="datasets/square/ph/low_dim_v141.hdf5"
OUTPUT_PATH="diffusion_models/joint_a7_square.pt"

mkdir -p diffusion_models

export MUJOCO_GL=egl
PYTHON="python -u"

echo "============================================================"
echo "[TRAIN] Joint denoiser  anchor=A7  task=square"
echo "  bc_rnn_ckpt: ${BC_RNN_CKPT}"
echo "  hdf5_path:   ${HDF5_PATH}"
echo "  output:      ${OUTPUT_PATH}"
echo "============================================================"

$PYTHON -m diffusion.train_joint_denoiser \
    --bc_rnn_ckpt     "${BC_RNN_CKPT}" \
    --hdf5_path       "${HDF5_PATH}" \
    --anchor          A7 \
    --horizon         16 \
    --diffusion_steps 100 \
    --epochs          200 \
    --batch_size      256 \
    --lr              1e-4 \
    --noise_schedule  asymmetric \
    --output_path     "${OUTPUT_PATH}" \
    "$@"

echo ""
echo "Done. Checkpoint: ${OUTPUT_PATH}"
