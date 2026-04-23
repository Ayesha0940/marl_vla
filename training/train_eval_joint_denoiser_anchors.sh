#!/usr/bin/env bash
set -euo pipefail

# Train joint denoiser for anchors A1..A8, then evaluate each checkpoint
# at t_start values 10, 20, 30, 40, 50.

# Ensure all relative paths and python module imports resolve from repo root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BC_RNN_CKPT="checkpoints/bc_rnn_lift/bc_rnn_lift/20260405174006/models/model_epoch_600.pth"
HDF5_PATH="datasets/lift/ph/low_dim_v141.hdf5"

HORIZON=16
DIFFUSION_STEPS=100
EPOCHS=200
BATCH_SIZE=256
LR=1e-4

N_ROLLOUTS=25

ANCHORS=(A1 A2 A3 A4 A5 A6 A7 A8)
T_STARTS=(10 20 30)

mkdir -p diffusion_models

for anchor in "${ANCHORS[@]}"; do
  anchor_lower="${anchor,,}"
  output_path="diffusion_models/joint_${anchor_lower}_lift.pt"

  echo "============================================================"
  echo "[TRAIN] anchor=${anchor} -> ${output_path}"

  # python -m diffusion.train_joint_denoiser \
  #   --bc_rnn_ckpt "${BC_RNN_CKPT}" \
  #   --hdf5_path "${HDF5_PATH}" \
  #   --anchor "${anchor}" \
  #   --horizon "${HORIZON}" \
  #   --diffusion_steps "${DIFFUSION_STEPS}" \
  #   --epochs "${EPOCHS}" \
  #   --batch_size "${BATCH_SIZE}" \
  #   --lr "${LR}" \
  #   --output_path "${output_path}"

  for t_start in "${T_STARTS[@]}"; do
    echo "[EVAL ] anchor=${anchor}, t_start=${t_start}, ckpt=${output_path}"

    python evaluation/eval_joint_denoiser.py \
      --joint_ckpt "${output_path}" \
      --anchor "${anchor}" \
      --n_rollouts "${N_ROLLOUTS}" \
      --t_start "${t_start}"
  done
done

echo "All anchor trainings and evaluations completed."