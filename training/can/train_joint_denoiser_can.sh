#!/usr/bin/env bash
# Train CrossAttnJointDenoiser on Can task (two variants)
set -e

BC_RNN=checkpoints/bc_rnn_can/bc_rnn_can/20260405211805/models/model_epoch_600.pth
HDF5=datasets/can/ph/low_dim_v141.hdf5

# Primary model: A7 anchor + inverse/forward dynamics auxiliaries
python -m diffusion.train_joint_denoiser \
    --bc_rnn_ckpt $BC_RNN \
    --hdf5_path $HDF5 \
    --anchor A7 \
    --lam_inv 0.30 \
    --lam_dyn 0.05 \
    --epochs 600 \
    --output_path diffusion_models/joint_cross_attn_can_a7.pt

# Ablation: A0 anchor, no aux losses (backbone-only test)
python -m diffusion.train_joint_denoiser \
    --bc_rnn_ckpt $BC_RNN \
    --hdf5_path $HDF5 \
    --anchor A0 \
    --lam_inv 0.0 \
    --lam_dyn 0.0 \
    --epochs 600 \
    --output_path diffusion_models/joint_cross_attn_can_a0_noaux.pt

# A0 anchor + fixed aux — fills {U-Net, x-attn} × {no-aux, aux} factorial corner
python -m diffusion.train_joint_denoiser \
    --bc_rnn_ckpt $BC_RNN \
    --hdf5_path $HDF5 \
    --anchor A0 \
    --lam_inv 0.30 \
    --lam_dyn 0.05 \
    --epochs 600 \
    --output_path diffusion_models/joint_cross_attn_can_a0_aux.pt
