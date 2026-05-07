#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

# Ensure top-level package imports work when running the script directly
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:}${PYTHONPATH:-}"
export PYTHONPATH

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/diffusion_lift.json}"
NUM_EPOCHS="${NUM_EPOCHS:-4500}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACTION_HORIZON="${ACTION_HORIZON:-16}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
LR="${LR:-1e-4}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-0}"
BACKBONE="${BACKBONE:-unet}"

ARGS=(
  --num_epochs "${NUM_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --obs_horizon "${OBS_HORIZON}"
  --action_horizon "${ACTION_HORIZON}"
  --diffusion_steps "${DIFFUSION_STEPS}"
  --lr "${LR}"
  --device "${DEVICE}"
  --seed "${SEED}"
  --backbone "${BACKBONE}"
)

if [[ -f "${CONFIG}" ]]; then
  echo "Using config: ${CONFIG}"
  ARGS+=(--config "${CONFIG}")
else
  echo "Config not found at ${CONFIG}; the trainer will generate its default config."
fi

echo "Starting Lift diffusion-policy training  [backbone=${BACKBONE}]"
exec "${PYTHON}" training/lift/train_diffusion_lift.py "${ARGS[@]}"