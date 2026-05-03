#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
CONFIG="${CONFIG:-configs/diffusion_lift.json}"
NUM_EPOCHS="${NUM_EPOCHS:-4000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
OBS_HORIZON="${OBS_HORIZON:-2}"
ACTION_HORIZON="${ACTION_HORIZON:-8}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
DIFFUSION_STEPS="${DIFFUSION_STEPS:-100}"
LR="${LR:-1e-4}"
DEVICE="${DEVICE:-auto}"
SEED="${SEED:-0}"

ARGS=(
  --num_epochs "${NUM_EPOCHS}"
  --batch_size "${BATCH_SIZE}"
  --obs_horizon "${OBS_HORIZON}"
  --action_horizon "${ACTION_HORIZON}"
  --hidden_dim "${HIDDEN_DIM}"
  --diffusion_steps "${DIFFUSION_STEPS}"
  --lr "${LR}"
  --device "${DEVICE}"
  --seed "${SEED}"
)

if [[ -f "${CONFIG}" ]]; then
  echo "Using config: ${CONFIG}"
  ARGS+=(--config "${CONFIG}")
else
  echo "Config not found at ${CONFIG}; the trainer will generate its default config."
fi

echo "Starting custom Lift diffusion-policy training"
exec "${PYTHON}" training/lift/train_diffusion_lift.py "${ARGS[@]}"
