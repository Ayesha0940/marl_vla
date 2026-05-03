# #!/usr/bin/env bash

# set -euo pipefail

# SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# cd "${REPO_ROOT}"

# PYTHON="${PYTHON:-python}"
# BEST_CKPT="${BEST_CKPT:-/home/axs0940/marl_vla/checkpoints/lift_diffusion_policy/model_epoch_2500.pt}"
# FINAL_CKPT="${FINAL_CKPT:-/home/axs0940/marl_vla/checkpoints/lift_diffusion_policy/model_final.pt}"
# ENV_META_CKPT="${ENV_META_CKPT:-checkpoints/bc_rnn_lift/bc_rnn_lift/20260405174006/models/model_epoch_600.pth}"
# N_ROLLOUTS="${N_ROLLOUTS:-50}"
# HORIZON="${HORIZON:-400}"
# SEED="${SEED:-42}"
# T_START="${T_START:-}"

# run_eval() {
#   local ckpt_path="$1"
#   local label="$2"

#   if [[ ! -f "${ckpt_path}" ]]; then
#     echo "Checkpoint not found: ${ckpt_path}" >&2
#     return 1
#   fi

#   echo "============================================================"
#   echo "[EVAL] ${label}"
#   echo "Checkpoint: ${ckpt_path}"
#   echo "Env bootstrap checkpoint: ${ENV_META_CKPT}"
#   echo "============================================================"

#   args=(
#     evaluation/eval_diffusion_policy.py
#     --agent "${ckpt_path}"
#     --env_ckpt "${ENV_META_CKPT}"
#     --n_rollouts "${N_ROLLOUTS}"
#     --horizon "${HORIZON}"
#     --seed "${SEED}"
#   )

#   if [[ -n "${T_START}" ]]; then
#     args+=(--t_start "${T_START}")
#   fi

#   "${PYTHON}" "${args[@]}"
# }

# run_eval "${BEST_CKPT}" "best_model"
# run_eval "${FINAL_CKPT}" "model_final"


#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python}"
CKPT_DIR="${CKPT_DIR:-/home/axs0940/marl_vla/checkpoints/lift_diffusion_policy}"
ENV_META_CKPT="${ENV_META_CKPT:-checkpoints/bc_rnn_lift/bc_rnn_lift/20260405174006/models/model_epoch_600.pth}"
N_ROLLOUTS="${N_ROLLOUTS:-200}"
HORIZON="${HORIZON:-400}"
SEED="${SEED:-42}"
T_START="${T_START:-}"

echo "============================================================"
echo "[SWEEP] ${CKPT_DIR}"
echo "Env bootstrap checkpoint: ${ENV_META_CKPT}"
echo "N_ROLLOUTS: ${N_ROLLOUTS}"
echo "HORIZON:    ${HORIZON}"
echo "SEED:       ${SEED}"
echo "============================================================"

args=(
    evaluation/eval_diffusion_policy.py
    --sweep "${CKPT_DIR}"
    --env_ckpt "${ENV_META_CKPT}"
    --n_rollouts "${N_ROLLOUTS}"
    --horizon "${HORIZON}"
    --seed "${SEED}"
)

if [[ -n "${T_START}" ]]; then
    args+=(--t_start "${T_START}")
fi

"${PYTHON}" "${args[@]}"