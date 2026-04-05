import h5py
import numpy as np
import torch
from tqdm import tqdm

import robomimic.utils.file_utils as FileUtils

# =========================
# CONFIG
# =========================
CKPT = "/home/axs0940/miniconda3/envs/vla_marl/lib/python3.10/site-packages/robomimic/checkpoints/bc_baseline/transport_bc_baseline/20260404203751/models/model_epoch_100.pth"
DATASET = "datasets/transport/ph/low_dim_v141.hdf5"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# LOAD POLICY
# =========================
policy, _ = FileUtils.policy_from_checkpoint(
    ckpt_path=CKPT,
    device=device,
    verbose=False
)
print("✅ Policy loaded")

# =========================
# LOAD DATASET
# =========================
f = h5py.File(DATASET, "r")
demos = list(f["data"].keys())

# Take a subset
NUM_DEMOS = 5
demos = demos[:NUM_DEMOS]

print(f"Evaluating on {NUM_DEMOS} demos")

# =========================
# METRICS
# =========================
l2_errors = []
cosine_sims = []

# =========================
# EVALUATION LOOP
# =========================
for demo in tqdm(demos):

    obs_group = f["data"][demo]["obs"]
    actions = f["data"][demo]["actions"]

    T = actions.shape[0]

    policy.start_episode()

    for t in range(T):

        # =========================
        # BUILD OBS (MATCH TRAINING)
        # =========================
        obs = {
            key: obs_group[key][t]
            for key in obs_group.keys()
        }

        # =========================
        # PREDICT ACTION
        # =========================
        pred_action = policy(obs)

        gt_action = actions[t]

        # =========================
        # METRICS
        # =========================
        l2 = np.linalg.norm(pred_action - gt_action)
        l2_errors.append(l2)

        # cosine similarity
        cos = np.dot(pred_action, gt_action) / (
            np.linalg.norm(pred_action) * np.linalg.norm(gt_action) + 1e-8
        )
        cosine_sims.append(cos)

# =========================
# RESULTS
# =========================
print("\n" + "="*50)
print("OFFLINE DATASET EVALUATION")
print("="*50)
print(f"Mean L2 Error:      {np.mean(l2_errors):.4f}")
print(f"Mean Cosine Sim:    {np.mean(cosine_sims):.4f}")
print("="*50)