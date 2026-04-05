"""
Step 1: Behaviour Cloning (BC) on Transport Expert Demonstrations
=================================================================
Trains a decentralised BC policy for each agent independently.

Each agent sees:
  - Own proprioception (45 dims)
  - Shared object state (41 dims)
  = 86 dims total, stacked over obs_horizon timesteps

Actions: 7 dims per agent (from 14-dim joint action vector)

Usage:
    python step1_bc.py

Outputs:
    checkpoints/bc/agent0_policy.pt
    checkpoints/bc/agent1_policy.pt
    checkpoints/bc/norm_stats.npy   <- needed by all later steps
    bc_training_curves.png
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CFG = {
    "dataset_path": os.path.expanduser(
        "~/marl_vla/datasets/transport/ph/low_dim_v141.hdf5"
    ),
    "save_dir":     os.path.expanduser("~/marl_vla/checkpoints/bc/"),

    # Observation / action dims
    "obs_dim":      86,   # 45 proprio + 41 object state
    "action_dim":   7,    # per agent
    "n_agents":     2,
    "obs_horizon":  2,    # stack last N timesteps

    # Architecture
    "hidden_dim":   256,

    # Training
    "epochs":       150,
    "batch_size":   512,
    "lr":           1e-4,
    "weight_decay": 1e-5,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers":  2,
    "log_freq":     10,
}

# ─────────────────────────────────────────────
# OBSERVATION EXTRACTION
# ─────────────────────────────────────────────

def extract_obs_sequence(demo, agent_idx):
    """
    Extract all timestep observations for one agent from a demo.

    Returns:
        obs:     np.array [T, obs_dim]
        actions: np.array [T, action_dim]
    """
    obs_d   = demo["obs"]
    actions = demo["actions"][:]          # [T, 14]
    T       = actions.shape[0]
    prefix  = f"robot{agent_idx}"

    obs = np.concatenate([
        obs_d[f"{prefix}_joint_pos"][:],       # 7
        obs_d[f"{prefix}_joint_pos_cos"][:],   # 7
        obs_d[f"{prefix}_joint_pos_sin"][:],   # 7
        obs_d[f"{prefix}_joint_vel"][:],       # 7
        obs_d[f"{prefix}_eef_pos"][:],         # 3
        obs_d[f"{prefix}_eef_quat"][:],        # 4
        obs_d[f"{prefix}_eef_vel_lin"][:],     # 3
        obs_d[f"{prefix}_eef_vel_ang"][:],     # 3
        obs_d[f"{prefix}_gripper_qpos"][:],    # 2
        obs_d[f"{prefix}_gripper_qvel"][:],    # 2
        obs_d["object"][:],                    # 41 shared
    ], axis=1)                                 # [T, 86]

    agent_actions = actions[:, agent_idx * 7 : (agent_idx + 1) * 7]  # [T, 7]
    return obs, agent_actions


# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class BCDataset(Dataset):
    """
    Sliding-window (obs_history, action) pairs for one agent.
    Loads all demos into RAM — 200 demos * 714 steps * 86 dims ~ 120MB, fine.
    """

    def __init__(self, hdf5_path, obs_horizon, agent_idx, norm_stats=None):
        self.obs_list = []
        self.act_list = []

        with h5py.File(hdf5_path, "r") as f:
            for key in sorted(f["data"].keys()):
                demo          = f["data"][key]
                obs, actions  = extract_obs_sequence(demo, agent_idx)
                T             = obs.shape[0]

                for t in range(obs_horizon - 1, T):
                    window = obs[t - obs_horizon + 1 : t + 1]  # [H, 86]
                    self.obs_list.append(window)
                    self.act_list.append(actions[t])

        self.obs = np.array(self.obs_list, dtype=np.float32)  # [N, H, 86]
        self.act = np.array(self.act_list, dtype=np.float32)  # [N, 7]

        # Compute or apply normalisation stats
        if norm_stats is None:
            self.obs_mean = self.obs.mean(axis=(0, 1), keepdims=True)
            self.obs_std  = self.obs.std(axis=(0, 1), keepdims=True) + 1e-6
            self.act_mean = self.act.mean(axis=0, keepdims=True)
            self.act_std  = self.act.std(axis=0, keepdims=True) + 1e-6
        else:
            self.obs_mean = norm_stats["obs_mean"]
            self.obs_std  = norm_stats["obs_std"]
            self.act_mean = norm_stats["act_mean"]
            self.act_std  = norm_stats["act_std"]

        self.obs = (self.obs - self.obs_mean) / self.obs_std
        self.act = (self.act - self.act_mean) / self.act_std

        print(f"  Agent {agent_idx}: {len(self.obs):,} samples loaded")

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.obs[idx]),  # [H, 86]
            torch.FloatTensor(self.act[idx]),  # [7]
        )

    def get_norm_stats(self):
        return {
            "obs_mean": self.obs_mean,
            "obs_std":  self.obs_std,
            "act_mean": self.act_mean,
            "act_std":  self.act_std,
        }


# ─────────────────────────────────────────────
# POLICY NETWORK
# ─────────────────────────────────────────────

class BCPolicy(nn.Module):
    """
    MLP behaviour cloning policy.
    obs [B, H, obs_dim] -> action [B, action_dim]

    LayerNorm for stability, Mish activations, Tanh output
    to keep actions bounded in [-1, 1] normalised space.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, obs_horizon):
        super().__init__()
        inp = obs_dim * obs_horizon

        self.net = nn.Sequential(
            nn.Linear(inp, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Mish(),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Tanh(),
        )

    def forward(self, obs):
        """obs: [B, H, obs_dim]"""
        return self.net(obs.reshape(obs.shape[0], -1))


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train_agent(agent_idx, cfg, device):
    print(f"\n{'='*55}")
    print(f"  Training BC — Agent {agent_idx}")
    print(f"{'='*55}")

    # Dataset
    dataset    = BCDataset(cfg["dataset_path"], cfg["obs_horizon"], agent_idx)
    dataloader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # Model
    policy = BCPolicy(
        cfg["obs_dim"], cfg["action_dim"],
        cfg["hidden_dim"], cfg["obs_horizon"]
    ).to(device)

    opt = Adam(policy.parameters(), lr=cfg["lr"],
               weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=cfg["epochs"], eta_min=1e-6
    )

    losses = []

    for epoch in trange(cfg["epochs"], desc=f"Agent {agent_idx}"):
        ep_loss = []

        for obs_b, act_b in dataloader:
            obs_b = obs_b.to(device)
            act_b = act_b.to(device)

            pred  = policy(obs_b)
            loss  = nn.functional.mse_loss(pred, act_b)

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

            ep_loss.append(loss.item())

        scheduler.step()
        losses.append(np.mean(ep_loss))

        if (epoch + 1) % cfg["log_freq"] == 0:
            print(f"    Epoch {epoch+1:4d}/{cfg['epochs']} | "
                  f"Loss: {losses[-1]:.6f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.2e}")

    return policy, dataset.get_norm_stats(), losses


def train():
    cfg    = CFG
    device = torch.device(cfg["device"])
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg["save_dir"], exist_ok=True)

    all_losses = []
    policies   = []
    all_stats  = []

    for agent_idx in range(cfg["n_agents"]):
        policy, stats, losses = train_agent(agent_idx, cfg, device)
        policies.append(policy)
        all_stats.append(stats)
        all_losses.append(losses)

        # Save policy weights
        torch.save(
            policy.state_dict(),
            os.path.join(cfg["save_dir"], f"agent{agent_idx}_policy.pt")
        )

    # Save normalisation stats (shared shape, per-agent)
    np.save(
        os.path.join(cfg["save_dir"], "norm_stats.npy"),
        {"agent0": all_stats[0], "agent1": all_stats[1]},
        allow_pickle=True
    )
    print(f"\nSaved weights and norm stats to {cfg['save_dir']}")

    # Save config for later steps
    np.save(
        os.path.join(cfg["save_dir"], "cfg.npy"),
        cfg, allow_pickle=True
    )

    # ── Plot ──────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for i, ax in enumerate(axes):
        ax.plot(all_losses[i], alpha=0.5, label="per epoch")
        w = 10
        if len(all_losses[i]) >= w:
            smooth = np.convolve(all_losses[i],
                                 np.ones(w) / w, mode="valid")
            ax.plot(range(w - 1, len(all_losses[i])),
                    smooth, label=f"{w}-epoch avg", linewidth=2)
        ax.set_title(f"Agent {i} BC Loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("MSE Loss")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("Step 1: Behaviour Cloning Training Curves", fontsize=13)
    plt.tight_layout()
    out = os.path.expanduser("~/marl_vla/bc_training_curves.png")
    plt.savefig(out, dpi=150)
    print(f"Saved training curves to {out}")

    # ── Quick sanity check ────────────────────
    print("\n── Sanity Check ──")
    print("Loading norm stats back...")
    stats_loaded = np.load(
        os.path.join(cfg["save_dir"], "norm_stats.npy"),
        allow_pickle=True
    ).item()
    for i in range(2):
        p = BCPolicy(
            cfg["obs_dim"], cfg["action_dim"],
            cfg["hidden_dim"], cfg["obs_horizon"]
        ).to(device)
        p.load_state_dict(
            torch.load(
                os.path.join(cfg["save_dir"], f"agent{i}_policy.pt"),
                map_location=device
            )
        )
        p.eval()
        dummy = torch.zeros(1, cfg["obs_horizon"],
                            cfg["obs_dim"]).to(device)
        out_a = p(dummy)
        print(f"  Agent {i} policy loaded OK — "
              f"output shape: {out_a.shape}, "
              f"final loss: {all_losses[i][-1]:.6f}")

    print("\nStep 1 complete. Run step2_td3bc.py next.")


if __name__ == "__main__":
    train()