"""
BC Ablation Study — All Four Improvement Variants
==================================================
Trains all BC variants as controlled experiments.
Each variant saves to its own checkpoint directory.

Variants:
  baseline    — MLP, obs_horizon=2,  hidden=256, no aug
  horizon16   — MLP, obs_horizon=16, hidden=256, no aug
  hidden512   — MLP, obs_horizon=2,  hidden=512, no aug
  augmented   — MLP, obs_horizon=2,  hidden=256, obs noise aug
  diffusion   — DDPM, obs_horizon=16, hidden=256

Usage:
    python bc_ablation.py --variant baseline
    python bc_ablation.py --variant horizon16
    python bc_ablation.py --variant hidden512
    python bc_ablation.py --variant augmented
    python bc_ablation.py --variant diffusion
    python bc_ablation.py --variant all   # run all sequentially

Outputs per variant:
    checkpoints/bc_{variant}/agent{i}_policy.pt
    checkpoints/bc_{variant}/norm_stats.npy
    checkpoints/bc_{variant}/cfg.npy
    results/bc_ablation_{variant}.csv
    results/bc_ablation_comparison.png  (after all variants)
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
import h5py
from tqdm import trange
import matplotlib.pyplot as plt
from diffusers import DDPMScheduler

# ─────────────────────────────────────────────
# VARIANT CONFIGS
# ─────────────────────────────────────────────

VARIANTS = {
    "baseline": {
        "obs_horizon":   2,
        "hidden_dim":    256,
        "obs_noise_std": 0.0,
        "use_diffusion": False,
        "epochs":        150,
        "batch_size":    512,
        "lr":            1e-4,
        "label":         "BC Baseline (MLP, H=2, hidden=256)",
    },
    "horizon16": {
        "obs_horizon":   16,
        "hidden_dim":    256,
        "obs_noise_std": 0.0,
        "use_diffusion": False,
        "epochs":        150,
        "batch_size":    512,
        "lr":            1e-4,
        "label":         "BC Horizon-16 (MLP, H=16, hidden=256)",
    },
    "hidden512": {
        "obs_horizon":   2,
        "hidden_dim":    512,
        "obs_noise_std": 0.0,
        "use_diffusion": False,
        "epochs":        150,
        "batch_size":    512,
        "lr":            1e-4,
        "label":         "BC Hidden-512 (MLP, H=2, hidden=512)",
    },
    "augmented": {
        "obs_horizon":   2,
        "hidden_dim":    256,
        "obs_noise_std": 0.05,   # Gaussian std on obs during training
        "use_diffusion": False,
        "epochs":        150,
        "batch_size":    512,
        "lr":            1e-4,
        "label":         "BC Augmented (MLP, H=2, obs_noise=0.05)",
    },
    "diffusion": {
        "obs_horizon":   16,
        "hidden_dim":    256,
        "obs_noise_std": 0.0,
        "use_diffusion": True,
        "num_diffusion_steps": 100,
        "epochs":        150,
        "batch_size":    256,    # smaller batch for diffusion
        "lr":            1e-4,
        "label":         "BC Diffusion (DDPM, H=16, hidden=256)",
    },
}

BASE_CFG = {
    "dataset_path": os.path.expanduser(
        "~/marl_vla/datasets/transport/ph/low_dim_v141.hdf5"
    ),
    "results_dir":  os.path.expanduser("~/marl_vla/results/"),
    "obs_dim":      86,
    "action_dim":   7,
    "n_agents":     2,
    "weight_decay": 1e-5,
    "num_workers":  2,
    "log_freq":     10,
    "device":       "cuda" if torch.cuda.is_available() else "cpu",
}

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────

class BCDataset(Dataset):
    def __init__(self, hdf5_path, obs_horizon, agent_idx,
                 obs_noise_std=0.0, norm_stats=None, use_action_norm=True):
        self.use_action_norm = use_action_norm
        self.obs_noise_std = obs_noise_std
        self.obs_list = []
        self.act_list = []

        with h5py.File(hdf5_path, "r") as f:
            for key in sorted(f["data"].keys()):
                demo   = f["data"][key]
                obs_d  = demo["obs"]
                acts   = demo["actions"][:]
                T      = acts.shape[0]
                prefix = f"robot{agent_idx}"

                obs = np.concatenate([
                    obs_d[f"{prefix}_joint_pos"][:],
                    obs_d[f"{prefix}_joint_pos_cos"][:],
                    obs_d[f"{prefix}_joint_pos_sin"][:],
                    obs_d[f"{prefix}_joint_vel"][:],
                    obs_d[f"{prefix}_eef_pos"][:],
                    obs_d[f"{prefix}_eef_quat"][:],
                    obs_d[f"{prefix}_eef_vel_lin"][:],
                    obs_d[f"{prefix}_eef_vel_ang"][:],
                    obs_d[f"{prefix}_gripper_qpos"][:],
                    obs_d[f"{prefix}_gripper_qvel"][:],
                    obs_d["object"][:],
                ], axis=1)  # [T, 86]

                agent_acts = acts[:, agent_idx*7:(agent_idx+1)*7]

                for t in range(obs_horizon - 1, T):
                    self.obs_list.append(obs[t-obs_horizon+1:t+1])
                    self.act_list.append(agent_acts[t])

        self.obs = np.array(self.obs_list, dtype=np.float32)
        self.act = np.array(self.act_list, dtype=np.float32)

        # Normalise
        if norm_stats is None:
            self.obs_mean = self.obs.mean(axis=(0,1), keepdims=True)
            self.obs_std  = self.obs.std(axis=(0,1),  keepdims=True) + 1e-6
            self.act_mean = self.act.mean(axis=0, keepdims=True)
            self.act_std  = self.act.std(axis=0,  keepdims=True) + 1e-6
        else:
            self.obs_mean = norm_stats["obs_mean"]
            self.obs_std  = norm_stats["obs_std"]
            self.act_mean = norm_stats["act_mean"]
            self.act_std  = norm_stats["act_std"]

        self.obs = (self.obs - self.obs_mean) / self.obs_std
        if self.use_action_norm:
            self.act = (self.act - self.act_mean) / self.act_std
        else:
            # Diffusion Policy: do NOT normalise actions
            # Actions are already in [-1,1], DDPM diverges with small-std normalisation
            self.act_mean = np.zeros_like(self.act_mean)
            self.act_std  = np.ones_like(self.act_std)

        print(f"  Agent {agent_idx}: {len(self.obs):,} samples "
              f"(H={obs_horizon}, noise={obs_noise_std})")

    def __len__(self): return len(self.obs)

    def __getitem__(self, idx):
        obs = torch.FloatTensor(self.obs[idx])
        act = torch.FloatTensor(self.act[idx])
        # Apply obs noise augmentation at sample time
        if self.obs_noise_std > 0:
            obs = obs + torch.randn_like(obs) * self.obs_noise_std
        return obs, act

    def get_norm_stats(self):
        return {"obs_mean": self.obs_mean, "obs_std":  self.obs_std,
                "act_mean": self.act_mean, "act_std":  self.act_std}


# ─────────────────────────────────────────────
# MLP POLICY
# ─────────────────────────────────────────────

class MLPPolicy(nn.Module):
    """Standard MLP BC policy."""
    def __init__(self, obs_dim, action_dim, hidden_dim, obs_horizon):
        super().__init__()
        inp = obs_dim * obs_horizon
        self.net = nn.Sequential(
            nn.Linear(inp, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Mish(),
            nn.Linear(hidden_dim // 2, action_dim), nn.Tanh(),
        )
    def forward(self, obs):
        return self.net(obs.reshape(obs.shape[0], -1))


# ─────────────────────────────────────────────
# DIFFUSION POLICY
# ─────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Timestep embedding — identical to your MPE DDPM code."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        device = t.device
        half   = self.dim // 2
        emb    = torch.log(torch.tensor(10000.0)) / (half - 1)
        emb    = torch.exp(torch.arange(half, device=device) * -emb)
        emb    = t[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class DiffusionPolicyNet(nn.Module):
    """
    DDPM denoising network for single-step action prediction.

    Conditions on obs history, predicts noise added to action.
    This directly reuses your MPE TrajectoryDiffusion architecture
    but applied per-step (not trajectory-level).

    Input:  noisy_action [B, action_dim] + timestep [B] + obs [B, H, obs_dim]
    Output: predicted noise [B, action_dim]
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, obs_horizon):
        super().__init__()
        obs_flat = obs_dim * obs_horizon

        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
        )
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_flat, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.Mish(),
        )
        self.net = nn.Sequential(
            nn.Linear(action_dim + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2), nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, noisy_action, timestep, obs):
        B      = noisy_action.shape[0]
        t_emb  = self.time_emb(timestep)
        o_emb  = self.obs_enc(obs.reshape(B, -1))
        x      = torch.cat([noisy_action, t_emb, o_emb], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def sample(self, obs, scheduler, device):
        """
        Full reverse diffusion to generate a clean action from noise.
        Mirrors your MPE diffusion_denoise_action() exactly.
        """
        B      = obs.shape[0]
        action = torch.randn(B, 7).to(device)
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)

        for t in scheduler.timesteps:
            t_batch    = t.unsqueeze(0).expand(B).to(device)
            noise_pred = self(action, t_batch, obs)
            action     = scheduler.step(noise_pred, t, action).prev_sample

        return action.clamp(-1, 1)


# ─────────────────────────────────────────────
# TRAINING — MLP VARIANT
# ─────────────────────────────────────────────

def train_mlp(agent_idx, cfg, variant_cfg, device, save_dir):
    dataset = BCDataset(
        cfg["dataset_path"], variant_cfg["obs_horizon"],
        agent_idx, variant_cfg["obs_noise_std"]
    )
    loader = DataLoader(dataset, batch_size=variant_cfg["batch_size"],
                        shuffle=True, num_workers=cfg["num_workers"],
                        pin_memory=(device.type == "cuda"))

    policy = MLPPolicy(
        cfg["obs_dim"], cfg["action_dim"],
        variant_cfg["hidden_dim"], variant_cfg["obs_horizon"]
    ).to(device)

    opt = Adam(policy.parameters(), lr=variant_cfg["lr"],
               weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=variant_cfg["epochs"], eta_min=1e-6)

    losses = []
    for epoch in trange(variant_cfg["epochs"],
                        desc=f"  Agent {agent_idx} MLP"):
        ep = []
        for obs_b, act_b in loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            loss = nn.functional.mse_loss(policy(obs_b), act_b)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step(); ep.append(loss.item())
        sched.step()
        losses.append(np.mean(ep))
        if (epoch + 1) % cfg["log_freq"] == 0:
            print(f"    Epoch {epoch+1:4d} | Loss: {losses[-1]:.6f}")

    torch.save(policy.state_dict(),
               os.path.join(save_dir, f"agent{agent_idx}_policy.pt"))
    return dataset.get_norm_stats(), losses


# ─────────────────────────────────────────────
# TRAINING — DIFFUSION VARIANT
# ─────────────────────────────────────────────

def train_diffusion(agent_idx, cfg, variant_cfg, device, save_dir):
    dataset = BCDataset(
        cfg["dataset_path"], variant_cfg["obs_horizon"],
        agent_idx, obs_noise_std=0.0, use_action_norm=False
    )
    loader = DataLoader(dataset, batch_size=variant_cfg["batch_size"],
                        shuffle=True, num_workers=cfg["num_workers"],
                        pin_memory=(device.type == "cuda"))

    policy = DiffusionPolicyNet(
        cfg["obs_dim"], cfg["action_dim"],
        variant_cfg["hidden_dim"], variant_cfg["obs_horizon"]
    ).to(device)

    scheduler = DDPMScheduler(
        num_train_timesteps=variant_cfg["num_diffusion_steps"],
        beta_schedule="squaredcos_cap_v2",
        prediction_type="epsilon",
        clip_sample=True,
    )

    opt   = Adam(policy.parameters(), lr=variant_cfg["lr"],
                 weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=variant_cfg["epochs"], eta_min=1e-6)

    losses = []
    for epoch in trange(variant_cfg["epochs"],
                        desc=f"  Agent {agent_idx} Diffusion"):
        ep = []
        for obs_b, act_b in loader:
            obs_b, act_b = obs_b.to(device), act_b.to(device)
            B = act_b.shape[0]

            # Sample random diffusion timesteps
            t_rand     = torch.randint(
                0, variant_cfg["num_diffusion_steps"], (B,), device=device)
            noise      = torch.randn_like(act_b)
            noisy_acts = scheduler.add_noise(act_b, noise, t_rand)

            # Predict noise (same DDPM objective as your MPE work)
            noise_pred = policy(noisy_acts, t_rand, obs_b)
            loss       = nn.functional.mse_loss(noise_pred, noise)

            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step(); ep.append(loss.item())

        sched.step()
        losses.append(np.mean(ep))
        if (epoch + 1) % cfg["log_freq"] == 0:
            print(f"    Epoch {epoch+1:4d} | Diffusion Loss: {losses[-1]:.6f}")

    # Save model + scheduler config
    torch.save({
        "model": policy.state_dict(),
        "scheduler_config": scheduler.config,
        "num_diffusion_steps": variant_cfg["num_diffusion_steps"],
    }, os.path.join(save_dir, f"agent{agent_idx}_policy.pt"))

    return dataset.get_norm_stats(), losses


# ─────────────────────────────────────────────
# COMPARE ALL VARIANTS
# ─────────────────────────────────────────────

def plot_comparison(all_losses, variant_names):
    """Plot all variant training curves on one figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(variant_names)))

    for ax_idx, agent_idx in enumerate([0, 1]):
        ax = axes[ax_idx]
        for vi, name in enumerate(variant_names):
            if name in all_losses and agent_idx < len(all_losses[name]):
                losses = all_losses[name][agent_idx]
                ax.plot(losses, alpha=0.3, color=colors[vi])
                w = 10
                if len(losses) >= w:
                    sm = np.convolve(losses, np.ones(w)/w, mode="valid")
                    ax.plot(range(w-1, len(losses)), sm,
                            color=colors[vi], linewidth=2,
                            label=f"{name} (final={losses[-1]:.4f})")
        ax.set_title(f"Agent {agent_idx} BC Loss — All Variants")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.suptitle("BC Ablation Study — Training Loss Comparison", fontsize=13)
    plt.tight_layout()
    out = os.path.expanduser("~/marl_vla/results/bc_ablation_comparison.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.savefig(out, dpi=150)
    print(f"\nComparison plot saved to {out}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def run_variant(variant_name):
    cfg         = BASE_CFG.copy()
    variant_cfg = VARIANTS[variant_name]
    device      = torch.device(cfg["device"])

    save_dir = os.path.expanduser(
        f"~/marl_vla/checkpoints/bc_{variant_name}/")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(cfg["results_dir"], exist_ok=True)

    print(f"\n{'='*60}")
    print(f"VARIANT: {variant_name}")
    print(f"  {variant_cfg['label']}")
    print(f"{'='*60}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_stats  = []
    all_losses = []

    for agent_idx in range(cfg["n_agents"]):
        print(f"\nAgent {agent_idx}:")
        if variant_cfg["use_diffusion"]:
            stats, losses = train_diffusion(
                agent_idx, cfg, variant_cfg, device, save_dir)
        else:
            stats, losses = train_mlp(
                agent_idx, cfg, variant_cfg, device, save_dir)
        all_stats.append(stats)
        all_losses.append(losses)

    # Save norm stats and config
    np.save(os.path.join(save_dir, "norm_stats.npy"),
            {"agent0": all_stats[0], "agent1": all_stats[1]},
            allow_pickle=True)
    np.save(os.path.join(save_dir, "cfg.npy"),
            {**cfg, **variant_cfg, "variant": variant_name},
            allow_pickle=True)

    # Save loss CSV
    csv_path = os.path.join(
        cfg["results_dir"], f"bc_ablation_{variant_name}.csv")
    with open(csv_path, "w") as f:
        f.write("epoch,agent0_loss,agent1_loss\n")
        for i, (l0, l1) in enumerate(
                zip(all_losses[0], all_losses[1])):
            f.write(f"{i+1},{l0:.6f},{l1:.6f}\n")
    print(f"\nSaved loss CSV to {csv_path}")

    print(f"\nFinal losses — Agent 0: {all_losses[0][-1]:.6f} | "
          f"Agent 1: {all_losses[1][-1]:.6f}")
    print(f"Checkpoint saved to {save_dir}")
    return all_losses


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant",
        choices=list(VARIANTS.keys()) + ["all"],
        default="all",
        help="Which variant to run"
    )
    args = parser.parse_args()

    if args.variant == "all":
        run_order = ["baseline", "horizon16", "hidden512",
                     "augmented", "diffusion"]
        all_losses = {}
        for name in run_order:
            losses = run_variant(name)
            all_losses[name] = losses

        print("\n" + "="*60)
        print("ALL VARIANTS COMPLETE — SUMMARY")
        print("="*60)
        print(f"{'Variant':<15} {'Agent0 Final':>14} {'Agent1 Final':>14}")
        print("-"*45)
        for name in run_order:
            if name in all_losses:
                l0 = all_losses[name][0][-1]
                l1 = all_losses[name][1][-1]
                print(f"{name:<15} {l0:>14.6f} {l1:>14.6f}")

        plot_comparison(all_losses, run_order)
        print("\nNext: run validate_ab.py and validate_c.py for each variant")
        print("      then compare online rollout performance")

    else:
        run_variant(args.variant)


if __name__ == "__main__":
    main()