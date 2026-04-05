"""
Step 2: MARL Fine-tuning with TD3-BC on Transport Task
=======================================================
Fine-tunes the BC policy from Step 1 using TD3-BC
(Twin Delayed DDPG + Behaviour Cloning regularisation).

TD3-BC objective per agent:
    actor_loss = -Q(s, a) + alpha * MSE(a, a_bc)
                  ^RL term    ^BC regularisation term

This is decentralised execution (each agent has its own
actor) with centralised critics (each critic sees all obs
and all actions) — matching your MADDPG CTDE structure.

Usage:
    python step2_td3bc.py

Outputs:
    checkpoints/td3bc/agent{i}_actor.pt
    checkpoints/td3bc/agent{i}_critic1.pt
    checkpoints/td3bc/agent{i}_critic2.pt
    td3bc_training_curves.png
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
import h5py
from tqdm import trange
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CFG = {
    "dataset_path": os.path.expanduser(
        "~/marl_vla/datasets/transport/ph/low_dim_v141.hdf5"
    ),
    "bc_dir":   os.path.expanduser("~/marl_vla/checkpoints/bc/"),
    "save_dir": os.path.expanduser("~/marl_vla/checkpoints/td3bc/"),

    # Dims
    "obs_dim":    86,
    "action_dim": 7,
    "n_agents":   2,
    "obs_horizon": 2,

    # Architecture
    "hidden_dim": 256,

    # TD3-BC hyperparams
    "alpha":         2.5,    # BC regularisation weight
    "gamma":         0.99,   # discount
    "tau":           0.005,  # soft target update
    "policy_noise":  0.2,    # target policy smoothing noise
    "noise_clip":    0.5,    # clamp target noise
    "policy_freq":   2,      # delayed actor update (every N critic steps)

    # Training
    "num_updates":   100_000, # gradient steps
    "batch_size":    256,
    "lr_actor":      3e-4,
    "lr_critic":     3e-4,
    "device":        "cuda" if torch.cuda.is_available() else "cpu",
    "log_freq":      5_000,
    "save_freq":     25_000,
}

# ─────────────────────────────────────────────
# LOAD BC POLICY (reuse from step1)
# ─────────────────────────────────────────────

class BCPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, obs_horizon):
        super().__init__()
        inp = obs_dim * obs_horizon
        self.net = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Mish(),
            nn.Linear(hidden_dim // 2, action_dim), nn.Tanh(),
        )
    def forward(self, obs):
        return self.net(obs.reshape(obs.shape[0], -1))


# ─────────────────────────────────────────────
# TD3-BC NETWORKS
# ─────────────────────────────────────────────

class Actor(nn.Module):
    """
    Decentralised actor — sees only own agent observation.
    Initialised from BC policy weights.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, obs_horizon):
        super().__init__()
        inp = obs_dim * obs_horizon
        self.net = nn.Sequential(
            nn.Linear(inp, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Mish(),
            nn.Linear(hidden_dim // 2, action_dim), nn.Tanh(),
        )

    def forward(self, obs):
        """obs: [B, H, obs_dim]"""
        return self.net(obs.reshape(obs.shape[0], -1))


class CentralisedCritic(nn.Module):
    """
    Centralised critic — sees ALL agents' observations and actions.
    CTDE: Centralised Training, Decentralised Execution.
    Mirrors your MADDPG critic structure.

    Input: concat of [obs_agent0, obs_agent1, action_agent0, action_agent1]
    """
    def __init__(self, obs_dim, action_dim, hidden_dim, obs_horizon, n_agents=2):
        super().__init__()
        inp = n_agents * (obs_dim * obs_horizon + action_dim)
        self.net = nn.Sequential(
            nn.Linear(inp, hidden_dim * 2), nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Mish(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, all_obs, all_actions):
        """
        all_obs:     [B, n_agents, H, obs_dim]
        all_actions: [B, n_agents, action_dim]
        """
        B = all_obs.shape[0]
        x = torch.cat([
            all_obs.reshape(B, -1),
            all_actions.reshape(B, -1),
        ], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────

class ReplayBuffer:
    """
    Stores (obs, action, reward, next_obs, done) tuples.
    Pre-loaded from HDF5 demos for offline TD3-BC.
    """

    def __init__(self, obs_dim, action_dim, n_agents, obs_horizon, device):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.n_agents   = n_agents
        self.obs_horizon = obs_horizon
        self.device     = device
        self.obs      = []
        self.actions  = []
        self.rewards  = []
        self.next_obs = []
        self.dones    = []

    def load_from_hdf5(self, hdf5_path, norm_stats):
        """Load all demo transitions into buffer."""
        print("Loading replay buffer from demos...")
        with h5py.File(hdf5_path, "r") as f:
            for key in sorted(f["data"].keys()):
                demo    = f["data"][key]
                actions = demo["actions"][:]   # [T, 14]
                rewards = demo["rewards"][:]   # [T]
                dones   = demo["dones"][:]     # [T]
                T       = actions.shape[0]
                H       = self.obs_horizon

                # Build per-agent obs sequences
                agent_obs = []
                for i in range(self.n_agents):
                    prefix = f"robot{i}"
                    obs_d  = demo["obs"]
                    obs_i  = np.concatenate([
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

                    # Normalise
                    stats  = norm_stats[f"agent{i}"]
                    obs_i  = (obs_i - stats["obs_mean"][0]) / stats["obs_std"][0]
                    agent_obs.append(obs_i)

                # Normalise actions
                for t in range(H - 1, T - 1):
                    obs_windows = []
                    nxt_windows = []
                    for i in range(self.n_agents):
                        obs_windows.append(agent_obs[i][t - H + 1 : t + 1])      # [H, 86]
                        nxt_windows.append(agent_obs[i][t - H + 2 : t + 2])      # [H, 86]

                    obs_stack = np.stack(obs_windows)   # [2, H, 86]
                    nxt_stack = np.stack(nxt_windows)   # [2, H, 86]

                    # Normalise actions per agent
                    act_norm = np.zeros((self.n_agents, self.action_dim),
                                        dtype=np.float32)
                    for i in range(self.n_agents):
                        stats = norm_stats[f"agent{i}"]
                        raw   = actions[t, i * 7 : (i + 1) * 7]
                        act_norm[i] = (raw - stats["act_mean"][0]) / stats["act_std"][0]

                    self.obs.append(obs_stack)
                    self.actions.append(act_norm)
                    self.rewards.append(rewards[t])
                    self.next_obs.append(nxt_stack)
                    self.dones.append(dones[t])

        self.obs      = torch.FloatTensor(np.array(self.obs)).to(self.device)      # [N,2,H,86]
        self.actions  = torch.FloatTensor(np.array(self.actions)).to(self.device)  # [N,2,7]
        self.rewards  = torch.FloatTensor(np.array(self.rewards)).unsqueeze(1).to(self.device)  # [N,1]
        self.next_obs = torch.FloatTensor(np.array(self.next_obs)).to(self.device) # [N,2,H,86]
        self.dones    = torch.FloatTensor(np.array(self.dones)).unsqueeze(1).to(self.device)    # [N,1]
        self.size     = len(self.rewards)
        print(f"Replay buffer loaded: {self.size:,} transitions")

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,))
        return (
            self.obs[idx],
            self.actions[idx],
            self.rewards[idx],
            self.next_obs[idx],
            self.dones[idx],
        )


# ─────────────────────────────────────────────
# SOFT UPDATE
# ─────────────────────────────────────────────

def soft_update(net, target, tau):
    for p, tp in zip(net.parameters(), target.parameters()):
        tp.data.copy_(tau * p.data + (1 - tau) * tp.data)


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train():
    cfg    = CFG
    device = torch.device(cfg["device"])
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(cfg["save_dir"], exist_ok=True)

    # Load norm stats from Step 1
    norm_stats = np.load(
        os.path.join(cfg["bc_dir"], "norm_stats.npy"),
        allow_pickle=True
    ).item()

    # Replay buffer
    replay = ReplayBuffer(
        cfg["obs_dim"], cfg["action_dim"],
        cfg["n_agents"], cfg["obs_horizon"], device
    )
    replay.load_from_hdf5(cfg["dataset_path"], norm_stats)

    # ── Build actors (initialise from BC weights) ─────────────
    actors  = []
    actors_target = []
    for i in range(cfg["n_agents"]):
        actor = Actor(
            cfg["obs_dim"], cfg["action_dim"],
            cfg["hidden_dim"], cfg["obs_horizon"]
        ).to(device)

        # Load BC weights as initialisation
        bc_weights = torch.load(
            os.path.join(cfg["bc_dir"], f"agent{i}_policy.pt"),
            map_location=device
        )
        actor.load_state_dict(bc_weights)
        print(f"Actor {i} initialised from BC weights")

        actors.append(actor)
        actors_target.append(copy.deepcopy(actor))

    # ── Build centralised critics (two per agent for TD3) ─────
    critics1, critics2 = [], []
    critics1_target, critics2_target = [], []

    for i in range(cfg["n_agents"]):
        c1 = CentralisedCritic(
            cfg["obs_dim"], cfg["action_dim"],
            cfg["hidden_dim"], cfg["obs_horizon"]
        ).to(device)
        c2 = CentralisedCritic(
            cfg["obs_dim"], cfg["action_dim"],
            cfg["hidden_dim"], cfg["obs_horizon"]
        ).to(device)
        critics1.append(c1)
        critics2.append(c2)
        critics1_target.append(copy.deepcopy(c1))
        critics2_target.append(copy.deepcopy(c2))

    # ── Optimisers ────────────────────────────────────────────
    actor_opts  = [Adam(a.parameters(), lr=cfg["lr_actor"])  for a in actors]
    critic_opts = [
        Adam(list(c1.parameters()) + list(c2.parameters()), lr=cfg["lr_critic"])
        for c1, c2 in zip(critics1, critics2)
    ]

    # ── Logging ───────────────────────────────────────────────
    critic_losses = [[] for _ in range(cfg["n_agents"])]
    actor_losses  = [[] for _ in range(cfg["n_agents"])]
    log_steps     = []

    print(f"\nStarting TD3-BC fine-tuning for {cfg['num_updates']:,} steps...")

    for step in trange(1, cfg["num_updates"] + 1, desc="TD3-BC"):

        obs_b, act_b, rew_b, nxt_b, done_b = replay.sample(cfg["batch_size"])
        # obs_b:  [B, 2, H, 86]
        # act_b:  [B, 2, 7]
        # rew_b:  [B, 1]
        # nxt_b:  [B, 2, H, 86]
        # done_b: [B, 1]

        # ── Critic update ──────────────────────────────────────
        with torch.no_grad():
            # Target actions with smoothing noise
            nxt_acts = []
            for i in range(cfg["n_agents"]):
                noise = (
                    torch.randn_like(act_b[:, i]) * cfg["policy_noise"]
                ).clamp(-cfg["noise_clip"], cfg["noise_clip"])
                nxt_a = (actors_target[i](nxt_b[:, i]) + noise).clamp(-1, 1)
                nxt_acts.append(nxt_a)
            nxt_acts_t = torch.stack(nxt_acts, dim=1)  # [B, 2, 7]

            # Target Q values (shared reward — cooperative task)
            q1_tgt = critics1_target[0](nxt_b, nxt_acts_t)
            q2_tgt = critics2_target[0](nxt_b, nxt_acts_t)
            q_tgt  = rew_b + cfg["gamma"] * (1 - done_b) * torch.min(q1_tgt, q2_tgt)

        for i in range(cfg["n_agents"]):
            q1 = critics1[i](obs_b, act_b)
            q2 = critics2[i](obs_b, act_b)
            c_loss = nn.functional.mse_loss(q1, q_tgt) + \
                     nn.functional.mse_loss(q2, q_tgt)

            critic_opts[i].zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(
                list(critics1[i].parameters()) +
                list(critics2[i].parameters()), 1.0
            )
            critic_opts[i].step()

        # ── Delayed actor update ───────────────────────────────
        if step % cfg["policy_freq"] == 0:
            for i in range(cfg["n_agents"]):
                # Current actor actions for all agents
                curr_acts = []
                for j in range(cfg["n_agents"]):
                    if j == i:
                        curr_acts.append(actors[j](obs_b[:, j]))
                    else:
                        with torch.no_grad():
                            curr_acts.append(actors[j](obs_b[:, j]))
                curr_acts_t = torch.stack(curr_acts, dim=1)  # [B, 2, 7]

                # TD3-BC actor loss
                q_val  = critics1[i](obs_b, curr_acts_t)

                # Normalise Q for stable BC weighting (from TD3-BC paper)
                lam    = cfg["alpha"] / (q_val.abs().mean().detach() + 1e-8)

                # BC term: stay close to demo actions
                bc_loss = nn.functional.mse_loss(
                    curr_acts_t[:, i], act_b[:, i]
                )

                a_loss = -lam * q_val.mean() + bc_loss

                actor_opts[i].zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(actors[i].parameters(), 1.0)
                actor_opts[i].step()

                actor_losses[i].append(a_loss.item())

            # Soft update targets
            for i in range(cfg["n_agents"]):
                soft_update(actors[i],   actors_target[i],   cfg["tau"])
                soft_update(critics1[i], critics1_target[i], cfg["tau"])
                soft_update(critics2[i], critics2_target[i], cfg["tau"])

        # Append critic loss for logging
        for i in range(cfg["n_agents"]):
            q1 = critics1[i](obs_b, act_b)
            q2 = critics2[i](obs_b, act_b)
            critic_losses[i].append(
                (nn.functional.mse_loss(q1, q_tgt) +
                 nn.functional.mse_loss(q2, q_tgt)).item()
            )

        # ── Logging ───────────────────────────────────────────
        if step % cfg["log_freq"] == 0:
            log_steps.append(step)
            for i in range(cfg["n_agents"]):
                recent_c = np.mean(critic_losses[i][-cfg["log_freq"]:])
                recent_a = np.mean(actor_losses[i][-max(1, cfg["log_freq"] // cfg["policy_freq"]):])
                print(f"  Step {step:7d} | Agent {i} | "
                      f"Critic Loss: {recent_c:.5f} | "
                      f"Actor Loss: {recent_a:.5f}")

        # ── Save checkpoints ──────────────────────────────────
        if step % cfg["save_freq"] == 0:
            for i in range(cfg["n_agents"]):
                torch.save(actors[i].state_dict(),
                    os.path.join(cfg["save_dir"], f"agent{i}_actor.pt"))
                torch.save(critics1[i].state_dict(),
                    os.path.join(cfg["save_dir"], f"agent{i}_critic1.pt"))
                torch.save(critics2[i].state_dict(),
                    os.path.join(cfg["save_dir"], f"agent{i}_critic2.pt"))
            print(f"  → Checkpoint saved at step {step}")

    # Final save
    for i in range(cfg["n_agents"]):
        torch.save(actors[i].state_dict(),
            os.path.join(cfg["save_dir"], f"agent{i}_actor_final.pt"))
    print(f"\nSaved final weights to {cfg['save_dir']}")

    # ── Plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    titles = [["Agent 0 Critic Loss", "Agent 0 Actor Loss"],
              ["Agent 1 Critic Loss", "Agent 1 Actor Loss"]]

    for i in range(cfg["n_agents"]):
        # Critic loss
        ax = axes[i][0]
        ax.plot(critic_losses[i], alpha=0.3)
        w = 500
        if len(critic_losses[i]) >= w:
            sm = np.convolve(critic_losses[i], np.ones(w)/w, mode="valid")
            ax.plot(range(w-1, len(critic_losses[i])), sm, linewidth=2)
        ax.set_title(titles[i][0])
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)

        # Actor loss
        ax = axes[i][1]
        ax.plot(actor_losses[i], alpha=0.3)
        w2 = max(1, w // cfg["policy_freq"])
        if len(actor_losses[i]) >= w2:
            sm = np.convolve(actor_losses[i], np.ones(w2)/w2, mode="valid")
            ax.plot(range(w2-1, len(actor_losses[i])), sm, linewidth=2)
        ax.set_title(titles[i][1])
        ax.set_xlabel("Step")
        ax.grid(alpha=0.3)

    plt.suptitle("Step 2: TD3-BC Fine-tuning Curves", fontsize=13)
    plt.tight_layout()
    out = os.path.expanduser("~/marl_vla/td3bc_training_curves.png")
    plt.savefig(out, dpi=150)
    print(f"Saved training curves to {out}")
    print("\nStep 2 complete. Run step3_robustness.py next.")


if __name__ == "__main__":
    train()