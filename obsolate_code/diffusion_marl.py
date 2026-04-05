"""
Diffusion Policy for Multi-Agent Robotic Manipulation
======================================================
A minimal implementation of DDPM-based action denoising for two-agent
manipulation using RoboSuite's TwoArmLift environment.

Architecture:
  - Centralized critic (sees all observations)
  - Decentralized actors (each agent denoises its own actions)
  - DDPM noise scheduler (same as your MPE work)

Author: generated for MARL+VLA research
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from diffusers import DDPMScheduler
from collections import deque
import matplotlib.pyplot as plt
import robosuite as suite
from tqdm import trange

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CFG = {
    # Environment
    "env_name":         "TwoArmLift",
    "robots":           ["Panda", "Panda"],
    "camera_obs":       False,           # no vision yet — Stage 1 is proprioception only
    "horizon":          200,

    # Diffusion
    "num_diffusion_steps": 100,          # DDPM timesteps (same concept as your MPE work)
    "prediction_type":  "epsilon",       # predict noise, not x0

    # Architecture
    "obs_dim":          38,              # robot0_proprio(32) + object-state(14) = see below
    "action_dim":       7,               # 7 DoF arm + 1 gripper per agent
    "hidden_dim":       256,
    "obs_horizon":      2,               # stack last N obs for temporal context

    # Training
    "num_episodes":     200,
    "max_steps":        200,
    "batch_size":       64,
    "lr":               1e-4,
    "replay_capacity":  10_000,
    "min_replay_size":  500,             # wait before training
    "device":           "cuda" if torch.cuda.is_available() else "cpu",
    "log_freq":         10,
    "save_path":        "checkpoints/",
}

# ─────────────────────────────────────────────
# REPLAY BUFFER
# ─────────────────────────────────────────────

class ReplayBuffer:
    """Simple replay buffer storing (obs, action) pairs per agent."""

    def __init__(self, capacity, obs_dim, action_dim, obs_horizon, device):
        self.capacity   = capacity
        self.device     = device
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.obs_horizon= obs_horizon
        self.ptr        = 0
        self.size       = 0

        # Store for both agents
        self.obs     = np.zeros((capacity, 2, obs_horizon, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, 2, action_dim),           dtype=np.float32)

    def add(self, obs, actions):
        """
        obs:     np.array [2, obs_horizon, obs_dim]
        actions: np.array [2, action_dim]
        """
        self.obs[self.ptr]     = obs
        self.actions[self.ptr] = actions
        self.ptr  = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.obs[idxs]).to(self.device),     # [B, 2, H, obs_dim]
            torch.FloatTensor(self.actions[idxs]).to(self.device),  # [B, 2, action_dim]
        )

    def __len__(self):
        return self.size


# ─────────────────────────────────────────────
# DIFFUSION POLICY NETWORK (per agent)
# ─────────────────────────────────────────────

class SinusoidalPosEmb(nn.Module):
    """Timestep embedding — same as standard DDPM."""

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
    Denoising network for one agent.

    Input:  noisy action + timestep embedding + observation context
    Output: predicted noise (epsilon)

    This is the same U-Net-style MLP denoiser from your DDPM work,
    just conditioned on robot observations instead of agent states.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, obs_horizon):
        super().__init__()

        obs_flat_dim = obs_dim * obs_horizon  # flatten temporal obs

        # Timestep embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        # Observation encoder
        self.obs_enc = nn.Sequential(
            nn.Linear(obs_flat_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )

        # Denoising MLP (action + time + obs → noise)
        self.net = nn.Sequential(
            nn.Linear(action_dim + hidden_dim + hidden_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, noisy_action, timestep, obs):
        """
        noisy_action: [B, action_dim]
        timestep:     [B]
        obs:          [B, obs_horizon, obs_dim]
        """
        B = noisy_action.shape[0]

        t_emb   = self.time_emb(timestep)                          # [B, hidden]
        obs_enc = self.obs_enc(obs.reshape(B, -1))                 # [B, hidden]
        x       = torch.cat([noisy_action, t_emb, obs_enc], dim=-1)
        return self.net(x)                                         # [B, action_dim]


# ─────────────────────────────────────────────
# CENTRALIZED CRITIC (optional, for future MARL training)
# ─────────────────────────────────────────────

class CentralizedCritic(nn.Module):
    """
    Sees ALL agents' observations and actions.
    Not used in BC training below, but scaffolded for your MARL extension.
    You can plug this into MADDPG / MAPPO style training later.
    """

    def __init__(self, obs_dim, action_dim, hidden_dim, obs_horizon, n_agents=2):
        super().__init__()
        joint_dim = n_agents * (obs_dim * obs_horizon + action_dim)

        self.net = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim * 2),
            nn.Mish(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, all_obs, all_actions):
        """
        all_obs:     [B, n_agents, obs_horizon, obs_dim]
        all_actions: [B, n_agents, action_dim]
        """
        B = all_obs.shape[0]
        x = torch.cat([
            all_obs.reshape(B, -1),
            all_actions.reshape(B, -1)
        ], dim=-1)
        return self.net(x)  # [B, 1]


# ─────────────────────────────────────────────
# OBSERVATION HELPER
# ─────────────────────────────────────────────

def extract_obs(raw_obs, cfg):
    """
    Extract per-agent observations from raw robosuite obs dict.

    Agent 0: robot0_proprio-state + object-state
    Agent 1: robot1_proprio-state + object-state

    Returns np.array of shape [2, obs_dim]
    """
    obj    = raw_obs["object-state"]                  # shared object state
    prop0  = raw_obs["robot0_proprio-state"]
    prop1  = raw_obs["robot1_proprio-state"]

    obs0   = np.concatenate([prop0, obj])
    obs1   = np.concatenate([prop1, obj])

    # Pad or trim to cfg obs_dim
    def pad(x, d):
        if len(x) >= d: return x[:d]
        return np.concatenate([x, np.zeros(d - len(x))])

    return np.stack([pad(obs0, cfg["obs_dim"]),
                     pad(obs1, cfg["obs_dim"])])      # [2, obs_dim]


# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────

def train():
    cfg    = CFG
    device = torch.device(cfg["device"])
    print(f"Training on: {device} ({torch.cuda.get_device_name(0) if device.type=='cuda' else 'CPU'})")

    os.makedirs(cfg["save_path"], exist_ok=True)

    # --- Environment ---
    env = suite.make(
        cfg["env_name"],
        robots=cfg["robots"],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=cfg["camera_obs"],
        horizon=cfg["horizon"],
        reward_shaping=True,
    )

    # --- Models (one denoising net per agent, shared DDPM scheduler) ---
    agents = nn.ModuleList([
        DiffusionPolicyNet(
            cfg["obs_dim"], cfg["action_dim"],
            cfg["hidden_dim"], cfg["obs_horizon"]
        ).to(device)
        for _ in range(2)
    ])

    critic = CentralizedCritic(
        cfg["obs_dim"], cfg["action_dim"],
        cfg["hidden_dim"], cfg["obs_horizon"]
    ).to(device)

    # One optimizer per agent (decentralized training)
    optimizers = [Adam(a.parameters(), lr=cfg["lr"]) for a in agents]

    # DDPM noise scheduler — same API you know from diffusers
    scheduler = DDPMScheduler(
        num_train_timesteps=cfg["num_diffusion_steps"],
        beta_schedule="squaredcos_cap_v2",   # cosine schedule, better than linear
        prediction_type=cfg["prediction_type"],
        clip_sample=True,
    )

    # --- Replay Buffer ---
    replay = ReplayBuffer(
        cfg["replay_capacity"],
        cfg["obs_dim"],
        cfg["action_dim"],
        cfg["obs_horizon"],
        device,
    )

    # --- Logging ---
    ep_rewards  = []
    agent_losses = [[], []]

    # ── Main Loop ──────────────────────────────
    for episode in trange(cfg["num_episodes"], desc="Episodes"):
        raw_obs = env.reset()

        # Observation history buffer (one per agent)
        obs_history = [
            deque([np.zeros(cfg["obs_dim"])] * cfg["obs_horizon"],
                  maxlen=cfg["obs_horizon"])
            for _ in range(2)
        ]

        ep_reward  = 0.0
        ep_loss    = [0.0, 0.0]
        ep_updates = 0

        for step in range(cfg["max_steps"]):

            # 1. Extract observations
            cur_obs = extract_obs(raw_obs, cfg)        # [2, obs_dim]
            for i in range(2):
                obs_history[i].append(cur_obs[i])

            # obs_stack: [2, obs_horizon, obs_dim]
            obs_stack = np.stack([
                np.array(obs_history[i]) for i in range(2)
            ])

            # 2. Sample actions via DDPM reverse process (inference)
            actions = []
            for i in range(2):
                obs_t   = torch.FloatTensor(obs_stack[i]).unsqueeze(0).to(device)  # [1, H, obs_dim]
                act_t   = torch.randn(1, cfg["action_dim"]).to(device)             # start from noise

                # Reverse diffusion (denoising loop)
                scheduler.set_timesteps(cfg["num_diffusion_steps"])
                for t in scheduler.timesteps:
                    with torch.no_grad():
                        noise_pred = agents[i](act_t, t.unsqueeze(0).to(device), obs_t)
                    act_t = scheduler.step(noise_pred, t, act_t).prev_sample

                actions.append(act_t.squeeze(0).cpu().numpy())

            # Clip to valid action range
            low, high = env.action_spec
            action_vec = np.concatenate(actions)
            action_vec = np.clip(action_vec, low, high)

            # 3. Step environment
            raw_obs, reward, done, _ = env.step(action_vec)
            ep_reward += reward

            # 4. Store transition
            # Action for replay: split back per agent
            act0 = action_vec[:cfg["action_dim"]]
            act1 = action_vec[cfg["action_dim"]:]
            replay.add(obs_stack, np.stack([act0, act1]))

            # 5. Train if enough data (Behavior Cloning on collected experience)
            if len(replay) >= cfg["min_replay_size"]:
                obs_b, act_b = replay.sample(cfg["batch_size"])
                # obs_b: [B, 2, H, obs_dim]
                # act_b: [B, 2, action_dim]

                for i in range(2):
                    obs_i = obs_b[:, i]    # [B, H, obs_dim]
                    act_i = act_b[:, i]    # [B, action_dim]

                    # Sample random diffusion timesteps
                    t_rand = torch.randint(
                        0, cfg["num_diffusion_steps"],
                        (cfg["batch_size"],), device=device
                    )

                    # Add noise to actions (forward diffusion)
                    noise       = torch.randn_like(act_i)
                    noisy_acts  = scheduler.add_noise(act_i, noise, t_rand)

                    # Predict noise
                    noise_pred  = agents[i](noisy_acts, t_rand, obs_i)

                    # MSE loss on noise prediction (same as DDPM training objective)
                    loss = nn.functional.mse_loss(noise_pred, noise)

                    optimizers[i].zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(agents[i].parameters(), 1.0)
                    optimizers[i].step()

                    ep_loss[i] += loss.item()
                ep_updates += 1

            if done:
                break

        ep_rewards.append(ep_reward)
        for i in range(2):
            agent_losses[i].append(ep_loss[i] / max(ep_updates, 1))

        # Logging
        if (episode + 1) % cfg["log_freq"] == 0:
            avg_r  = np.mean(ep_rewards[-cfg["log_freq"]:])
            avg_l0 = np.mean(agent_losses[0][-cfg["log_freq"]:])
            avg_l1 = np.mean(agent_losses[1][-cfg["log_freq"]:])
            print(f"\nEp {episode+1:4d} | Reward: {avg_r:7.3f} | "
                  f"Loss agent0: {avg_l0:.4f} | Loss agent1: {avg_l1:.4f} | "
                  f"Replay: {len(replay)}")

        # Save checkpoints
        if (episode + 1) % 50 == 0:
            for i in range(2):
                torch.save(agents[i].state_dict(),
                           f"{cfg['save_path']}agent{i}_ep{episode+1}.pt")
            print(f"  → Saved checkpoints at episode {episode+1}")

    env.close()

    # ── Plot results ───────────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(ep_rewards, alpha=0.4, label="per episode")
    window = 10
    if len(ep_rewards) >= window:
        smoothed = np.convolve(ep_rewards, np.ones(window)/window, mode='valid')
        ax1.plot(range(window-1, len(ep_rewards)), smoothed, label=f"{window}-ep avg")
    ax1.set_title("Episode Reward")
    ax1.set_xlabel("Episode")
    ax1.legend()

    ax2.plot(agent_losses[0], alpha=0.6, label="Agent 0")
    ax2.plot(agent_losses[1], alpha=0.6, label="Agent 1")
    ax2.set_title("Diffusion Loss (MSE on noise)")
    ax2.set_xlabel("Episode")
    ax2.legend()

    plt.tight_layout()
    plt.savefig("training_curves.png", dpi=150)
    print("\nTraining curves saved to training_curves.png")
    print("Done!")


if __name__ == "__main__":
    train()