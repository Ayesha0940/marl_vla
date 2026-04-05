"""
Diffusion BC Rollout Evaluation
================================
Evaluates the two-agent Diffusion BC policy on TwoArmTransport.
Compares against MLP-BC baseline in the same script.

Usage:
    python validate_diffusion_bc.py

Outputs:
    results/validate_diffusion_bc.csv
    results/validate_diffusion_bc_summary.txt
"""

import os
import numpy as np
import torch
import torch.nn as nn
from collections import deque
import robosuite as suite
from diffusers import DDPMScheduler
from tqdm import trange
import csv

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CFG = {
    "bc_mlp_dir":       os.path.expanduser("~/marl_vla/checkpoints/bc/"),
    "bc_diff_dir":      os.path.expanduser("~/marl_vla/checkpoints/bc_diffusion/"),
    "results_dir":      os.path.expanduser("~/marl_vla/results/"),

    "obs_dim":          86,
    "action_dim":       7,
    "n_agents":         2,
    "obs_horizon":      2,    # MLP uses H=2
    "obs_horizon_diff": 16,   # Diffusion uses H=16
    "hidden_dim":       256,

    # Rollout
    "n_episodes":       50,
    "max_steps":        800,

    # Diffusion inference — 20 steps is fast and nearly as good as 100
    "inference_steps":  20,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ─────────────────────────────────────────────
# NETWORKS
# ─────────────────────────────────────────────

class MLPPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim, obs_horizon):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim * obs_horizon, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim), nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.Mish(),
            nn.Linear(hidden_dim // 2, action_dim), nn.Tanh(),
        )
    def forward(self, obs):
        return self.net(obs.reshape(obs.shape[0], -1))


class SinusoidalPosEmb(nn.Module):
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
        B     = noisy_action.shape[0]
        t_emb = self.time_emb(timestep)
        o_emb = self.obs_enc(obs.reshape(B, -1))
        x     = torch.cat([noisy_action, t_emb, o_emb], dim=-1)
        return self.net(x)

    @torch.no_grad()
    def sample(self, obs, scheduler, device, inference_steps=20):
        """Fast inference using subset of denoising steps."""
        B      = obs.shape[0]
        action = torch.randn(B, 7).to(device)
        scheduler.set_timesteps(inference_steps)
        for t in scheduler.timesteps:
            t_batch    = t.unsqueeze(0).expand(B).to(device)
            noise_pred = self(action, t_batch, obs)
            action     = scheduler.step(noise_pred, t, action).prev_sample
        return action.clamp(-1, 1)


# ─────────────────────────────────────────────
# OBSERVATION EXTRACTION
# ─────────────────────────────────────────────

def extract_obs(raw_obs, norm_stats, cfg, obs_histories, obs_horizon):
    for i in range(cfg["n_agents"]):
        prefix = f"robot{i}"
        obs_i  = np.concatenate([
            raw_obs[f"{prefix}_joint_pos"],
            raw_obs[f"{prefix}_joint_pos_cos"],
            raw_obs[f"{prefix}_joint_pos_sin"],
            raw_obs[f"{prefix}_joint_vel"],
            raw_obs[f"{prefix}_eef_pos"],
            raw_obs[f"{prefix}_eef_quat"],
            raw_obs[f"{prefix}_gripper_qpos"],
            raw_obs[f"{prefix}_gripper_qvel"],
            raw_obs["object-state"],
        ])
        if len(obs_i) < cfg["obs_dim"]:
            obs_i = np.concatenate(
                [obs_i, np.zeros(cfg["obs_dim"] - len(obs_i))])
        stats = norm_stats[f"agent{i}"]
        obs_i = (obs_i - stats["obs_mean"][0, 0]) / stats["obs_std"][0, 0]
        obs_histories[i].append(obs_i)

    return np.stack([np.array(obs_histories[i])
                     for i in range(cfg["n_agents"])])


# ─────────────────────────────────────────────
# ROLLOUT
# ─────────────────────────────────────────────

def run_rollouts(policy_type, actors, norm_stats, cfg,
                 scheduler=None, device=None):
    obs_horizon = (cfg["obs_horizon_diff"]
                   if policy_type == "diffusion"
                   else cfg["obs_horizon"])

    env = suite.make(
        "TwoArmTransport",
        robots=["Panda", "Panda"],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        horizon=cfg["max_steps"],
        reward_shaping=False,
        control_freq=20,
    )

    results = []
    for ep in trange(cfg["n_episodes"],
                     desc=f"Rollout [{policy_type}]"):
        raw_obs = env.reset()
        obs_histories = [
            deque([np.zeros(cfg["obs_dim"])] * obs_horizon,
                  maxlen=obs_horizon)
            for _ in range(cfg["n_agents"])
        ]

        ep_reward = 0.0
        ep_steps  = 0
        subgoals  = set()

        for step in range(cfg["max_steps"]):
            obs_stack = extract_obs(
                raw_obs, norm_stats, cfg,
                obs_histories, obs_horizon)

            actions = []
            for i in range(cfg["n_agents"]):
                obs_t = torch.FloatTensor(
                    obs_stack[i]).unsqueeze(0).to(device)

                if policy_type == "mlp":
                    with torch.no_grad():
                        act_norm = actors[i](obs_t).squeeze(0).cpu().numpy()
                else:
                    act_norm = actors[i].sample(
                        obs_t, scheduler, device,
                        cfg["inference_steps"]
                    ).squeeze(0).cpu().numpy()

                stats = norm_stats[f"agent{i}"]
                act   = act_norm * stats["act_std"][0] + stats["act_mean"][0]
                actions.append(act)

            action_vec = np.concatenate(actions)
            low, high  = env.action_spec
            action_vec = np.clip(action_vec, low, high)

            raw_obs, reward, done, info = env.step(action_vec)
            ep_reward += reward
            ep_steps  += 1

            if reward >= 1.0:
                subgoals.add(round(ep_reward))
            if done:
                break

        n_subgoals   = min(5, len(subgoals))
        full_success = ep_reward >= 4.5

        results.append({
            "episode":      ep,
            "policy":       policy_type,
            "total_reward": ep_reward,
            "steps":        ep_steps,
            "n_subgoals":   n_subgoals,
            "full_success": int(full_success),
        })

    env.close()

    rewards   = [r["total_reward"] for r in results]
    successes = [r["full_success"] for r in results]
    subgoals  = [r["n_subgoals"]   for r in results]

    print(f"\n  [{policy_type.upper()}] {cfg['n_episodes']} episodes:")
    print(f"    Mean reward:    {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"    Success rate:   {np.mean(successes)*100:.1f}%")
    print(f"    Mean sub-goals: {np.mean(subgoals):.2f} / 5")
    print(f"    Max reward:     {max(rewards):.1f}")
    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    cfg    = CFG
    device = torch.device(cfg["device"])
    os.makedirs(cfg["results_dir"], exist_ok=True)

    print(f"Device: {device}")
    print("="*60)
    print("Diffusion BC vs MLP BC — Online Rollout Evaluation")
    print("="*60)

    # Load MLP-BC
    print("\nLoading MLP-BC...")
    mlp_norm = np.load(
        os.path.join(cfg["bc_mlp_dir"], "norm_stats.npy"),
        allow_pickle=True).item()
    mlp_actors = []
    for i in range(cfg["n_agents"]):
        a = MLPPolicy(cfg["obs_dim"], cfg["action_dim"],
                      cfg["hidden_dim"], cfg["obs_horizon"]).to(device)
        a.load_state_dict(torch.load(
            os.path.join(cfg["bc_mlp_dir"], f"agent{i}_policy.pt"),
            map_location=device))
        a.eval()
        mlp_actors.append(a)
    print("  OK")

    # Load Diffusion-BC
    print("Loading Diffusion-BC...")
    diff_norm = np.load(
        os.path.join(cfg["bc_diff_dir"], "norm_stats.npy"),
        allow_pickle=True).item()
    diff_actors = []
    shared_scheduler = None
    for i in range(cfg["n_agents"]):
        ckpt = torch.load(
            os.path.join(cfg["bc_diff_dir"], f"agent{i}_policy.pt"),
            map_location=device, weights_only=False)
        a = DiffusionPolicyNet(
            cfg["obs_dim"], cfg["action_dim"],
            cfg["hidden_dim"], cfg["obs_horizon_diff"]).to(device)
        a.load_state_dict(ckpt["model"])
        a.eval()
        diff_actors.append(a)
        if shared_scheduler is None:
            shared_scheduler = DDPMScheduler.from_config(
                ckpt["scheduler_config"])
    print(f"  OK (inference_steps={cfg['inference_steps']})")

    # Run rollouts
    print("\nRunning MLP-BC rollouts...")
    mlp_results = run_rollouts(
        "mlp", mlp_actors, mlp_norm, cfg, device=device)

    print("\nRunning Diffusion-BC rollouts...")
    diff_results = run_rollouts(
        "diffusion", diff_actors, diff_norm, cfg,
        scheduler=shared_scheduler, device=device)

    # Save CSV
    csv_path = os.path.join(cfg["results_dir"], "validate_diffusion_bc.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "policy", "total_reward",
            "steps", "n_subgoals", "full_success"])
        writer.writeheader()
        writer.writerows(mlp_results + diff_results)
    print(f"\nSaved CSV to {csv_path}")

    # Summary
    def summarise(results, label):
        rewards   = [r["total_reward"] for r in results]
        successes = [r["full_success"] for r in results]
        subgoals  = [r["n_subgoals"]   for r in results]
        return {
            "label":         label,
            "mean_reward":   np.mean(rewards),
            "std_reward":    np.std(rewards),
            "success_rate":  np.mean(successes) * 100,
            "mean_subgoals": np.mean(subgoals),
            "max_reward":    max(rewards),
        }

    mlp_s  = summarise(mlp_results,  "MLP-BC")
    diff_s = summarise(diff_results, "Diffusion-BC")
    improvement = (diff_s["mean_reward"] - mlp_s["mean_reward"]) / \
                  (abs(mlp_s["mean_reward"]) + 1e-8) * 100

    print("\n" + "="*65)
    print(f"{'Policy':<16} {'Reward':>10} {'Success%':>10} "
          f"{'SubGoals':>10} {'MaxRew':>8}")
    print("-"*65)
    for s in [mlp_s, diff_s]:
        print(f"{s['label']:<16} "
              f"{s['mean_reward']:>10.4f} "
              f"{s['success_rate']:>9.1f}% "
              f"{s['mean_subgoals']:>10.2f}/5 "
              f"{s['max_reward']:>8.1f}")
    print("="*65)
    print(f"\nDiffusion-BC vs MLP-BC improvement: {improvement:+.1f}%")

    print("\n── Literature Comparison (Transport task) ────────────────")
    print(f"  BC-RNN image  (Mandlekar 2021):  72.0%  [image, single-agent]")
    print(f"  Diffusion Policy (Chi 2023):     84.0%  [state, single-agent]")
    print(f"  DPPO (Ren 2024):                >90.0%  [state, single-agent+RL]")
    print(f"  Your MLP-BC:                    {mlp_s['success_rate']:>5.1f}%"
          f"  [state, 2-agent decentralised]")
    print(f"  Your Diffusion-BC:              {diff_s['success_rate']:>5.1f}%"
          f"  [state, 2-agent decentralised]")

    summary_path = os.path.join(
        cfg["results_dir"], "validate_diffusion_bc_summary.txt")
    with open(summary_path, "w") as f:
        f.write("DIFFUSION BC vs MLP BC\n" + "="*60 + "\n\n")
        for s in [mlp_s, diff_s]:
            f.write(f"{s['label']}:\n")
            f.write(f"  Mean reward:    {s['mean_reward']:.4f} "
                    f"± {s['std_reward']:.4f}\n")
            f.write(f"  Success rate:   {s['success_rate']:.1f}%\n")
            f.write(f"  Mean sub-goals: {s['mean_subgoals']:.2f}/5\n\n")
        f.write(f"Improvement: {improvement:+.1f}%\n")
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    main()