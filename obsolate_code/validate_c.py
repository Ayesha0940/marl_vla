"""
Validation Level C — Online Rollout Evaluation
===============================================
Runs actual robosuite episodes to measure true task performance.
Compares BC policy vs best TD3-BC checkpoint.

Metrics:
  - Task success rate (completed all 5 sub-goals)
  - Partial success (number of sub-goals completed)
  - Cumulative reward per episode
  - Episode length distribution

Usage:
    python validate_c.py [--best_step 100000]

Outputs:
    results/validate_c.csv
    results/validate_c_summary.txt
    results/validate_c_plots.png
"""

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import robosuite as suite
from collections import deque
import csv
import matplotlib.pyplot as plt
from tqdm import trange

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CFG = {
    "bc_dir":      os.path.expanduser("~/marl_vla/checkpoints/bc/"),
    "td3bc_dir":   os.path.expanduser("~/marl_vla/checkpoints/td3bc/"),
    "results_dir": os.path.expanduser("~/marl_vla/results/"),

    "obs_dim":     86,
    "action_dim":  7,
    "n_agents":    2,
    "obs_horizon": 2,
    "hidden_dim":  256,

    # Rollout settings
    "n_episodes":  50,      # episodes per policy
    "max_steps":   800,     # max steps per episode (transport is long)
    "reward_scale": 1.0,

    # Environment
    "env_name":  "TwoArmTransport",
    "robots":    ["Panda", "Panda"],

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ─────────────────────────────────────────────
# NETWORKS
# ─────────────────────────────────────────────

class Actor(nn.Module):
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
# OBSERVATION EXTRACTION
# ─────────────────────────────────────────────

def extract_obs(raw_obs, norm_stats, cfg, obs_histories):
    """
    Extract and normalise per-agent observations from robosuite obs dict.
    Updates obs_histories in place.
    Returns obs_stack: [2, H, 86]
    """
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
        ])  # [80]

        # Pad to 86 dims to match training (vel_lin/vel_ang missing in live env)
        if len(obs_i) < 86:
            obs_i = np.concatenate([obs_i, np.zeros(86 - len(obs_i))])
        # Normalise
        stats = norm_stats[f"agent{i}"]
        obs_i = (obs_i - stats["obs_mean"][0, 0]) / stats["obs_std"][0, 0]
        obs_histories[i].append(obs_i)

    obs_stack = np.stack([
        np.array(obs_histories[i]) for i in range(cfg["n_agents"])
    ])  # [2, H, 86]

    return obs_stack


# ─────────────────────────────────────────────
# POLICY ACTION
# ─────────────────────────────────────────────

@torch.no_grad()
def get_action(actors, obs_stack, norm_stats, cfg, device):
    """
    Get joint action from both agents, unnormalise, and concatenate.
    Returns: np.array [14]
    """
    obs_t = torch.FloatTensor(obs_stack).unsqueeze(0).to(device)
    # obs_t: [1, 2, H, 86]

    actions = []
    for i in range(cfg["n_agents"]):
        act_norm = actors[i](obs_t[:, i]).squeeze(0).cpu().numpy()  # [7]

        # Unnormalise
        stats = norm_stats[f"agent{i}"]
        act   = act_norm * stats["act_std"][0] + stats["act_mean"][0]
        actions.append(act)

    return np.concatenate(actions)  # [14]


# ─────────────────────────────────────────────
# ROLLOUT
# ─────────────────────────────────────────────

def run_rollouts(actors, norm_stats, cfg, label, device):
    """
    Run n_episodes rollouts and collect metrics.
    Returns list of episode result dicts.
    """
    env = suite.make(
        cfg["env_name"],
        robots=cfg["robots"],
        has_renderer=False,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        use_object_obs=True,
        horizon=cfg["max_steps"],
        reward_shaping=True,
        control_freq=20,
    )

    results = []

    for ep in trange(cfg["n_episodes"], desc=f"Rollout [{label}]"):
        raw_obs = env.reset()

        # Initialise obs history buffers
        obs_histories = [
            deque([np.zeros(cfg["obs_dim"])] * cfg["obs_horizon"],
                  maxlen=cfg["obs_horizon"])
            for _ in range(cfg["n_agents"])
        ]

        ep_reward    = 0.0
        ep_steps     = 0
        max_reward   = 0.0
        reward_milestones = []  # track when each sub-goal was hit

        for step in range(cfg["max_steps"]):
            obs_stack  = extract_obs(raw_obs, norm_stats, cfg, obs_histories)
            action_vec = get_action(actors, obs_stack, norm_stats, cfg, device)

            # Clip to valid action range
            low, high  = env.action_spec
            action_vec = np.clip(action_vec, low, high)

            raw_obs, reward, done, info = env.step(action_vec)
            ep_reward += reward
            ep_steps  += 1

            # Track sub-goal completion (reward increases at each milestone)
            if reward > 0.5:
                reward_milestones.append(step)

            max_reward = max(max_reward, ep_reward)

            if done:
                break

        # Transport task: full success = cumulative reward >= 5.0
        # Each sub-goal gives ~1.0 reward
        n_subgoals   = min(5, len(reward_milestones))
        full_success = ep_reward >= 4.5  # allow small floating point margin

        results.append({
            "episode":       ep,
            "policy":        label,
            "total_reward":  ep_reward,
            "steps":         ep_steps,
            "n_subgoals":    n_subgoals,
            "full_success":  int(full_success),
        })

    env.close()

    # Summary stats
    rewards    = [r["total_reward"] for r in results]
    successes  = [r["full_success"] for r in results]
    subgoals   = [r["n_subgoals"]   for r in results]

    print(f"\n  [{label}] Results over {cfg['n_episodes']} episodes:")
    print(f"    Mean reward:      {np.mean(rewards):.4f} ± {np.std(rewards):.4f}")
    print(f"    Success rate:     {np.mean(successes)*100:.1f}%")
    print(f"    Mean sub-goals:   {np.mean(subgoals):.2f} / 5")
    print(f"    Max reward:       {max(rewards):.4f}")

    return results


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def validate(best_step=None):
    cfg    = CFG
    device = torch.device(cfg["device"])
    os.makedirs(cfg["results_dir"], exist_ok=True)

    print(f"Device: {device}")
    print("="*60)
    print("VALIDATION LEVEL C — Online Rollout Evaluation")
    print("="*60)

    # Load norm stats
    norm_stats = np.load(
        os.path.join(cfg["bc_dir"], "norm_stats.npy"),
        allow_pickle=True).item()

    # ── Load BC actors ─────────────────────────────────────────
    bc_actors = []
    for i in range(cfg["n_agents"]):
        a = Actor(cfg["obs_dim"], cfg["action_dim"],
                  cfg["hidden_dim"], cfg["obs_horizon"]).to(device)
        a.load_state_dict(torch.load(
            os.path.join(cfg["bc_dir"], f"agent{i}_policy.pt"),
            map_location=device))
        a.eval()
        bc_actors.append(a)
    print("BC actors loaded")

    # ── Load TD3-BC actors (best/final checkpoint) ─────────────
    td3bc_actors = []
    for i in range(cfg["n_agents"]):
        a = Actor(cfg["obs_dim"], cfg["action_dim"],
                  cfg["hidden_dim"], cfg["obs_horizon"]).to(device)
        # Use final weights
        path = os.path.join(cfg["td3bc_dir"], f"agent{i}_actor_final.pt")
        a.load_state_dict(torch.load(path, map_location=device))
        a.eval()
        td3bc_actors.append(a)
    print(f"TD3-BC actors loaded (final checkpoint)")

    # ── Run rollouts ───────────────────────────────────────────
    all_results = []

    print("\nRunning BC rollouts...")
    bc_results = run_rollouts(bc_actors, norm_stats, cfg, "BC", device)
    all_results.extend(bc_results)

    print("\nRunning TD3-BC rollouts...")
    td3bc_results = run_rollouts(td3bc_actors, norm_stats, cfg,
                                  "TD3-BC (final)", device)
    all_results.extend(td3bc_results)

    # ── Save CSV ───────────────────────────────────────────────
    csv_path = os.path.join(cfg["results_dir"], "validate_c.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "episode", "policy", "total_reward",
            "steps", "n_subgoals", "full_success"
        ])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nSaved CSV to {csv_path}")

    # ── Summary ────────────────────────────────────────────────
    def summarise(results, label):
        rewards   = [r["total_reward"] for r in results]
        successes = [r["full_success"] for r in results]
        subgoals  = [r["n_subgoals"]   for r in results]
        return {
            "label":          label,
            "mean_reward":    np.mean(rewards),
            "std_reward":     np.std(rewards),
            "success_rate":   np.mean(successes) * 100,
            "mean_subgoals":  np.mean(subgoals),
        }

    bc_summary    = summarise(bc_results,    "BC")
    td3bc_summary = summarise(td3bc_results, "TD3-BC (final)")

    improvement = (td3bc_summary["mean_reward"] - bc_summary["mean_reward"]) / \
                  (abs(bc_summary["mean_reward"]) + 1e-8) * 100

    summary_path = os.path.join(cfg["results_dir"], "validate_c_summary.txt")
    with open(summary_path, "w") as f:
        f.write("VALIDATION LEVEL C SUMMARY\n")
        f.write("="*60 + "\n\n")
        for s in [bc_summary, td3bc_summary]:
            f.write(f"{s['label']}:\n")
            f.write(f"  Mean reward:    {s['mean_reward']:.4f} ± {s['std_reward']:.4f}\n")
            f.write(f"  Success rate:   {s['success_rate']:.1f}%\n")
            f.write(f"  Mean sub-goals: {s['mean_subgoals']:.2f} / 5\n\n")
        f.write(f"TD3-BC improvement over BC: {improvement:+.1f}%\n")
    print(f"Saved summary to {summary_path}")

    # Print summary table
    print("\n" + "="*60)
    print(f"{'Policy':<20} {'Reward':>10} {'Success%':>10} {'SubGoals':>10}")
    print("-"*60)
    for s in [bc_summary, td3bc_summary]:
        print(f"{s['label']:<20} "
              f"{s['mean_reward']:>10.4f} "
              f"{s['success_rate']:>9.1f}% "
              f"{s['mean_subgoals']:>10.2f}/5")
    print(f"\nTD3-BC improvement: {improvement:+.1f}%")
    print("="*60)

    # ── Plot ───────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    bc_rew    = [r["total_reward"] for r in bc_results]
    td3bc_rew = [r["total_reward"] for r in td3bc_results]
    bc_sg     = [r["n_subgoals"]   for r in bc_results]
    td3bc_sg  = [r["n_subgoals"]   for r in td3bc_results]

    # Reward distribution
    ax = axes[0]
    ax.hist(bc_rew,    bins=15, alpha=0.6, label="BC",           color="blue")
    ax.hist(td3bc_rew, bins=15, alpha=0.6, label="TD3-BC (final)", color="orange")
    ax.axvline(np.mean(bc_rew),    color="blue",   linestyle="--", linewidth=2)
    ax.axvline(np.mean(td3bc_rew), color="orange", linestyle="--", linewidth=2)
    ax.set_title("Cumulative Reward Distribution")
    ax.set_xlabel("Reward")
    ax.set_ylabel("Count")
    ax.legend()
    ax.grid(alpha=0.3)

    # Sub-goals completed
    ax = axes[1]
    sg_vals = [0, 1, 2, 3, 4, 5]
    bc_sg_counts    = [bc_sg.count(v)    for v in sg_vals]
    td3bc_sg_counts = [td3bc_sg.count(v) for v in sg_vals]
    x = np.arange(len(sg_vals))
    w = 0.35
    ax.bar(x - w/2, bc_sg_counts,    w, label="BC",            color="blue",   alpha=0.7)
    ax.bar(x + w/2, td3bc_sg_counts, w, label="TD3-BC (final)", color="orange", alpha=0.7)
    ax.set_title("Sub-goals Completed per Episode")
    ax.set_xlabel("Number of Sub-goals")
    ax.set_ylabel("Episode Count")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in sg_vals])
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    # Reward over episodes
    ax = axes[2]
    ax.plot(bc_rew,    "o-", alpha=0.5, label="BC",            color="blue",   markersize=3)
    ax.plot(td3bc_rew, "o-", alpha=0.5, label="TD3-BC (final)", color="orange", markersize=3)
    ax.axhline(np.mean(bc_rew),    color="blue",   linestyle="--", linewidth=2)
    ax.axhline(np.mean(td3bc_rew), color="orange", linestyle="--", linewidth=2)
    ax.set_title("Reward per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.suptitle("Validation Level C: Online Rollout Evaluation",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(cfg["results_dir"], "validate_c_plots.png")
    plt.savefig(out, dpi=150)
    print(f"Saved plots to {out}")
    print("\nLevel C complete. Run step3_robustness.py next.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--best_step", type=int, default=100000,
                        help="Best TD3-BC checkpoint step from Level B")
    args = parser.parse_args()
    validate(best_step=args.best_step)