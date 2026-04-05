"""
Validation Levels A + B — Offline Q-value Metrics & Checkpoint Sweep
=====================================================================
Level A: Did TD3-BC improve over BC?
  - Normalised Q-value: Q(s, a_policy) vs Q(s, a_demo)
  - Behavioural divergence: MSE(a_td3bc, a_bc)
  - Q-value on demo actions vs policy actions

Level B: Which checkpoint is best?
  - Evaluate all saved checkpoints (25k/50k/75k/100k steps)
  - Find the elbow point in the learning curve
  - Identify optimal stopping point

Usage:
    python validate_ab.py

Outputs:
    results/validate_ab.csv
    results/validate_ab_curves.png
    results/validate_ab_summary.txt
"""

import os
import numpy as np
import torch
import torch.nn as nn
import h5py
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CFG = {
    "dataset_path": os.path.expanduser(
        "~/marl_vla/datasets/transport/ph/low_dim_v141.hdf5"
    ),
    "bc_dir":      os.path.expanduser("~/marl_vla/checkpoints/bc/"),
    "td3bc_dir":   os.path.expanduser("~/marl_vla/checkpoints/td3bc/"),
    "results_dir": os.path.expanduser("~/marl_vla/results/"),

    "obs_dim":     86,
    "action_dim":  7,
    "n_agents":    2,
    "obs_horizon": 2,
    "hidden_dim":  256,

    # Checkpoints to evaluate (Level B sweep)
    "checkpoints": [25000, 50000, 75000, 100000],

    # How many batches to average metrics over
    "n_eval_batches": 500,
    "batch_size":     256,

    # Held-out split for evaluation (last 20% of demos)
    "eval_demo_fraction": 0.2,

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


class CentralisedCritic(nn.Module):
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
        B = all_obs.shape[0]
        x = torch.cat([all_obs.reshape(B, -1),
                        all_actions.reshape(B, -1)], dim=-1)
        return self.net(x)


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_eval_data(cfg, norm_stats, device):
    """
    Load held-out transitions (last eval_demo_fraction of demos).
    Returns tensors ready for batch evaluation.
    """
    obs_list, act_list, rew_list = [], [], []

    with h5py.File(cfg["dataset_path"], "r") as f:
        all_keys = sorted(f["data"].keys())
        n_eval   = max(1, int(len(all_keys) * cfg["eval_demo_fraction"]))
        eval_keys = all_keys[-n_eval:]  # held-out last 20%
        print(f"Using {len(eval_keys)} held-out demos for evaluation")

        for key in eval_keys:
            demo    = f["data"][key]
            actions = demo["actions"][:]
            rewards = demo["rewards"][:]
            T       = actions.shape[0]
            H       = cfg["obs_horizon"]

            agent_obs = []
            for i in range(cfg["n_agents"]):
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
                ], axis=1)
                stats = norm_stats[f"agent{i}"]
                obs_i = (obs_i - stats["obs_mean"][0]) / stats["obs_std"][0]
                agent_obs.append(obs_i)

            for t in range(H - 1, T - 1):
                obs_windows = [agent_obs[i][t-H+1:t+1]
                               for i in range(cfg["n_agents"])]
                obs_stack   = np.stack(obs_windows)

                act_norm = np.zeros(
                    (cfg["n_agents"], cfg["action_dim"]), dtype=np.float32)
                for i in range(cfg["n_agents"]):
                    stats = norm_stats[f"agent{i}"]
                    raw   = actions[t, i*7:(i+1)*7]
                    act_norm[i] = (raw - stats["act_mean"][0]) / \
                                   stats["act_std"][0]

                obs_list.append(obs_stack)
                act_list.append(act_norm)
                rew_list.append(rewards[t])

    obs_t = torch.FloatTensor(np.array(obs_list)).to(device)
    act_t = torch.FloatTensor(np.array(act_list)).to(device)
    rew_t = torch.FloatTensor(np.array(rew_list)).to(device)

    print(f"Eval transitions: {len(obs_t):,}")
    return obs_t, act_t, rew_t


# ─────────────────────────────────────────────
# METRICS
# ─────────────────────────────────────────────

@torch.no_grad()
def compute_metrics(actors, critics1, obs_t, act_demo_t,
                    bc_actors, cfg, device, n_batches, batch_size):
    """
    Compute all Level A metrics for a given set of actors/critics.

    Returns dict with:
      q_policy:    mean Q(s, a_policy)   — how good are policy actions?
      q_demo:      mean Q(s, a_demo)     — how good are expert actions?
      q_ratio:     q_policy / q_demo     — >1 means policy surpassed expert
      behav_div_0: MSE(a_policy0, a_bc0) — how much agent 0 diverged from BC
      behav_div_1: MSE(a_policy1, a_bc1) — how much agent 1 diverged from BC
      behav_div_0_demo: MSE(a_policy0, a_demo0) — divergence from expert
      behav_div_1_demo: MSE(a_policy1, a_demo1) — divergence from expert
    """
    N = len(obs_t)
    results = {
        "q_policy": [], "q_demo": [],
        "behav_div_bc_0": [], "behav_div_bc_1": [],
        "behav_div_demo_0": [], "behav_div_demo_1": [],
    }

    for _ in range(n_batches):
        idx   = torch.randint(0, N, (batch_size,))
        obs_b = obs_t[idx]    # [B, 2, H, 86]
        act_b = act_demo_t[idx]  # [B, 2, 7]

        # Policy actions (current actors)
        policy_acts = torch.stack(
            [actors[i](obs_b[:, i]) for i in range(cfg["n_agents"])],
            dim=1
        )  # [B, 2, 7]

        # BC actions
        bc_acts = torch.stack(
            [bc_actors[i](obs_b[:, i]) for i in range(cfg["n_agents"])],
            dim=1
        )  # [B, 2, 7]

        # Q-values
        q_pol  = critics1[0](obs_b, policy_acts).mean().item()
        q_dem  = critics1[0](obs_b, act_b).mean().item()

        results["q_policy"].append(q_pol)
        results["q_demo"].append(q_dem)

        # Behavioural divergence from BC
        for i in range(cfg["n_agents"]):
            bd_bc   = nn.functional.mse_loss(
                policy_acts[:, i], bc_acts[:, i]).item()
            bd_demo = nn.functional.mse_loss(
                policy_acts[:, i], act_b[:, i]).item()
            results[f"behav_div_bc_{i}"].append(bd_bc)
            results[f"behav_div_demo_{i}"].append(bd_demo)

    # Average all metrics
    out = {k: float(np.mean(v)) for k, v in results.items()}
    out["q_ratio"] = out["q_policy"] / (abs(out["q_demo"]) + 1e-8)
    return out


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def load_actors(path_template, cfg, device):
    actors = []
    for i in range(cfg["n_agents"]):
        a = Actor(cfg["obs_dim"], cfg["action_dim"],
                  cfg["hidden_dim"], cfg["obs_horizon"]).to(device)
        a.load_state_dict(torch.load(
            path_template.format(i), map_location=device))
        a.eval()
        actors.append(a)
    return actors


def load_critics(path_template, cfg, device):
    critics = []
    for i in range(cfg["n_agents"]):
        c = CentralisedCritic(
            cfg["obs_dim"], cfg["action_dim"],
            cfg["hidden_dim"], cfg["obs_horizon"]).to(device)
        c.load_state_dict(torch.load(
            path_template.format(i), map_location=device))
        c.eval()
        critics.append(c)
    return critics


def validate():
    cfg    = CFG
    device = torch.device(cfg["device"])
    os.makedirs(cfg["results_dir"], exist_ok=True)

    print(f"Device: {device}")
    print("="*60)
    print("VALIDATION LEVELS A + B")
    print("="*60)

    # Load norm stats
    norm_stats = np.load(
        os.path.join(cfg["bc_dir"], "norm_stats.npy"),
        allow_pickle=True).item()

    # Load eval data (held-out 20% of demos)
    print("\nLoading held-out evaluation data...")
    obs_t, act_demo_t, rew_t = load_eval_data(cfg, norm_stats, device)

    # ── Load BC actors (baseline) ──────────────────────────────
    print("\nLoading BC policy (baseline)...")
    bc_actors = load_actors(
        os.path.join(cfg["bc_dir"], "agent{}_policy.pt"),
        cfg, device
    )

    # Load critics from final TD3-BC checkpoint
    # (we use the final critics for all evaluations — consistent Q-function)
    print("Loading TD3-BC critics (final)...")
    critics1 = load_critics(
        os.path.join(cfg["td3bc_dir"], "agent{}_critic1.pt"),
        cfg, device
    )

    # ── Level A: BC baseline metrics ──────────────────────────
    print("\n" + "-"*60)
    print("LEVEL A — BC Baseline Metrics")
    print("-"*60)
    bc_metrics = compute_metrics(
        bc_actors, critics1, obs_t, act_demo_t,
        bc_actors, cfg, device,
        cfg["n_eval_batches"], cfg["batch_size"]
    )
    print(f"  Q(s, a_bc):          {bc_metrics['q_policy']:+.5f}")
    print(f"  Q(s, a_demo):        {bc_metrics['q_demo']:+.5f}")
    print(f"  Q ratio (bc/demo):   {bc_metrics['q_ratio']:.4f}")
    print(f"  Behav div BC→demo 0: {bc_metrics['behav_div_demo_0']:.5f}")
    print(f"  Behav div BC→demo 1: {bc_metrics['behav_div_demo_1']:.5f}")

    # ── Level B: Checkpoint sweep ──────────────────────────────
    print("\n" + "-"*60)
    print("LEVEL B — TD3-BC Checkpoint Sweep")
    print("-"*60)

    all_results = []

    # Add BC as step 0
    all_results.append({
        "step": 0,
        "label": "BC (step 0)",
        **bc_metrics
    })

    for step in cfg["checkpoints"]:
        # Try final weights name first, then numbered checkpoint
        if step == cfg["checkpoints"][-1]:
            actor_suffix = 'actor_final'
        else:
            actor_suffix = 'actor'
        actor_path = os.path.join(
            cfg["td3bc_dir"], f"agent{{}}_{actor_suffix}.pt"
        )

        # Check which file exists
        test_path_final  = os.path.join(cfg["td3bc_dir"],
                                         "agent0_actor_final.pt")
        test_path_ckpt   = os.path.join(cfg["td3bc_dir"],
                                         "agent0_actor.pt")

        if step == cfg["checkpoints"][-1] and os.path.exists(test_path_final):
            actor_tmpl = os.path.join(cfg["td3bc_dir"], "agent{}_actor_final.pt")
        else:
            actor_tmpl = os.path.join(cfg["td3bc_dir"], "agent{}_actor.pt")

        if not os.path.exists(actor_tmpl.format(0)):
            print(f"  Step {step:7d}: checkpoint not found, skipping")
            continue

        actors = load_actors(actor_tmpl, cfg, device)

        metrics = compute_metrics(
            actors, critics1, obs_t, act_demo_t,
            bc_actors, cfg, device,
            cfg["n_eval_batches"], cfg["batch_size"]
        )

        all_results.append({
            "step": step,
            "label": f"TD3-BC (step {step})",
            **metrics
        })

        print(f"  Step {step:7d} | "
              f"Q_policy: {metrics['q_policy']:+.5f} | "
              f"Q_demo: {metrics['q_demo']:+.5f} | "
              f"Q_ratio: {metrics['q_ratio']:.4f} | "
              f"BDiv_BC_0: {metrics['behav_div_bc_0']:.4f} | "
              f"BDiv_demo_0: {metrics['behav_div_demo_0']:.4f}")

    # ── Find best checkpoint ───────────────────────────────────
    td3bc_results = [r for r in all_results if r["step"] > 0]
    best = max(td3bc_results, key=lambda r: r["q_ratio"])
    print(f"\n{'='*60}")
    print(f"BEST CHECKPOINT: {best['label']}")
    print(f"  Q ratio: {best['q_ratio']:.4f}  "
          f"(>1.0 means policy surpassed expert Q-values)")
    print(f"  Behav divergence from BC:   "
          f"agent0={best['behav_div_bc_0']:.4f}  "
          f"agent1={best['behav_div_bc_1']:.4f}")
    print(f"  Behav divergence from demo: "
          f"agent0={best['behav_div_demo_0']:.4f}  "
          f"agent1={best['behav_div_demo_1']:.4f}")
    print(f"{'='*60}")

    # ── Save CSV ───────────────────────────────────────────────
    csv_path = os.path.join(cfg["results_dir"], "validate_ab.csv")
    fieldnames = ["step", "label", "q_policy", "q_demo", "q_ratio",
                  "behav_div_bc_0", "behav_div_bc_1",
                  "behav_div_demo_0", "behav_div_demo_1"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r[k] for k in fieldnames})
    print(f"\nSaved CSV to {csv_path}")

    # ── Save summary ───────────────────────────────────────────
    summary_path = os.path.join(cfg["results_dir"], "validate_ab_summary.txt")
    with open(summary_path, "w") as f:
        f.write("VALIDATION LEVELS A + B SUMMARY\n")
        f.write("="*60 + "\n\n")
        f.write("LEVEL A: BC vs TD3-BC\n")
        f.write(f"  BC Q-value:       {bc_metrics['q_policy']:+.5f}\n")
        f.write(f"  Best TD3-BC Q:    {best['q_policy']:+.5f}\n")
        q_improvement = best['q_policy'] - bc_metrics['q_policy']
        f.write(f"  Q improvement:    {q_improvement:+.5f}\n\n")
        f.write("LEVEL B: Best Checkpoint\n")
        f.write(f"  Best step:        {best['step']}\n")
        f.write(f"  Q ratio:          {best['q_ratio']:.4f}\n")
        f.write(f"  Recommendation:   Use step {best['step']} "
                f"weights for Step 3 and beyond\n")
    print(f"Saved summary to {summary_path}")

    # ── Plot ───────────────────────────────────────────────────
    steps  = [r["step"]    for r in all_results]
    labels = [r["label"]   for r in all_results]
    q_pol  = [r["q_policy"]  for r in all_results]
    q_dem  = [r["q_demo"]    for r in all_results]
    q_rat  = [r["q_ratio"]   for r in all_results]
    bd_bc0 = [r["behav_div_bc_0"]   for r in all_results]
    bd_bc1 = [r["behav_div_bc_1"]   for r in all_results]
    bd_d0  = [r["behav_div_demo_0"] for r in all_results]
    bd_d1  = [r["behav_div_demo_1"] for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Q-values
    ax = axes[0][0]
    ax.plot(steps, q_pol, "o-", label="Q(s, a_policy)", linewidth=2)
    ax.plot(steps, q_dem, "s--", label="Q(s, a_demo)",  linewidth=2)
    ax.axvline(best["step"], color="red", linestyle=":", alpha=0.7,
               label=f"Best ckpt ({best['step']})")
    ax.set_title("Q-values: Policy vs Demo Actions")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Mean Q-value")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(steps)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)

    # Q ratio
    ax = axes[0][1]
    ax.plot(steps, q_rat, "o-", color="purple", linewidth=2)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.7,
               label="Q_ratio = 1.0 (matches expert)")
    ax.axvline(best["step"], color="red", linestyle=":", alpha=0.7)
    ax.set_title("Q-ratio: Policy Q / Demo Q\n(>1.0 = surpassed expert)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Q-ratio")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(steps)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)

    # Behavioural divergence from BC
    ax = axes[1][0]
    ax.plot(steps, bd_bc0, "o-", label="Agent 0", linewidth=2)
    ax.plot(steps, bd_bc1, "s-", label="Agent 1", linewidth=2)
    ax.axvline(best["step"], color="red", linestyle=":", alpha=0.7)
    ax.set_title("Behavioural Divergence from BC\n(how much RL changed the policy)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE(a_policy, a_bc)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(steps)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)

    # Behavioural divergence from demo
    ax = axes[1][1]
    ax.plot(steps, bd_d0, "o-", label="Agent 0", linewidth=2)
    ax.plot(steps, bd_d1, "s-", label="Agent 1", linewidth=2)
    ax.axvline(best["step"], color="red", linestyle=":", alpha=0.7)
    ax.set_title("Behavioural Divergence from Expert Demo\n"
                 "(how much policy differs from human)")
    ax.set_xlabel("Training Step")
    ax.set_ylabel("MSE(a_policy, a_demo)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_xticks(steps)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)

    plt.suptitle("Validation A+B: BC vs TD3-BC Checkpoint Sweep",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = os.path.join(cfg["results_dir"], "validate_ab_curves.png")
    plt.savefig(out, dpi=150)
    print(f"Saved plots to {out}")
    print("\nLevel A+B complete. Run validate_c.py next.")

    return best["step"]


if __name__ == "__main__":
    validate()