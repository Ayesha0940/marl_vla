import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

RESULTS_DIR = os.path.dirname(os.path.abspath(__file__))

# Load all anchor files — keep only the newest file per (anchor, tstart) pair
all_paths = glob.glob(os.path.join(RESULTS_DIR, "joint_A*_tstart*.csv"))

# Group paths by (anchor, tstart), pick the one with the latest timestamp in the filename
from collections import defaultdict
path_groups = defaultdict(list)
for path in all_paths:
    fname = os.path.basename(path)
    parts = fname.split("_")
    anchor_id = parts[1]
    tstart = parts[2]  # e.g. tstart10
    path_groups[(anchor_id, tstart)].append(path)

anchor_dfs = []
for (anchor_id, tstart_str), paths in path_groups.items():
    path = sorted(paths)[-1]  # latest by filename timestamp
    df = pd.read_csv(path)
    df["anchor"] = anchor_id
    df["tstart"] = int(tstart_str.replace("tstart", ""))
    anchor_dfs.append(df)

all_data = pd.concat(anchor_dfs, ignore_index=True)

# For each (anchor, alpha_s, alpha_a), keep best mean_reward across tstart values
anchor_rows = all_data[all_data["method"].str.startswith("JOINT")]
anchor_best = (
    anchor_rows.groupby(["anchor", "alpha_s", "alpha_a"], as_index=False)["mean_reward"].max()
)

# Clean baseline: BASE-noisy rows (same across files, just pick one tstart per condition)
clean_rows = all_data[all_data["method"] == "BASE-noisy"]
clean_best = (
    clean_rows.groupby(["alpha_s", "alpha_a"], as_index=False)["mean_reward"].max()
)

alpha_s_values = sorted(anchor_best["alpha_s"].unique())
anchors = sorted(anchor_best["anchor"].unique(), key=lambda x: int(x[1:]))  # A0..A8

colors = cm.tab10(np.linspace(0, 1, len(anchors) + 1))

fig, axes = plt.subplots(1, len(alpha_s_values), figsize=(6 * len(alpha_s_values), 5), sharey=True)
if len(alpha_s_values) == 1:
    axes = [axes]

for ax, alpha_s in zip(axes, alpha_s_values):
    # Clean line
    clean_sub = clean_best[clean_best["alpha_s"] == alpha_s].sort_values("alpha_a")
    ax.plot(clean_sub["alpha_a"], clean_sub["mean_reward"],
            color=colors[0], marker="o", linewidth=2, label="Clean (no denoiser)")

    # Anchor lines
    for i, anchor in enumerate(anchors):
        sub = anchor_best[(anchor_best["anchor"] == anchor) & (anchor_best["alpha_s"] == alpha_s)]
        sub = sub.sort_values("alpha_a")
        ax.plot(sub["alpha_a"], sub["mean_reward"],
                color=colors[i + 1], marker="o", linewidth=1.5, label=anchor)

    ax.set_title(f"s_alpha = {alpha_s:.3f}")
    ax.set_xlabel("Action noise std (alpha_a)")
    ax.set_ylabel("Mean reward")
    ax.set_xticks(sorted(anchor_best["alpha_a"].unique()))
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

axes[-1].legend(loc="upper right", fontsize=8)
fig.suptitle("Joint Denoiser Anchors vs Clean Baseline", fontsize=13, fontweight="bold")
plt.tight_layout()

out_path = os.path.join(RESULTS_DIR, "anchors_vs_clean.png")
plt.savefig(out_path, dpi=150)
print(f"Saved: {out_path}")
plt.show()
