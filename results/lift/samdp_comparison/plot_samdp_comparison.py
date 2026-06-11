import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_dir = "/home/axs0940/marl_vla/results/lift/samdp_comparison"
denoiser_dir = "/home/axs0940/marl_vla/results/lift/diffusion_joint_denoiser"

samdp_joint = pd.read_csv(f"{results_dir}/samdp_k03_sweep_joint.csv")
quick_eval = pd.read_csv(f"{denoiser_dir}/quick_eval.csv")

samdp_joint = samdp_joint.sort_values(["alpha_s", "alpha_a"]).reset_index(drop=True)
quick_eval = quick_eval.sort_values(["alpha_s", "alpha_a"]).reset_index(drop=True)

x_labels = [f"s={r.alpha_s:.3f},a={r.alpha_a:.3f}" for _, r in samdp_joint.iterrows()]
x = np.arange(len(x_labels))

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(x, quick_eval["BASELINE (diffusion only)"].values, marker="o", label="Diffusion Policy", color="tab:blue")
ax.plot(x, samdp_joint["samdp"].values, marker="s", label="SA-MDP Regularised Diffusion Policy", color="tab:orange")
ax.plot(x, quick_eval["baseline_A0"].values, marker="^", label="Joint Denoiser", color="tab:green")

ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
ax.set_xlabel("alpha_s, alpha_a")
ax.set_ylabel("value")
ax.set_title("Metrics vs alpha combos")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.4)
ax.legend()

plt.tight_layout()
out_path = f"{results_dir}/samdp_comparison_plot.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
