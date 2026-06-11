import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_dir = "/home/axs0940/marl_vla/results/can/samdp_comparison"
denoiser_dir = "/home/axs0940/marl_vla/results/can/diffusion_joint_denoiser"


def load_pivot(path):
    """Melt pivot table (rows=alpha_s, cols=alpha_a) into long format."""
    df = pd.read_csv(path, index_col=0)
    df.index.name = "alpha_s"
    df.columns.name = "alpha_a"
    df.index   = df.index.astype(float)
    df.columns = df.columns.astype(float)
    return df.reset_index().melt(id_vars="alpha_s", var_name="alpha_a", value_name="success_rate")


samdp    = load_pivot(f"{results_dir}/samdp_k03_sweep_joint.csv")
samdp_k1 = load_pivot(f"{results_dir}/samdp_k1_sweep_joint.csv")

eval2 = pd.read_csv(f"{denoiser_dir}/unet_eval2.csv")
eval2["alpha_s"] = eval2["alpha_s"].round(3)
eval2["alpha_a"] = eval2["alpha_a"].round(3)

vanilla  = eval2[["alpha_s", "alpha_a", "BASELINE (diffusion only)"]].rename(
    columns={"BASELINE (diffusion only)": "success_rate"}
)
denoiser = eval2[["alpha_s", "alpha_a", "lambda01_A7"]].rename(
    columns={"lambda01_A7": "success_rate"}
)

# Round to avoid float-key mismatch
samdp["alpha_s"]    = samdp["alpha_s"].round(3)
samdp["alpha_a"]    = samdp["alpha_a"].round(3)
samdp_k1["alpha_s"] = samdp_k1["alpha_s"].round(3)
samdp_k1["alpha_a"] = samdp_k1["alpha_a"].round(3)

# Merge on common (alpha_s, alpha_a) pairs
merged = (
    vanilla.rename(columns={"success_rate": "vanilla"})
    .merge(samdp.rename(columns={"success_rate": "samdp_k03"}), on=["alpha_s", "alpha_a"])
    .merge(samdp_k1.rename(columns={"success_rate": "samdp_k1"}), on=["alpha_s", "alpha_a"])
    .merge(denoiser.rename(columns={"success_rate": "joint_denoiser"}), on=["alpha_s", "alpha_a"])
)
merged = merged.sort_values(["alpha_s", "alpha_a"]).reset_index(drop=True)

x_labels = [f"s={r.alpha_s:.3f},a={r.alpha_a:.3f}" for _, r in merged.iterrows()]
x = np.arange(len(x_labels))

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(x, merged["vanilla"].values,        marker="o", label="Diffusion Policy",                          color="tab:blue")
ax.plot(x, merged["samdp_k03"].values,      marker="s", label="SA-MDP Regularised Diffusion Policy (k=3)",  color="tab:orange")
ax.plot(x, merged["samdp_k1"].values,       marker="D", label="SA-MDP Regularised Diffusion Policy (k=1)",  color="tab:red")
ax.plot(x, merged["joint_denoiser"].values, marker="^", label="Joint Denoiser",                              color="tab:green")

ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
ax.set_xlabel("alpha_s, alpha_a")
ax.set_ylabel("Success Rate")
ax.set_title("Can Task — Robustness Comparison")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.4)
ax.legend()

plt.tight_layout()
out_path = f"{results_dir}/samdp_comparison_plot_can.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
