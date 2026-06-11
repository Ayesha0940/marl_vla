import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results_dir   = "/home/axs0940/marl_vla/results/square/samdp_comparison"
denoiser_dir  = "/home/axs0940/marl_vla/results/square/joint_denoiser_comparison"


def load_pivot(path):
    df = pd.read_csv(path, index_col=0)
    df.index.name = "alpha_s"
    df.columns.name = "alpha_a"
    df.index = df.index.astype(float)
    df.columns = df.columns.astype(float)
    return df.reset_index().melt(id_vars="alpha_s", var_name="alpha_a", value_name="success_rate")


vanilla = load_pivot(f"{results_dir}/vanilla_sweep_joint.csv")
samdp   = load_pivot(f"{results_dir}/samdp_k03_sweep_joint.csv")

for df in (vanilla, samdp):
    df["alpha_s"] = df["alpha_s"].round(3)
    df["alpha_a"] = df["alpha_a"].round(3)

joint_sweep = pd.read_csv(f"{denoiser_dir}/joint_sweep.csv")
joint_sweep["alpha_s"] = joint_sweep["alpha_s"].round(3)
joint_sweep["alpha_a"] = joint_sweep["alpha_a"].round(3)
denoiser = joint_sweep[["alpha_s", "alpha_a", "joint_no_warmstart_a0"]]

merged = (
    vanilla.rename(columns={"success_rate": "vanilla"})
    .merge(samdp.rename(columns={"success_rate": "samdp_k03"}), on=["alpha_s", "alpha_a"])
    .merge(denoiser, on=["alpha_s", "alpha_a"])
)
merged = merged.sort_values(["alpha_s", "alpha_a"]).reset_index(drop=True)

x_labels = [f"s={r.alpha_s:.3f},a={r.alpha_a:.3f}" for _, r in merged.iterrows()]
x = np.arange(len(x_labels))

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(x, merged["vanilla"].values,               marker="o", label="Diffusion Policy",                        color="tab:blue")
ax.plot(x, merged["samdp_k03"].values,             marker="s", label="SA-MDP Regularised Diffusion Policy",      color="tab:orange")
ax.plot(x, merged["joint_no_warmstart_a0"].values, marker="^", label="Joint Denoiser",                            color="tab:green")

ax.set_xticks(x)
ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=7)
ax.set_xlabel("alpha_s, alpha_a")
ax.set_ylabel("value")
ax.set_title("Square Task — Robustness Comparison")
ax.set_ylim(-0.05, 1.05)
ax.grid(True, alpha=0.4)
ax.legend()

plt.tight_layout()
out_path = f"{results_dir}/samdp_comparison_plot_square.png"
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved: {out_path}")
