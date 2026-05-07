import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CUSTOM_PATH = os.path.join(
    DATA_DIR,
    "..",
    "custom_diffusion_policy",
    "noise_modes_joint.csv",
)
BASELINE_PATH = os.path.join(DATA_DIR, "ablation_results.csv")
OUT_PATH = os.path.join(DATA_DIR, "noise_modes_joint_vs_baseline.png")


def load_custom_matrix(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.rename(columns={raw.columns[0]: "alpha_s"})
    long_df = raw.melt(id_vars="alpha_s", var_name="alpha_a", value_name="success_rate")
    long_df["alpha_s"] = long_df["alpha_s"].astype(float)
    long_df["alpha_a"] = long_df["alpha_a"].astype(float)
    long_df["success_rate"] = long_df["success_rate"].astype(float)
    return long_df


def load_baseline_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    value_col = next(col for col in df.columns if col not in {"alpha_s", "alpha_a"})
    df = df.rename(columns={value_col: "success_rate"})
    df["alpha_s"] = df["alpha_s"].astype(float)
    df["alpha_a"] = df["alpha_a"].astype(float)
    df["success_rate"] = df["success_rate"].astype(float)
    return df[["alpha_s", "alpha_a", "success_rate"]]


custom_df = load_custom_matrix(CUSTOM_PATH)
baseline_df = load_baseline_table(BASELINE_PATH)

merged = (
    custom_df.merge(
        baseline_df,
        on=["alpha_s", "alpha_a"],
        how="left",
        suffixes=("_custom", "_baseline"),
    )
    .sort_values(["alpha_s", "alpha_a"])
    .reset_index(drop=True)
)

missing_baseline = merged[merged["success_rate_baseline"].isna()][["alpha_s", "alpha_a"]]
if not missing_baseline.empty:
    print("Baseline diffusion is missing these alpha configurations; the line will break there:")
    print(missing_baseline.to_string(index=False))

x_labels = [f"({row.alpha_s:.3f}, {row.alpha_a:.3f})" for _, row in merged.iterrows()]
x = range(len(merged))

plt.figure(figsize=(14, 6))
plt.plot(
    x,
    merged["success_rate_custom"].values,
    label="Without denoiser",
    color="#d62728",
    linewidth=2.2,
    marker="o",
    markersize=4,
)
plt.plot(
    x,
    merged["success_rate_baseline"].values,
    label="Baseline diffusion",
    color="#1f77b4",
    linewidth=2.2,
    marker="s",
    markersize=4,
)

plt.xticks(list(x), x_labels, rotation=45, ha="right", fontsize=8)
plt.xlabel("(alpha_s, alpha_a)")
plt.ylabel("Success rate")
plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
plt.ylim(0.0, 1.05)
plt.grid(axis="y", alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT_PATH}")
plt.show()