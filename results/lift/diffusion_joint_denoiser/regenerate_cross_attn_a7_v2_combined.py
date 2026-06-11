import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

CSV_MAIN = os.path.join(DATA_DIR, "cross_attn_a7_v2.csv")
CSV_QUICK = os.path.join(DATA_DIR, "quick_eval.csv")
OUT_PNG = os.path.join(DATA_DIR, "cross_attn_a7_v2_combined_plot.png")


def main():
    a = pd.read_csv(CSV_MAIN)
    b = pd.read_csv(CSV_QUICK)

    # merge to ensure consistent ordering and presence of baseline_A0
    merged = a.merge(
        b[["alpha_s", "alpha_a", "baseline_A0"]],
        on=["alpha_s", "alpha_a"],
        how="left",
    )

    # Columns we will plot
    col_baseline = "BASELINE (diffusion only)"
    col_unet = "baseline_A0"
    col_cross = "joint_cross_attn_a7_v2"

    if col_cross not in merged.columns:
        raise RuntimeError(f"Expected column missing: {col_cross}")

    x_labels = [f"({row.alpha_s:.3f}, {row.alpha_a:.3f})" for _, row in merged.iterrows()]
    x = range(len(merged))

    plt.figure(figsize=(14, 6))

    plt.plot(
        x,
        merged[col_baseline].values,
        label="BASELINE (diffusion policy)",
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=4,
    )

    if col_unet in merged.columns:
        plt.plot(
            x,
            merged[col_unet].values,
            label="Unet Joint denoiser",
            color="#d62728",
            linewidth=2.2,
            marker="s",
            markersize=4,
        )
    else:
        print(f"Warning: column {col_unet} not found; skipping Unet series")

    plt.plot(
        x,
        merged[col_cross].values,
        label="Joint Cross attn",
        color="#2ca02c",
        linewidth=2.2,
        marker="^",
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
    plt.savefig(OUT_PNG, dpi=200, bbox_inches="tight")
    print(f"Saved: {OUT_PNG}")


if __name__ == "__main__":
    main()
