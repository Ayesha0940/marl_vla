import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "joint_denoiser_eval.csv",
)
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "joint_denoiser_eval.png",
)


def load_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {"alpha_s", "alpha_a", "BASELINE (diffusion only)", "joint_baseline_a0"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {sorted(missing)}")

    df = df.copy()
    df["alpha_s"] = df["alpha_s"].astype(float)
    df["alpha_a"] = df["alpha_a"].astype(float)
    df["BASELINE (diffusion only)"] = df["BASELINE (diffusion only)"].astype(float)
    df["joint_baseline_a0"] = df["joint_baseline_a0"].astype(float)
    df = df.sort_values(["alpha_s", "alpha_a"]).reset_index(drop=True)
    df["alpha_combo"] = df.apply(
        lambda row: f"({row.alpha_s:.3f}, {row.alpha_a:.3f})",
        axis=1,
    )
    return df


def plot_data(df: pd.DataFrame, out_path: str | None) -> None:
    x = range(len(df))

    plt.figure(figsize=(14, 6))
    plt.plot(
        x,
        df["BASELINE (diffusion only)"].values,
        label="Baseline diffusion",
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=4,
    )
    plt.plot(
        x,
        df["joint_baseline_a0"].values,
        label="Joint baseline",
        color="#d62728",
        linewidth=2.2,
        marker="s",
        markersize=4,
    )

    plt.xticks(list(x), df["alpha_combo"].tolist(), rotation=45, ha="right", fontsize=8)
    plt.xlabel(r"$(\alpha_s, \alpha_a)$")
    plt.ylabel("Success rate")
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot joint denoiser evaluation results.")
    parser.add_argument("--csv", type=str, default=DEFAULT_CSV, help="Path to joint_denoiser_eval.csv")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT, help="Output image path")
    parser.add_argument("--no-save", action="store_true", help="Show the plot without saving it")
    args = parser.parse_args()

    df = load_data(args.csv)
    plot_data(df, None if args.no_save else args.output)


if __name__ == "__main__":
    main()