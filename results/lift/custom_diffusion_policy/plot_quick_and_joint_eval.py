import argparse
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


DATA_DIR = os.path.dirname(os.path.abspath(__file__))
QUICK_EVAL_CSV = os.path.join(
    DATA_DIR,
    "..",
    "diffusion_joint_denoiser",
    "quick_eval.csv",
)
JOINT_EVAL_CSV = os.path.join(DATA_DIR, "joint_denoiser_eval.csv")
DEFAULT_OUT = os.path.join(DATA_DIR, "quick_and_joint_eval.png")


def load_eval_table(csv_path: str, value_columns: list[str], rename_prefix: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {"alpha_s", "alpha_a"}.union(value_columns)
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {sorted(missing)}")

    df = df.copy()
    df["alpha_s"] = df["alpha_s"].astype(float)
    df["alpha_a"] = df["alpha_a"].astype(float)
    for column in value_columns:
        df[column] = df[column].astype(float)

    df = df.sort_values(["alpha_s", "alpha_a"]).reset_index(drop=True)
    renamed = {column: f"{rename_prefix}{column}" for column in value_columns}
    return df.rename(columns=renamed)


def load_joint_eval(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    expected_cols = {"alpha_s", "alpha_a", "BASELINE (diffusion only)", "joint_lam01_a0"}
    missing = expected_cols.difference(df.columns)
    if missing:
        raise ValueError(f"{csv_path} is missing expected columns: {sorted(missing)}")

    df = df.copy()
    df["alpha_s"] = df["alpha_s"].astype(float)
    df["alpha_a"] = df["alpha_a"].astype(float)
    df["BASELINE (diffusion only)"] = df["BASELINE (diffusion only)"].astype(float)
    df["joint_lam01_a0"] = df["joint_lam01_a0"].astype(float)
    df = df.sort_values(["alpha_s", "alpha_a"]).reset_index(drop=True)
    df["alpha_combo"] = df.apply(
        lambda row: f"({row.alpha_s:.3f}, {row.alpha_a:.3f})",
        axis=1,
    )
    return df


def plot_figure(quick_df: pd.DataFrame, joint_df: pd.DataFrame, out_path: str | None) -> None:
    merged = quick_df.merge(
        joint_df,
        on=["alpha_s", "alpha_a"],
        how="inner",
        validate="one_to_one",
    ).sort_values(["alpha_s", "alpha_a"]).reset_index(drop=True)

    if len(merged) != len(quick_df) or len(merged) != len(joint_df):
        raise ValueError("The two CSVs do not share the same alpha_s/alpha_a grid.")

    merged["alpha_combo"] = merged.apply(
        lambda row: f"({row.alpha_s:.3f}, {row.alpha_a:.3f})",
        axis=1,
    )

    x = range(len(merged))
    fig, ax = plt.subplots(figsize=(16, 6), constrained_layout=True)
    ax.plot(
        x,
        merged["quick_BASELINE (diffusion only)"].values,
        label="Quick eval: baseline diffusion",
        color="#1f77b4",
        linewidth=2.2,
        marker="o",
        markersize=4,
    )
    ax.plot(
        x,
        merged["quick_baseline_A0"].values,
        label="Quick eval: baseline_A0",
        color="#ff7f0e",
        linewidth=2.2,
        marker="s",
        markersize=4,
    )
    ax.plot(
        x,
        merged["BASELINE (diffusion only)"].values,
        label="Joint eval: baseline diffusion",
        color="#2ca02c",
        linewidth=2.2,
        marker="^",
        markersize=4,
    )
    ax.plot(
        x,
        merged["joint_lam01_a0"].values,
        label="Joint eval: joint_lam01_a0",
        color="#d62728",
        linewidth=2.2,
        marker="D",
        markersize=4,
    )

    ax.set_xlabel(r"$(\alpha_s, \alpha_a)$")
    ax.set_ylabel("Success rate")
    ax.set_xticks(list(x))
    ax.set_xticklabels(merged["alpha_combo"].tolist(), rotation=45, ha="right", fontsize=8)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax.set_ylim(0.0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(ncol=2)

    if out_path:
        fig.savefig(out_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {out_path}")

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot quick eval and joint denoiser eval together.")
    parser.add_argument("--quick-csv", type=str, default=QUICK_EVAL_CSV, help="Path to quick_eval.csv")
    parser.add_argument("--joint-csv", type=str, default=JOINT_EVAL_CSV, help="Path to joint_denoiser_eval.csv")
    parser.add_argument("--output", type=str, default=DEFAULT_OUT, help="Output image path")
    parser.add_argument("--no-save", action="store_true", help="Show the figure without saving it")
    args = parser.parse_args()

    quick_df = load_eval_table(args.quick_csv, ["BASELINE (diffusion only)", "baseline_A0"], "quick_")
    joint_df = load_joint_eval(args.joint_csv)
    plot_figure(quick_df, joint_df, None if args.no_save else args.output)


if __name__ == "__main__":
    main()