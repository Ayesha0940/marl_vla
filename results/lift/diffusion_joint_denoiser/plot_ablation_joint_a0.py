import os
import argparse

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd


DEFAULT_DATA_PATH = os.path.join(os.path.dirname(__file__), "ablation_results_joint_a0.csv")
DEFAULT_OUT_PATH = os.path.join(os.path.dirname(__file__), "ablation_results_joint_a0_plot.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot one or more ablation CSVs on the same alpha_s/alpha_a grid.")
    parser.add_argument("--csv", nargs="+", default=[DEFAULT_DATA_PATH], help="One or more CSV paths to plot.")
    parser.add_argument("--out", default=None, help="Path to output image. Defaults to a name derived from the CSVs.")
    parser.add_argument("--baseline-col", action="append", default=None, help="Baseline column name for a CSV. Repeat once per CSV if needed.")
    parser.add_argument("--method-col", action="append", default=None, help="Method column name for a CSV. Repeat once per CSV if needed.")
    parser.add_argument("--label", action="append", default=None, help="Optional label prefix for a CSV. Repeat once per CSV if needed.")
    parser.add_argument("--title", default=None, help="Optional plot title.")
    return parser.parse_args()


def infer_series_columns(df: pd.DataFrame, baseline_col: str | None, method_col: str | None) -> tuple[str, str]:
    metric_cols = [c for c in df.columns if c not in {"alpha_s", "alpha_a"}]
    if len(metric_cols) < 2:
        raise ValueError("CSV must contain at least two metric columns besides alpha_s and alpha_a.")

    inferred_baseline = baseline_col
    if inferred_baseline is None:
        baseline_candidates = [c for c in metric_cols if "baseline" in c.lower()]
        inferred_baseline = baseline_candidates[0] if baseline_candidates else metric_cols[0]

    inferred_method = method_col
    if inferred_method is None:
        inferred_method = next(c for c in metric_cols if c != inferred_baseline)

    if inferred_baseline not in df.columns:
        raise ValueError(f"Baseline column not found: {inferred_baseline}")
    if inferred_method not in df.columns:
        raise ValueError(f"Method column not found: {inferred_method}")

    return inferred_baseline, inferred_method


def default_output_path(csv_path: str) -> str:
    base, _ = os.path.splitext(csv_path)
    return f"{base}_plot.png"


def pick_or_infer_columns(df: pd.DataFrame, baseline_col: str | None, method_col: str | None) -> tuple[str, str]:
    return infer_series_columns(df, baseline_col, method_col)


def default_multi_output_path(csv_paths: list[str]) -> str:
    stems = [os.path.splitext(os.path.basename(path))[0] for path in csv_paths]
    joined = "_vs_".join(stems[:2]) if len(stems) >= 2 else stems[0]
    return os.path.join(os.path.dirname(csv_paths[0]), f"{joined}_plot.png")


def main() -> None:
    args = parse_args()
    csv_paths = [os.path.abspath(path) for path in args.csv]
    out_path = os.path.abspath(args.out) if args.out else default_multi_output_path(csv_paths)

    label_prefixes = args.label or []
    baseline_cols = args.baseline_col or []
    method_cols = args.method_col or []

    series_list = []
    reference_grid = None
    reference_labels = None

    for index, csv_path in enumerate(csv_paths):
        df = pd.read_csv(csv_path)
        df = df.sort_values(["alpha_s", "alpha_a"]).reset_index(drop=True)
        baseline_col, method_col = pick_or_infer_columns(
            df,
            baseline_cols[index] if index < len(baseline_cols) else None,
            method_cols[index] if index < len(method_cols) else None,
        )

        grid = [(float(row.alpha_s), float(row.alpha_a)) for _, row in df.iterrows()]
        labels = [f"({row.alpha_s:.3f}, {row.alpha_a:.3f})" for _, row in df.iterrows()]

        if reference_grid is None:
            reference_grid = grid
            reference_labels = labels
        elif grid != reference_grid:
            raise ValueError(f"CSV grid mismatch: {csv_path} does not match the first CSV.")

        prefix = label_prefixes[index] if index < len(label_prefixes) else os.path.splitext(os.path.basename(csv_path))[0]
        series_list.append((prefix, baseline_col, df[baseline_col].astype(float).values, "o"))
        series_list.append((prefix, method_col, df[method_col].astype(float).values, "s"))

    x = range(len(reference_grid or []))
    labels = reference_labels or []

    plt.figure(figsize=(14, 6))

    palette = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]
    for idx, (prefix, column_name, values, marker) in enumerate(series_list):
        plt.plot(
            x,
            values,
            label=f"{prefix}: {column_name}",
            color=palette[idx % len(palette)],
            linewidth=2.2,
            marker=marker,
            markersize=4,
        )

    plt.xticks(list(x), labels, rotation=45, ha="right", fontsize=8)
    plt.xlabel("(alpha_s, alpha_a)")
    plt.ylabel("Success rate")
    if args.title:
        plt.title(args.title)
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    plt.ylim(0.0, 1.05)
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    print(f"CSVs: {', '.join(csv_paths)}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()