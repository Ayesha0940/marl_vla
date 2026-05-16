#!/usr/bin/env python3
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    HAS_SEABORN = True
except Exception:
    sns = None
    HAS_SEABORN = False


def clean_cols(df):
    # normalize column names
    df = df.rename(columns=lambda s: s.strip())
    if 'BASELINE (diffusion only)' in df.columns:
        df = df.rename(columns={'BASELINE (diffusion only)': 'baseline_diffusion'})
    return df


def plot_heatmaps(df, metrics, out_path):
    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5), constrained_layout=True)
    if n == 1:
        axes = [axes]

    for ax, metric in zip(axes, metrics):
        pivot = df.pivot(index='alpha_s', columns='alpha_a', values=metric)
        # ensure sorted axes
        pivot = pivot.sort_index().sort_index(axis=1)
        if HAS_SEABORN:
            sns.heatmap(pivot, annot=True, fmt='.3f', cmap='viridis', ax=ax, cbar_kws={'label': metric})
        else:
            im = ax.imshow(pivot.values, cmap='viridis', aspect='auto')
            ax.set_xticks(range(pivot.shape[1]))
            ax.set_xticklabels([str(x) for x in pivot.columns], rotation=45)
            ax.set_yticks(range(pivot.shape[0]))
            ax.set_yticklabels([str(x) for x in pivot.index])
            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label(metric)
        ax.set_title(metric)
        ax.set_xlabel('alpha_a')
        ax.set_ylabel('alpha_s')

    fig.suptitle('Quick eval heatmaps')
    fig.savefig(out_path)
    print(f'Saved heatmap to {out_path}')


def plot_scatter(df, metric, out_path):
    plt.figure(figsize=(6,5))
    sc = plt.scatter(df['alpha_a'], df['alpha_s'], c=df[metric], cmap='viridis', s=120, edgecolor='k')
    plt.colorbar(sc, label=metric)
    plt.xlabel('alpha_a')
    plt.ylabel('alpha_s')
    plt.title(f'{metric} scatter')
    plt.grid(True, alpha=0.3)
    plt.savefig(out_path)
    print(f'Saved scatter to {out_path}')


def plot_line(df, metrics, out_path):
    df2 = df.copy()
    df2['alpha_combo'] = df2.apply(lambda r: f"s={r['alpha_s']:.3f},a={r['alpha_a']:.3f}", axis=1)
    order = df2.sort_values(['alpha_s', 'alpha_a'])['alpha_combo']
    # keep unique in sorted order
    order = list(dict.fromkeys(order))
    df2['alpha_combo'] = pd.Categorical(df2['alpha_combo'], categories=order, ordered=True)
    df2 = df2.sort_values('alpha_combo')

    plt.figure(figsize=(max(8, len(order) * 0.6), 5))
    styles = ['-o', '-s', '-^', '-d']
    for i, metric in enumerate(metrics):
        if metric not in df2.columns:
            print(f'Warning: metric {metric} not in data; skipping')
            continue
        plt.plot(df2['alpha_combo'], df2[metric], styles[i % len(styles)], label=metric)

    plt.xticks(rotation=45, ha='right')
    plt.xlabel('alpha_s,alpha_a')
    plt.ylabel('value')
    plt.title('Metrics vs alpha combos')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    print(f'Saved line plot to {out_path}')


def main():
    p = argparse.ArgumentParser(description='Plot quick_eval.csv')
    p.add_argument('csv', help='Path to quick_eval.csv')
    p.add_argument('-o', '--out', help='Output image path', default=None)
    p.add_argument('--metrics', help='Comma-separated metric columns (default: all numeric except alphas)', default=None)
    p.add_argument('--scatter', action='store_true', help='Also save a scatter for the first metric')
    p.add_argument('--line', action='store_true', help='Also save a line plot across alpha combos for the first metric')
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise SystemExit(f'CSV not found: {csv_path}')

    df = pd.read_csv(csv_path)
    df = clean_cols(df)

    # default metrics: all numeric columns excluding alpha_s/alpha_a
    candidates = [c for c in df.columns if c not in ('alpha_s', 'alpha_a')]
    if args.metrics:
        metrics = [m.strip() for m in args.metrics.split(',')]
    else:
        metrics = candidates

    out = args.out
    if out is None:
        out_dir = csv_path.parent
        out = out_dir / (csv_path.stem + '_plot.png')
    else:
        out = Path(out)

    plot_heatmaps(df, metrics, out)

    if args.scatter and len(metrics) > 0:
        scatter_out = out.with_name(out.stem + f'_{metrics[0]}_scatter.png')
        plot_scatter(df, metrics[0], scatter_out)

    if args.line and len(metrics) > 0:
        line_out = out.with_name(out.stem + f"_{'_'.join([m.replace(' ', '') for m in metrics])}_line.png")
        plot_line(df, metrics, line_out)


if __name__ == '__main__':
    main()
