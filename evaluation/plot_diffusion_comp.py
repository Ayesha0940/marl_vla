"""
Plot robustness results comparing state / vision / state+vision conditioning modes.

Each CSV file is expected to have columns:
    method, noise_std, mean_reward, success_rate, n_rollouts, seed, t_start

Usage (compare three conditioning modes):
    python plot_diffusion_robustness.py \\
        --state    results/lift/state.csv \\
        --vision   results/lift/vision.csv \\
        --state-vision results/lift/state_vision.csv \\
        --output   results/lift/comparison.png

Usage (single CSV, legacy behaviour):
    python plot_diffusion_robustness.py --csv results/lift/state.csv

For the multi-CSV mode the script:
  - Picks the BEST diffusion value per noise level from each file.
  - Averages the 'none' baseline across all provided files.
  - Ignores kalman / ema / median rows entirely.
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_csv(path):
    """Load a robustness CSV, skipping comment / header lines gracefully."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    # Skip lines that don't start with a recognised method or 'method' header
    df = pd.read_csv(path, comment='#')
    # Drop any row whose 'method' value is not a real method label
    # (handles the single-word label lines like "state+vision" at the top)
    valid_methods = {'none', 'kalman', 'ema', 'median'}
    is_valid = (
        df['method'].astype(str).str.startswith('diffusion') |
        df['method'].astype(str).isin(valid_methods)
    )
    df = df[is_valid].copy()
    df['noise_std'] = pd.to_numeric(df['noise_std'], errors='coerce')
    df['success_rate'] = pd.to_numeric(df['success_rate'], errors='coerce')
    df['mean_reward'] = pd.to_numeric(df['mean_reward'], errors='coerce')
    return df


def best_diffusion_per_noise(df, metric):
    """Return the best diffusion row per noise_std level."""
    diff_df = df[df['method'].astype(str).str.startswith('diffusion')].copy()
    if diff_df.empty:
        return pd.DataFrame(columns=['noise_std', metric])
    best = (
        diff_df
        .sort_values(by=['noise_std', metric], ascending=[True, False])
        .groupby('noise_std', as_index=False)
        .first()
    )
    return best[['noise_std', metric, 'method']].rename(columns={'method': 'best_t'})


def none_baseline(df, metric):
    """Return the 'none' (no denoising) rows."""
    return df[df['method'] == 'none'][['noise_std', metric]].copy()


# ---------------------------------------------------------------------------
# Multi-CSV comparison plot
# ---------------------------------------------------------------------------

COLORS = {
    'state':        '#1f77b4',   # blue
    'vision':       '#ff7f0e',   # orange
    'state+vision': '#2ca02c',   # green
    'none':         '#7f7f7f',   # grey
}

MARKERS = {
    'state':        'o',
    'vision':       's',
    'state+vision': '^',
    'none':         'D',
}


def plot_comparison(csv_map, metric, output_file):
    """
    csv_map: dict  {label: path}  e.g. {'state': 'state.csv', 'vision': 'vision.csv', ...}
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect 'none' baselines to average
    none_frames = []

    print("\n=== Best diffusion config per noise std ===")

    for label, path in csv_map.items():
        df = load_csv(path)

        # Best diffusion line for this conditioning mode
        best = best_diffusion_per_noise(df, metric)
        if not best.empty:
            print(f"\n[{label}]")
            for _, row in best.sort_values('noise_std').iterrows():
                print(f"  noise={row['noise_std']:.2f}  t_start from '{row['best_t']}'  {metric}={row[metric]:.4f}")

            ax.plot(
                best['noise_std'], best[metric],
                marker=MARKERS.get(label, 'o'),
                linewidth=2,
                label=f"diffusion ({label})",
                color=COLORS.get(label),
            )

        # Collect none baseline
        none_df = none_baseline(df, metric)
        none_df = none_df.rename(columns={metric: f'_{label}'})
        none_frames.append(none_df.set_index('noise_std'))

    # Average 'none' across all files
    if none_frames:
        combined_none = pd.concat(none_frames, axis=1)
        avg_none = combined_none.mean(axis=1).reset_index()
        avg_none.columns = ['noise_std', metric]
        avg_none = avg_none.sort_values('noise_std')

        print(f"\n[none — averaged across {list(csv_map.keys())}]")
        for _, row in avg_none.iterrows():
            print(f"  noise={row['noise_std']:.2f}  {metric}={row[metric]:.4f}")

        ax.plot(
            avg_none['noise_std'], avg_none[metric],
            marker=MARKERS['none'],
            linewidth=2,
            linestyle='--',
            label='none (avg)',
            color=COLORS['none'],
        )

    ax.set_xlabel('Noise Std', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f"{metric.replace('_', ' ').title()} vs Noise Std — Conditioning Mode Comparison", fontsize=13)
    ax.legend(title='Method / Cond. Mode', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.6)
    fig.tight_layout()

    fig.savefig(output_file, dpi=150)
    print(f"\nPlot saved to {output_file}")


# ---------------------------------------------------------------------------
# Legacy single-CSV plot  (unchanged behaviour)
# ---------------------------------------------------------------------------

def plot_single(csv_file, output_file, metric, plot_diffusion_lines):
    df = load_csv(csv_file)

    if output_file is None:
        output_file = csv_file.replace('.csv', f'_{metric}.png')

    fig, ax = plt.subplots(figsize=(10, 6))

    diffusion_mask = df['method'].astype(str).str.startswith('diffusion')

    if plot_diffusion_lines:
        non_diff = df
    else:
        non_diff = df[~diffusion_mask]

    for name, group in non_diff.groupby('method'):
        group = group.sort_values('noise_std')
        ax.plot(group['noise_std'], group[metric], marker='o', linewidth=2, label=str(name))

    if diffusion_mask.any():
        best = best_diffusion_per_noise(df, metric)
        print('Best diffusion config per noise std:')
        for _, row in best.sort_values('noise_std').iterrows():
            print(f"  noise={row['noise_std']:.2f} -> {row['best_t']}, {metric}={row[metric]:.4f}")

        ax.plot(
            best['noise_std'], best[metric],
            marker='D', linewidth=3, linestyle='--',
            label='diffusion_best', color='black',
        )

    ax.set_xlabel('Noise Std', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title(f"{metric.replace('_', ' ').title()} vs Noise Std", fontsize=14)
    ax.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, linestyle='--', alpha=0.7)
    fig.tight_layout()

    fig.savefig(output_file, dpi=150)
    print(f"Plot saved to {output_file}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=(
            'Plot robustness results. '
            'Use --state / --vision / --state-vision for a 3-way comparison, '
            'or --csv for the legacy single-file mode.'
        )
    )

    # Multi-CSV comparison mode
    parser.add_argument('--state',        type=str, default='results/can/robustness_eval_can_state.csv', help='')
    parser.add_argument('--vision',       type=str, default='results/can/robustness_eval_can_vision.csv', help='CSV file for vision conditioning')
    parser.add_argument('--state-vision', type=str, default='results/can/robustness_eval_can_state_vision.csv', dest='state_vision',
                        help='CSV file for state+vision conditioning')

    # Legacy single-CSV mode
    parser.add_argument('--csv', type=str, default=None, help='Single CSV file (legacy mode)')
    parser.add_argument('--plot-diffusion-lines', action='store_true',
                        help='(legacy) plot all individual diffusion t_start lines')

    # Shared
    parser.add_argument('--output', type=str, default='results/can/comparison.png', help='Output image path')
    parser.add_argument('--metric', type=str, default='success_rate',
                        choices=['success_rate', 'mean_reward'],
                        help='Metric to plot')

    args = parser.parse_args()

    # Decide mode
    multi_inputs = {k: v for k, v in [
        ('state',        args.state),
        ('vision',       args.vision),
        ('state+vision', args.state_vision),
    ] if v is not None}

    if multi_inputs:
        # Multi-CSV comparison mode
        out = args.output or 'robustness_comparison.png'
        plot_comparison(multi_inputs, args.metric, out)

    elif args.csv:
        # Legacy single-CSV mode
        out = args.output
        if out is None:
            out = args.csv.replace('.csv', f'_{args.metric}.png')
        plot_single(args.csv, out, args.metric, args.plot_diffusion_lines)

    else:
        parser.error('Provide at least one of --state / --vision / --state-vision / --csv')