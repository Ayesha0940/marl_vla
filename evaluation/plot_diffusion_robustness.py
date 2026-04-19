import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def plot_data(csv_file, output_file=None, metric='success_rate', plot_diffusion_lines=False, filter_sweep_csv=None):
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        return

    df = pd.read_csv(csv_file)

    if filter_sweep_csv:
        if not os.path.exists(filter_sweep_csv):
            print(f"Error: Could not find {filter_sweep_csv}")
            return
        sweep_df = pd.read_csv(filter_sweep_csv)

        # Keep diffusion rows from the base CSV and replace filter rows with best sweep results.
        diffusion_df = df[df['method'].astype(str).str.contains('diffusion', na=False)].copy()
        none_df = df[df['method'].astype(str) == 'none'].copy() if 'none' in df['method'].values else pd.DataFrame()

        sweep_df = sweep_df[sweep_df['method'].astype(str).str.contains('best', na=False)]
        sweep_df['method'] = sweep_df['method'].astype(str).str.replace(r'\(best\)', '', regex=True)
        sweep_df = sweep_df[sweep_df['method'].isin(['ema', 'kalman', 'median'])]

        df = pd.concat([diffusion_df, none_df, sweep_df], ignore_index=True)

    category_col = 'method'
    x_col = 'noise_std'
    y_col = metric

    if output_file is None:
        output_file = csv_file.replace('.csv', f'_combined_{metric}.png')

    plt.figure(figsize=(10, 6))

    if category_col in df.columns:
        diffusion_mask = df[category_col].astype(str).str.contains('diffusion', na=False)
        if plot_diffusion_lines:
            plot_mask = slice(None)
        else:
            plot_mask = ~diffusion_mask

        for name, group in df[plot_mask].groupby(category_col):
            group = group.sort_values(by=x_col)
            plt.plot(group[x_col], group[y_col], marker='o', linewidth=2, label=str(name))

        if diffusion_mask.any():
            diffusion_df = df[diffusion_mask].copy()
            best_diffusion = (
                diffusion_df
                .sort_values(by=[x_col, y_col], ascending=[True, False])
                .groupby(x_col, as_index=False)
                .first()
            )
            best_diffusion['method'] = 'diffusion_best'

            print('Best diffusion config per noise std:')
            for _, row in best_diffusion.sort_values(by=x_col).iterrows():
                t_start = row.get('t_start', 'N/A')
                print(f"  noise={row[x_col]} -> t_start={t_start}, {y_col}={row[y_col]:.4f}")

            plt.plot(
                best_diffusion[x_col],
                best_diffusion[y_col],
                marker='D',
                linewidth=3,
                linestyle='--',
                label='diffusion_best',
                color='black',
            )
    else:
        df = df.sort_values(by=x_col)
        plt.plot(df[x_col], df[y_col], marker='o', linewidth=2)

    plt.xlabel('Noise Std', fontsize=12)
    plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
    title_text = f"{metric.replace('_', ' ').title()} vs Noise Std"
    plt.title(title_text, fontsize=14)
    plt.legend(title='Method', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(output_file)
    print(f"Plot saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot robustness CSV and highlight the best diffusion configuration per noise std.')
    parser.add_argument('--csv', type=str, help='Path to the CSV file to plot')
    parser.add_argument('--filters-csv', type=str, default=None, help='CSV file with sweep results for ema/kalman/median filters')
    parser.add_argument('--output', type=str, default=None, help='Output image path')
    parser.add_argument('--metric', type=str, default='success_rate', choices=['success_rate', 'mean_reward'], help='Metric to plot and use for best diffusion selection')
    parser.add_argument('--plot-diffusion-lines', action='store_true', help='Plot all individual diffusion lines in addition to the best diffusion line')

    args = parser.parse_args()

    if args.csv:
        csv_file = args.csv
    else:
        PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        csv_file = os.path.join(PROJECT_ROOT, 'results', 'square', 'robustness_eval_square_20260419_050551.csv')

    plot_data(
        csv_file,
        output_file=args.output,
        metric=args.metric,
        plot_diffusion_lines=args.plot_diffusion_lines,
        filter_sweep_csv=args.filters_csv,
    )
