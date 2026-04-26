import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

baselines = pd.read_csv(os.path.join(DATA_DIR, 'baselines.csv'))
method_files = {
    'Baseline Diffusion': 'baseline_results.csv',
    r'$\lambda$=0.1':    'lam01_results.csv',
    'Asymmetric Noise':  'asym_noise_results.csv',
    'All Three':         'all_three_results.csv',
}
methods = {
    name: pd.read_csv(os.path.join(DATA_DIR, fname))
    for name, fname in method_files.items()
}

sorted_base = baselines.sort_values(
    ['base_noisy', 'alpha_s', 'alpha_a'],
    ascending=[False, True, True]
).reset_index(drop=True)

x_labels = [f"({row.alpha_s:.2f},\n{row.alpha_a:.2f})" for _, row in sorted_base.iterrows()]
x = range(len(sorted_base))

def align(df):
    return sorted_base[['alpha_s', 'alpha_a']].merge(df, on=['alpha_s', 'alpha_a'], how='left')

method_colors = {
    'Baseline Diffusion': '#1f77b4',
    r'$\lambda$=0.1':    '#2ca02c',
    'Asymmetric Noise':  '#d62728',
    'All Three':         '#9467bd',
}

fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharey=True, sharex=True)

for ax, anchor in zip(axes, ['A0(t=10)', 'A7(t=10)']):
    anchor_label = anchor.split('(')[0]

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle=':', alpha=0.4, label='Base clean (1.0)')

    base_noisy_vals = sorted_base['base_noisy'].values
    ax.plot(x, base_noisy_vals, color='#888888', linewidth=2,
            linestyle='--', marker='o', markersize=4, label='Base noisy (no denoiser)')

    for name, df in methods.items():
        aligned = align(df)
        vals = aligned[anchor].values
        ax.plot(x, vals, color=method_colors[name], linewidth=2,
                linestyle='-', marker='s', markersize=4, label=name)

    ax.set_title(f'Anchor {anchor_label}', fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, fontsize=7.5)
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', alpha=0.15)
    ax.legend(fontsize=9, loc='upper right')

axes[1].set_xlabel('Noise configuration (α_s, α_a)  →  increasing difficulty', fontsize=10)

fig.suptitle(
    'Ablation Study: All Denoiser Methods vs Base Noisy\n'
    '(sorted easiest → hardest, left → right)',
    fontsize=13, y=1.01
)

plt.tight_layout()
out_path = os.path.join(DATA_DIR, 'ablation_by_anchor.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
