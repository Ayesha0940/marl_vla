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

# Sort configs: easiest (high base_noisy) → hardest (low base_noisy).
# Ties broken by alpha_s then alpha_a ascending (more noise = harder).
sorted_base = baselines.sort_values(
    ['base_noisy', 'alpha_s', 'alpha_a'],
    ascending=[False, True, True]
).reset_index(drop=True)

x_labels = [f"({row.alpha_s:.2f},\n{row.alpha_a:.2f})" for _, row in sorted_base.iterrows()]
x = range(len(sorted_base))

# Align each method dataframe to the same sorted order.
def align(df):
    merged = sorted_base[['alpha_s', 'alpha_a']].merge(df, on=['alpha_s', 'alpha_a'], how='left')
    return merged

fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharey=True, sharex=True)
axes = axes.flatten()

colors = {'base_noisy': '#888888', 'A0': '#1f77b4', 'A7': '#ff7f0e'}

for idx, (name, df) in enumerate(methods.items()):
    ax = axes[idx]
    aligned = align(df)

    base_noisy_vals = sorted_base['base_noisy'].values
    a0_vals = aligned['A0(t=10)'].values
    a7_vals = aligned['A7(t=10)'].values

    # Reference line: perfect clean performance
    ax.axhline(1.0, color='black', linewidth=0.8, linestyle=':', alpha=0.4, label='Base clean (1.0)')

    # Shaded recovery gap: base_noisy → A0 and base_noisy → A7
    ax.fill_between(x, base_noisy_vals, a0_vals, alpha=0.10, color=colors['A0'])
    ax.fill_between(x, base_noisy_vals, a7_vals, alpha=0.10, color=colors['A7'])

    ax.plot(x, base_noisy_vals, color=colors['base_noisy'], linewidth=2,
            linestyle='--', marker='o', markersize=4, label='Base noisy (no denoiser)')
    ax.plot(x, a0_vals, color=colors['A0'], linewidth=2,
            linestyle='-', marker='s', markersize=4, label='A0 (denoised)')
    ax.plot(x, a7_vals, color=colors['A7'], linewidth=2,
            linestyle='-', marker='^', markersize=4, label='A7 (denoised)')

    ax.set_title(name, fontsize=13, fontweight='bold')
    ax.set_ylim(-0.05, 1.08)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_xticks(list(x))
    ax.set_xticklabels(x_labels, fontsize=7.5)
    ax.grid(axis='y', alpha=0.3)
    ax.grid(axis='x', alpha=0.15)

    # Difficulty arrow annotation on bottom subplot
    if idx >= 2:
        ax.set_xlabel('Noise configuration (α_s, α_a)  →  increasing difficulty', fontsize=9)

    if idx == 0:
        ax.legend(fontsize=9, loc='upper right')

fig.suptitle(
    'Diffusion Denoiser Ablation: Recovery of Agent Performance Under Noise\n'
    '(sorted easiest → hardest, left → right)',
    fontsize=13, y=1.01
)

plt.tight_layout()
out_path = os.path.join(DATA_DIR, 'ablation_line_chart.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"Saved: {out_path}")
plt.show()
