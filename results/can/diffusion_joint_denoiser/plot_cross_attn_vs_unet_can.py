import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(DATA_DIR, "cross_attn_vs_unet_can.csv")
OUT_PNG = os.path.join(DATA_DIR, "cross_attn_vs_unet_can_plot.png")


def main():
    df = pd.read_csv(CSV_PATH)
    # Columns
    col_x1 = 'alpha_s'
    col_x2 = 'alpha_a'
    col_baseline = 'BASELINE (diffusion only)'
    col_unet = 'joint_all_three_a0_t5'
    col_cross = 'joint_cross_attn_can_a0_aux_t5'

    labels = [f"({row[col_x1]:.3f}, {row[col_x2]:.3f})" for _, row in df.iterrows()]
    x = range(len(df))

    plt.figure(figsize=(12, 6))

    plt.plot(x, df[col_baseline].values, label='BASELINE (diffusion policy)', color='#1f77b4', marker='o')
    plt.plot(x, df[col_unet].values, label='Unet Joint Denoiser', color='#d62728', marker='s')
    plt.plot(x, df[col_cross].values, label='Joint cross attn', color='#2ca02c', marker='^')

    plt.xticks(list(x), labels, rotation=45, ha='right', fontsize=8)
    plt.xlabel('(alpha_s, alpha_a)')
    plt.ylabel('Success rate')
    plt.gca().yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    plt.ylim(0.0, 1.05)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=200, bbox_inches='tight')
    print(f"Saved: {OUT_PNG}")


if __name__ == '__main__':
    main()
