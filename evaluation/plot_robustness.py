#!/usr/bin/env python
import pandas as pd
import matplotlib.pyplot as plt
import os

def main():
    # Define the input file path
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_file = os.path.join(PROJECT_ROOT, 'results', 'robustness_eval_square_20260412_010241.csv')
    
    if not os.path.exists(csv_file):
        print(f"Error: Could not find {csv_file}")
        return

    # Load the data
    df = pd.read_csv(csv_file)

    # Group by the 'method' column which contains "none", "kalman", "ema", "median"
    category_col = 'method' 
    x_col = 'noise_std'
    y_col = 'success_rate'

    plt.figure(figsize=(10, 6))

    # Group the dataframe and plot each group as a separate line
    if category_col in df.columns:
        for name, group in df.groupby(category_col):
            # Sort values by noise_std to ensure the line plots continuously from left to right
            group = group.sort_values(by=x_col)
            plt.plot(group[x_col], group[y_col], marker='o', linewidth=2, label=str(name))
    else:
        print(f"Warning: Category column '{category_col}' not found. Check your CSV headers.")
        print(f"Available columns are: {', '.join(df.columns)}")
        # Fallback: plot everything as a single line if no category is found
        df = df.sort_values(by=x_col)
        plt.plot(df[x_col], df[y_col], marker='o', linewidth=2)

    # Formatting the plot
    plt.title('Robustness Evaluation on Lift Task', fontsize=16)
    plt.xlabel('Noise Standard Deviation (noise_std)', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    
    if category_col in df.columns:
        plt.legend(title='Filter Method', fontsize=12)
        
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()

    # Save the plot and show it
    output_image = csv_file.replace('.csv', '.png')
    plt.savefig(output_image, dpi=300)
    print(f"Plot saved successfully as: {output_image}")
    plt.show()

if __name__ == '__main__':
    main()