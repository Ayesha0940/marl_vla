"""Shared result reporting and persistence helpers for evaluation scripts."""

import csv
import json
import os
from typing import Dict, Iterable, List, Sequence


def print_robustness_summary(results: Sequence[Dict], method_width: int = 18, table_width: int = 65) -> None:
    """Print a compact robustness summary table."""
    print("\n" + "=" * table_width)
    print("Summary Results")
    print("=" * table_width)
    for row in results:
        method = str(row.get("method", "none"))
        print(
            f"Noise {row['noise_std']:4.2f} | Method {method:{method_width}s} | "
            f"Reward={row['mean_reward']:7.4f} | Success={row['success_rate']*100:5.2f}%"
        )
    print("=" * table_width)


def save_results_json(results: Sequence[Dict], config: Dict, output_path: str) -> str:
    """Save result payload to JSON and return path."""
    with open(output_path, "w") as f:
        json.dump({"results": list(results), "config": config}, f, indent=4)
    print(f"Saved JSON: {output_path}")
    return output_path


def save_results_csv(
    results: Sequence[Dict],
    output_path: str,
    fieldnames: Iterable[str],
    float_keys: Iterable[str] = ("mean_reward", "success_rate"),
) -> str:
    """Save tabular results to CSV and return path."""
    float_key_set = set(float_keys)
    ordered_fields = list(fieldnames)

    with open(output_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=ordered_fields)
        writer.writeheader()

        for row in results:
            out = {}
            for key in ordered_fields:
                val = row.get(key, "")
                if key in float_key_set and isinstance(val, (float, int)):
                    out[key] = f"{float(val):.4f}"
                elif val is None:
                    out[key] = ""
                else:
                    out[key] = val
            writer.writerow(out)

    print(f"Saved CSV:  {output_path}")
    return output_path
