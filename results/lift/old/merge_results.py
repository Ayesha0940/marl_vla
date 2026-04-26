#!/usr/bin/env python3
"""
Merge all per-run CSVs into a single wide pivot table.

Output format (one row per alpha_s × alpha_a × experiment_tag):
  alpha_s, alpha_a, experiment_tag,
  base_clean, base_noisy,
  A0(t=5), A0(t=10), A2(t=5), A2(t=10), A3(t=5), A3(t=10), A7(t=5), A7(t=10)

Filename patterns recognised:
  joint_A0_tstart10_baseline_20260425_060457.csv   (tagged)
  joint_A7_tstart10_20260425_181840.csv            (untagged)

Usage:
    python merge_results.py            # dry-run: print preview
    python merge_results.py --execute  # write ablation_pivot.csv
"""

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path

HERE    = Path(__file__).parent
OUTPUT  = HERE / "ablation_pivot.csv"

# Anchors and t_start to include
ANCHORS  = ["A0", "A2", "A7"]
T_STARTS = ["10"]

_TS = r"(\d{8}_\d{6})"
_FNAME_PATTERNS = [
    re.compile(rf"^joint_(A\d+)_tstart(\d+)_(.+)_{_TS}$"),  # tagged
    re.compile(rf"^joint_(A\d+)_tstart(\d+)_{_TS}$"),        # untagged
]

def _parse_filename(stem):
    m = _FNAME_PATTERNS[0].match(stem)
    if m:
        return m.group(1), m.group(2), m.group(3)
    m = _FNAME_PATTERNS[1].match(stem)
    if m:
        return m.group(1), m.group(2), ""
    return None


def load_all():
    """
    Reads two sources and merges them:
      1. results.csv — long format (has anchor/t_start/experiment_tag columns)
      2. individual joint_*.csv files — old format (anchor/t_start/tag from filename)

    Returns:
        joint[(alpha_s, alpha_a, experiment_tag, anchor, t_start)] = success_rate
        base_noisy[(alpha_s, alpha_a)]  = success_rate
        base_clean                       = success_rate (scalar)
    """
    joint           = {}
    base_noisy      = {}
    base_clean_vals = []

    # ── Source 1: results.csv (long format) ───────────────────────────────────
    long_csv = HERE / "results.csv"
    if long_csv.exists():
        with open(long_csv, newline="") as f:
            for row in csv.DictReader(f):
                _ingest(row["method"], row["alpha_s"], row["alpha_a"],
                        row["success_rate"], row["anchor"], row["t_start"],
                        row["experiment_tag"],
                        joint, base_noisy, base_clean_vals)

    # ── Source 2: individual joint_*.csv files ────────────────────────────────
    skip = {"results.csv", "ablation_pivot.csv", "merged_results.csv"}
    for path in sorted(HERE.glob("joint_*.csv")):
        if path.name in skip:
            continue
        parsed = _parse_filename(path.stem)
        if parsed is None:
            continue
        anchor, t_start, tag = parsed
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                _ingest(row["method"], row["alpha_s"], row["alpha_a"],
                        row["success_rate"], anchor, t_start, tag,
                        joint, base_noisy, base_clean_vals)

    base_clean = sum(base_clean_vals) / len(base_clean_vals) if base_clean_vals else None
    return joint, base_noisy, base_clean


def _ingest(method, a_s, a_a, sr, anchor, t_start, tag,
            joint, base_noisy, base_clean_vals):
    if method == "BASE-clean":
        base_clean_vals.append(float(sr))
    elif method == "BASE-noisy":
        key = (a_s, a_a)
        if key not in base_noisy:
            base_noisy[key] = float(sr)
    elif method.startswith("JOINT-"):
        if anchor in ANCHORS and t_start in T_STARTS:
            key = (a_s, a_a, tag, anchor, t_start)
            joint[key] = float(sr)


def build_pivot(joint, base_noisy, base_clean):
    row_keys = sorted(
        {(a_s, a_a, tag) for (a_s, a_a, tag, *_) in joint},
        key=lambda x: (x[2], float(x[0]), float(x[1]))
    )

    col_keys  = [(a, t) for a in ANCHORS for t in T_STARTS]
    col_names = [f"{a}(t={t})" for a, t in col_keys]

    # Build all rows first
    rows = []
    for (a_s, a_a, tag) in row_keys:
        row = {
            "alpha_s":         a_s,
            "alpha_a":         a_a,
            "experiment_tag":  tag if tag else "(no tag)",
            "base_clean":      f"{base_clean:.4f}" if base_clean is not None else "",
            "base_noisy":      f"{base_noisy.get((a_s, a_a), float('nan')):.4f}",
        }
        for (anchor, t_start), col_name in zip(col_keys, col_names):
            val = joint.get((a_s, a_a, tag, anchor, t_start))
            row[col_name] = f"{val:.4f}" if val is not None else ""
        rows.append(row)

    # Drop columns that are empty across every row
    non_empty_cols = [
        c for c in col_names
        if any(r[c] != "" for r in rows)
    ]
    header = ["experiment_tag", "alpha_s", "alpha_a", "base_clean", "base_noisy"] + non_empty_cols
    rows   = [{k: r[k] for k in header} for r in rows]

    return header, rows


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true",
                   help="Write ablation_pivot.csv. Default is dry-run.")
    args = p.parse_args()

    joint, base_noisy, base_clean = load_all()
    header, rows = build_pivot(joint, base_noisy, base_clean)

    # Preview
    print(f"Pivot: {len(rows)} rows × {len(header)} columns")
    print(f"Columns: {header}")
    print(f"\nFirst 5 rows:")
    for r in rows[:5]:
        print("  ", {k: r[k] for k in header})

    if not args.execute:
        print(f"\nDry-run — pass --execute to write {OUTPUT.name}")
        return

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWritten: {OUTPUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
