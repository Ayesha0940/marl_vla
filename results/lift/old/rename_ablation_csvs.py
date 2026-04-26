#!/usr/bin/env python3
"""
Rename existing ablation CSVs to include the experiment variant tag.

Files are matched to variants by chronological order within each
(anchor, t_start) group, which mirrors the fixed execution order in
ablation_noise_loss.sh:
    baseline → asym_noise → no_warmstart → lam01 → lam025 → asym_lam01 → all_three

Only files whose timestamp date is 20260425 (the ablation run day) and
that live directly in this directory (not old/) are touched.

New filename format:
    joint_{anchor}_tstart{t}_{variant}_{timestamp}.csv

Usage:
    python rename_ablation_csvs.py            # dry-run: print proposed renames
    python rename_ablation_csvs.py --execute  # actually rename
"""

import argparse
import os
import re
from collections import defaultdict

ABLATION_DATE = "20260425"

VARIANT_ORDER = [
    "baseline",
    "asym_noise",
    "no_warmstart",
    "lam01",
    "lam025",
    "asym_lam01",
    "all_three",
]

# joint_A0_tstart10_20260425_060457.csv
PATTERN = re.compile(
    r"^joint_(A\d+)_tstart(\d+)_(\d{8}_\d{6})\.csv$"
)

HERE = os.path.dirname(os.path.abspath(__file__))


def collect_ablation_files():
    """Return {(anchor, tstart): [filename, ...]} sorted by timestamp for ablation date."""
    groups = defaultdict(list)
    for fname in os.listdir(HERE):
        m = PATTERN.match(fname)
        if not m:
            continue
        anchor, tstart, ts = m.group(1), m.group(2), m.group(3)
        if not ts.startswith(ABLATION_DATE):
            continue
        groups[(anchor, tstart)].append((ts, fname))
    # sort each group chronologically
    for key in groups:
        groups[key].sort(key=lambda x: x[0])
    return groups


def build_rename_plan(groups):
    """Return list of (old_path, new_path) pairs."""
    plan = []
    warnings = []
    for (anchor, tstart), entries in sorted(groups.items()):
        n = len(entries)
        if n > len(VARIANT_ORDER):
            warnings.append(
                f"  [WARN] {anchor} tstart{tstart}: {n} files but only "
                f"{len(VARIANT_ORDER)} variants — extra files skipped"
            )
        for i, (ts, fname) in enumerate(entries):
            if i >= len(VARIANT_ORDER):
                break
            variant = VARIANT_ORDER[i]
            new_name = f"joint_{anchor}_tstart{tstart}_{variant}_{ts}.csv"
            old_path = os.path.join(HERE, fname)
            new_path = os.path.join(HERE, new_name)
            plan.append((old_path, new_path, anchor, tstart, variant))
    return plan, warnings


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--execute", action="store_true",
                   help="Actually rename files. Default is dry-run only.")
    args = p.parse_args()

    groups = collect_ablation_files()
    if not groups:
        print("No ablation CSV files found for date", ABLATION_DATE)
        return

    plan, warnings = build_rename_plan(groups)

    for w in warnings:
        print(w)

    print(f"\n{'OLD NAME':<60}  →  NEW NAME")
    print("─" * 120)
    for old_path, new_path, anchor, tstart, variant in plan:
        old_name = os.path.basename(old_path)
        new_name = os.path.basename(new_path)
        marker = "" if old_name != new_name else "  [already named]"
        print(f"  {old_name:<58}  →  {new_name}{marker}")

    n_changes = sum(1 for o, n, *_ in plan if os.path.basename(o) != os.path.basename(n))
    print(f"\n{len(plan)} files matched, {n_changes} need renaming.")

    if not args.execute:
        print("\nDry-run — pass --execute to apply.")
        return

    renamed = 0
    for old_path, new_path, *_ in plan:
        if old_path == new_path:
            continue
        if os.path.exists(new_path):
            print(f"  [SKIP] target already exists: {os.path.basename(new_path)}")
            continue
        os.rename(old_path, new_path)
        renamed += 1
        print(f"  renamed: {os.path.basename(old_path)}  →  {os.path.basename(new_path)}")

    print(f"\nDone. {renamed} files renamed.")


if __name__ == "__main__":
    main()
