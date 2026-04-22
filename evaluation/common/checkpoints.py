"""Shared checkpoint discovery helpers for evaluation scripts."""

import glob
import os
from typing import Optional


def find_latest_checkpoint_for_epoch(checkpoint_dir: str, epoch: int) -> Optional[str]:
    """Return latest matching checkpoint path for a specific epoch."""
    pattern = os.path.join(checkpoint_dir, f"*/models/model_epoch_{epoch}.pth")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None
