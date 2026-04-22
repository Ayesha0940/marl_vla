"""Shared MuJoCo environment setup helpers for evaluation scripts."""

import os
from typing import Iterable, Tuple


DEFAULT_LIBRARY_PATHS: Tuple[str, ...] = (
    os.path.expanduser("~/.mujoco/mujoco210/bin"),
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib/x86_64-linux-gnu/nvidia",
)


def configure_mujoco_env(
    paths: Iterable[str] = DEFAULT_LIBRARY_PATHS,
    *,
    force_gl_egl: bool = False,
    verbose: bool = True,
) -> str:
    """Add discovered MuJoCo/NVIDIA paths to LD_LIBRARY_PATH and return it."""
    current = os.environ.get("LD_LIBRARY_PATH", "")
    discovered = [p for p in paths if os.path.exists(p)]

    updated = current
    for p in discovered:
        if p not in updated.split(":"):
            updated = f"{p}:{updated}" if updated else p

    os.environ["LD_LIBRARY_PATH"] = updated

    if force_gl_egl:
        os.environ.setdefault("MUJOCO_GL", "egl")

    if verbose:
        print("Updated LD_LIBRARY_PATH:")
        print(updated)
        if force_gl_egl:
            print(f"MUJOCO_GL={os.environ.get('MUJOCO_GL')}")

    return updated
