"""
Pluggable anchor modules for joint (state, action) denoiser conditioning.

Each anchor implements the interface:
    extract(traj: dict) -> Tensor   # raw features, (B, raw_dim)
    compute(traj: dict) -> Tensor   # projected to (B, D_C=128)

Use build_anchor(anchor_id, **dims) to construct by string ID.

Expected keys in the 'traj' dict per anchor variant:
    A0: (none)
    A1: 'object_pose_t0'   (B, D_obj)  — object pose at episode t=0
    A2: 'gripper_history'  (B, K)      — last K commanded gripper values
    A3: 'proprio'          (B, D_prop) — joint_pos + eef_pos + eef_quat
    A4: 'phase'            (B,) int64  — 0=approach, 1=grasp, 2=lift
    A5: (none — constant one-hot)
    A6: A1 + A2 keys
    A7: A1 + A3 keys
    A8: A1 + A2 + A3 + A4 keys
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

D_C = 128   # fixed projected anchor output dimension


# ── Utilities ──────────────────────────────────────────────────────────────────

def _batch_size(traj: dict) -> int:
    for v in traj.values():
        if isinstance(v, Tensor):
            return v.shape[0]
    raise ValueError("Cannot determine batch size: no Tensor in trajectory dict.")


def _device(traj: dict):
    for v in traj.values():
        if isinstance(v, Tensor):
            return v.device
    return torch.device("cpu")


def _mlp(in_dim: int, hidden: int = 256, out_dim: int = D_C) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden), nn.Mish(), nn.Linear(hidden, out_dim))


# ── Base class ─────────────────────────────────────────────────────────────────

class Anchor(nn.Module):
    """Base anchor interface. Subclasses implement extract() and compute()."""

    def extract(self, traj: dict) -> Tensor:
        """Return raw anchor features (B, raw_dim)."""
        raise NotImplementedError

    def compute(self, traj: dict) -> Tensor:
        """Return projected anchor embedding (B, D_C). Call this at train/inference."""
        raise NotImplementedError


# ── A0: No anchor ──────────────────────────────────────────────────────────────

class AnchorNone(Anchor):
    """A0: Unanchored joint denoising — returns zero vector (B, D_C)."""

    def extract(self, traj: dict) -> Tensor:
        return torch.zeros(_batch_size(traj), D_C, device=_device(traj))

    def compute(self, traj: dict) -> Tensor:
        return self.extract(traj)


# ── A1: Initial object pose ────────────────────────────────────────────────────

class AnchorInitialObjectPose(Anchor):
    """
    A1: Object pose at episode t=0, held constant across all H-windows
    in that episode. Source: obs['object'] at env reset.
    """

    def __init__(self, object_dim: int = 14):
        super().__init__()
        self.proj = _mlp(object_dim)

    def extract(self, traj: dict) -> Tensor:
        if "object_pose_t0" not in traj:
            raise KeyError(
                "A1 (AnchorInitialObjectPose) requires 'object_pose_t0' "
                "(B, D_obj) in trajectory dict."
            )
        return traj["object_pose_t0"].float()

    def compute(self, traj: dict) -> Tensor:
        return self.proj(self.extract(traj))


# ── A2: Gripper history ────────────────────────────────────────────────────────

class AnchorGripperHistory(Anchor):
    """
    A2: Last K commanded gripper values (action dim 6 of previous K actions).
    Binary signal: commanded, not sensed.
    """

    def __init__(self, k: int = 5):
        super().__init__()
        self.k    = k
        self.proj = _mlp(k)

    def extract(self, traj: dict) -> Tensor:
        if "gripper_history" not in traj:
            raise KeyError(
                "A2 (AnchorGripperHistory) requires 'gripper_history' "
                "(B, K) in trajectory dict."
            )
        return traj["gripper_history"].float()

    def compute(self, traj: dict) -> Tensor:
        return self.proj(self.extract(traj))


# ── A3: Clean proprioception ───────────────────────────────────────────────────

class AnchorCleanProprioception(Anchor):
    """
    A3: Joint angles + EE pose, assumed uncorrupted by the noise process.

    THREAT MODEL NOTE: This anchor assumes encoder-level sensing (joint angles,
    EE pose via FK) is NOT subject to the same corruption as the state channel
    being denoised. This is defensible for comm/vision faults but must be
    stated explicitly when reporting results. It is NOT defensible against
    sensor-level attacks targeting joint encoders.

    Default proprio_dim=14: robot0_joint_pos(7) + robot0_eef_pos(3) + robot0_eef_quat(4)
    Set include_joint_vel=True to also include robot0_joint_vel(7) → proprio_dim=21.
    """

    def __init__(self, proprio_dim: int = 14):
        super().__init__()
        self.proj = _mlp(proprio_dim)

    def extract(self, traj: dict) -> Tensor:
        if "proprio" not in traj:
            raise KeyError(
                "A3 (AnchorCleanProprioception) requires 'proprio' "
                "(B, D_prop) in trajectory dict."
            )
        return traj["proprio"].float()

    def compute(self, traj: dict) -> Tensor:
        return self.proj(self.extract(traj))


# ── A4: Phase indicator ────────────────────────────────────────────────────────

class AnchorPhaseIndicator(Anchor):
    """
    A4: One-hot over {approach=0, grasp=1, lift=2}.

    At training: derived from clean demo labels via label_phases() in dataset.py.
    At inference: use a lightweight phase classifier trained on clean data.
    """

    N_PHASES = 3

    def __init__(self):
        super().__init__()
        self.proj = _mlp(self.N_PHASES, hidden=64)

    def extract(self, traj: dict) -> Tensor:
        if "phase" not in traj:
            raise KeyError(
                "A4 (AnchorPhaseIndicator) requires 'phase' "
                "(B,) int64 in trajectory dict."
            )
        return F.one_hot(traj["phase"].long(), num_classes=self.N_PHASES).float()

    def compute(self, traj: dict) -> Tensor:
        return self.proj(self.extract(traj))


# ── A5: Task ID ────────────────────────────────────────────────────────────────

class AnchorTaskID(Anchor):
    """
    A5: One-hot task embedding. Trivial for single-task Lift;
    included to preserve multi-task extensibility.
    """

    def __init__(self, n_tasks: int = 5, task_idx: int = 0):
        super().__init__()
        self.n_tasks  = n_tasks
        self.task_idx = task_idx
        self.proj     = _mlp(n_tasks, hidden=64)

    def extract(self, traj: dict) -> Tensor:
        B      = _batch_size(traj)
        device = _device(traj)
        one_hot              = torch.zeros(B, self.n_tasks, device=device)
        one_hot[:, self.task_idx] = 1.0
        return one_hot

    def compute(self, traj: dict) -> Tensor:
        return self.proj(self.extract(traj))


# ── A6: Initial object pose + gripper history (A1 + A2) ───────────────────────

class AnchorA6(Anchor):
    """A6: A1 + A2 — initial object pose concatenated with gripper history."""

    def __init__(self, object_dim: int = 14, k: int = 5):
        super().__init__()
        self.proj = _mlp(object_dim + k)

    def extract(self, traj: dict) -> Tensor:
        for key in ("object_pose_t0", "gripper_history"):
            if key not in traj:
                raise KeyError(f"A6 requires '{key}' in trajectory dict.")
        return torch.cat(
            [traj["object_pose_t0"].float(), traj["gripper_history"].float()],
            dim=-1,
        )

    def compute(self, traj: dict) -> Tensor:
        return self.proj(self.extract(traj))


# ── A7: Initial object pose + clean proprioception (A1 + A3) ──────────────────

class AnchorA7(Anchor):
    """A7: A1 + A3 — initial object pose + clean proprioception."""

    def __init__(self, object_dim: int = 14, proprio_dim: int = 14):
        super().__init__()
        self.proj = _mlp(object_dim + proprio_dim)

    def extract(self, traj: dict) -> Tensor:
        if "object_pose_t0" not in traj:
            raise KeyError("A7 requires 'object_pose_t0' in trajectory dict.")
        if "proprio" not in traj:
            raise KeyError("A7 requires 'proprio' in trajectory dict.")
        return torch.cat(
            [traj["object_pose_t0"].float(), traj["proprio"].float()],
            dim=-1,
        )

    def compute(self, traj: dict) -> Tensor:
        return self.proj(self.extract(traj))


# ── A8: All combined (A1 + A2 + A3 + A4) ─────────────────────────────────────

class AnchorA8(Anchor):
    """A8: A1 + A2 + A3 + A4 — full anchor set."""

    def __init__(self, object_dim: int = 14, k: int = 5, proprio_dim: int = 14):
        super().__init__()
        # 3 = AnchorPhaseIndicator.N_PHASES
        self.proj = _mlp(object_dim + k + proprio_dim + 3)

    def extract(self, traj: dict) -> Tensor:
        for key in ("object_pose_t0", "gripper_history", "proprio", "phase"):
            if key not in traj:
                raise KeyError(f"A8 requires '{key}' in trajectory dict.")
        return torch.cat(
            [
                traj["object_pose_t0"].float(),
                traj["gripper_history"].float(),
                traj["proprio"].float(),
                F.one_hot(traj["phase"].long(), num_classes=3).float(),
            ],
            dim=-1,
        )

    def compute(self, traj: dict) -> Tensor:
        return self.proj(self.extract(traj))


# ── Factory ────────────────────────────────────────────────────────────────────

def build_anchor(
    anchor_id: str,
    *,
    object_dim: int = 14,
    proprio_dim: int = 14,
    gripper_k: int = 5,
    n_tasks: int = 5,
    task_idx: int = 0,
) -> Anchor:
    """
    Construct an anchor module by string ID.

    Args:
        anchor_id:   'A0' through 'A8'
        object_dim:  dim of 'object_pose_t0' obs (A1, A6, A7, A8)
        proprio_dim: dim of 'proprio' obs (A3, A7, A8)
        gripper_k:   gripper history length K (A2, A6, A8)
        n_tasks:     total number of tasks for one-hot (A5)
        task_idx:    this task's index in the one-hot (A5)

    Returns:
        Anchor instance (nn.Module, trainable)
    """
    mapping = {
        "A0": lambda: AnchorNone(),
        "A1": lambda: AnchorInitialObjectPose(object_dim),
        "A2": lambda: AnchorGripperHistory(gripper_k),
        "A3": lambda: AnchorCleanProprioception(proprio_dim),
        "A4": lambda: AnchorPhaseIndicator(),
        "A5": lambda: AnchorTaskID(n_tasks, task_idx),
        "A6": lambda: AnchorA6(object_dim, gripper_k),
        "A7": lambda: AnchorA7(object_dim, proprio_dim),
        "A8": lambda: AnchorA8(object_dim, gripper_k, proprio_dim),
    }
    if anchor_id not in mapping:
        raise ValueError(
            f"Unknown anchor ID '{anchor_id}'. Valid options: {list(mapping.keys())}"
        )
    return mapping[anchor_id]()
