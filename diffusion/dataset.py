"""
Windowed dataset for joint (state, action) denoiser training.

Loads robomimic low-dim HDF5 demos, builds horizon-H sliding windows, and
computes anchor features per window. No window ever crosses an episode boundary
(verified by construction: windows are built independently per demo).

Each dataset item is a dict of tensors:
    'state'           : (H, D_s)  z-score normalized
    'action'          : (H, D_a)  z-score normalized
    'object_pose_t0'  : (D_obj,)  A1 anchor input — object pose at episode t=0
    'gripper_history' : (K,)      A2 anchor input — K commanded gripper values
    'proprio'         : (D_prop,) A3 anchor input — proprio at window center
    'phase'           : ()        A4 anchor input — int64 {0=approach,1=grasp,2=lift}

Typical usage:
    ds = JointDenoiserDataset('lift/ph/low_dim_v141.hdf5', horizon=16)
    loader = DataLoader(ds, batch_size=256, shuffle=True)

    anchor = build_anchor('A7', object_dim=ds.object_dim, proprio_dim=ds.proprio_dim)
    for batch in loader:
        anchor_emb = anchor.compute(batch)
        total, la, ls = joint_diffusion_loss(model, batch['state'], batch['action'],
                                             anchor_emb, alphas_bar)
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import List, Optional


# ── Phase labeling ─────────────────────────────────────────────────────────────

def label_phases(
    gripper_cmds: np.ndarray,
    object_z: np.ndarray,
    lift_z_thresh: float = 0.85,
) -> np.ndarray:
    """
    Rule-based phase labeler for Robomimic Lift.

    Phases:
        0 = approach  (gripper open)
        1 = grasp     (gripper closed, object not yet lifted)
        2 = lift      (gripper closed AND object z > lift_z_thresh)

    Args:
        gripper_cmds:  (T,) commanded gripper values; action dim 6 (<0.5 = closed)
        object_z:      (T,) world-frame z-coordinate of the object
        lift_z_thresh: threshold for "lifted" state (inspect obs histogram to tune)

    Returns:
        labels: (T,) int64
    """
    T      = len(gripper_cmds)
    labels = np.zeros(T, dtype=np.int64)
    for t in range(T):
        closed = gripper_cmds[t] < 0.5
        lifted = object_z[t] > lift_z_thresh
        if closed and lifted:
            labels[t] = 2
        elif closed:
            labels[t] = 1
        # else: approach (0)
    return labels


# ── Dataset ────────────────────────────────────────────────────────────────────

class JointDenoiserDataset(Dataset):
    """
    Horizon-H sliding windows from robomimic low-dim HDF5 demos.

    Args:
        hdf5_path:          path to robomimic .hdf5 file
        horizon:            window length H (default 16; spec suggests H=16 for Lift)
        obs_keys:           list of obs keys concatenated to form the state vector;
                            defaults to all standard Lift low-dim keys
        object_key:         obs key for object pose (A1 anchor + phase z-coord)
        proprio_keys:       obs keys concatenated for clean proprioception (A3 anchor)
        gripper_action_dim: which action dimension is the gripper command (default 6)
        gripper_k:          gripper history length K for A2 anchor (default 5)
        lift_z_thresh:      object-z threshold for phase labeling (default 0.85 m)
        normalize:          z-score normalize state and action windows (default True)
    """

    # Standard Robomimic Lift low-dim obs keys
    DEFAULT_OBS_KEYS = [
        "object",
        "robot0_eef_pos",
        "robot0_eef_quat",
        "robot0_gripper_qpos",
        "robot0_joint_pos",
        "robot0_joint_vel",
    ]
    # Keys used for A3 (clean proprioception)
    DEFAULT_PROPRIO_KEYS = ["robot0_joint_pos", "robot0_eef_pos", "robot0_eef_quat"]

    def __init__(
        self,
        hdf5_path: str,
        horizon: int = 16,
        obs_keys: Optional[List[str]] = None,
        object_key: str = "object",
        proprio_keys: Optional[List[str]] = None,
        gripper_action_dim: int = 6,
        gripper_k: int = 5,
        lift_z_thresh: float = 0.85,
        normalize: bool = True,
        aug_alpha_s_max: float = 0.05,
        aug_alpha_a_max: float = 0.20,
        noise_schedule: str = "uniform",
    ):
        self.horizon            = horizon
        self.obs_keys           = obs_keys or self.DEFAULT_OBS_KEYS
        self.object_key         = object_key
        self.proprio_keys       = proprio_keys or self.DEFAULT_PROPRIO_KEYS
        self.gripper_action_dim = gripper_action_dim
        self.gripper_k          = gripper_k
        self.lift_z_thresh      = lift_z_thresh
        self.aug_alpha_s_max    = aug_alpha_s_max
        self.aug_alpha_a_max    = aug_alpha_a_max
        assert noise_schedule in ("uniform", "asymmetric"), \
            f"noise_schedule must be 'uniform' or 'asymmetric', got {noise_schedule!r}"
        self.noise_schedule     = noise_schedule

        self._windows: List[dict] = []
        self._load(hdf5_path)

        if len(self._windows) == 0:
            raise RuntimeError(
                f"No windows created from {hdf5_path}. "
                f"Check that demos are longer than horizon={horizon}."
            )

        self.state_mean = self.state_std = None
        self.action_mean = self.action_std = None
        if normalize:
            self._fit_normalizer()

    # ── Loading ────────────────────────────────────────────────────────────────

    def _load(self, hdf5_path: str) -> None:
        with h5py.File(hdf5_path, "r") as f:
            demos = sorted(f["data"].keys())
            for dk in demos:
                demo = f["data"][dk]
                T    = demo["actions"].shape[0]
                if T < self.horizon:
                    continue

                # ── State vector ───────────────────────────────────────────────
                state = np.concatenate(
                    [demo["obs"][k][:].reshape(T, -1) for k in self.obs_keys],
                    axis=1,
                ).astype(np.float32)                              # (T, D_s)

                # ── Action ─────────────────────────────────────────────────────
                action = demo["actions"][:].astype(np.float32)   # (T, D_a)

                # ── A1: object pose at t=0 (fixed for all windows in episode) ──
                object_pose_t0 = demo["obs"][self.object_key][0].astype(np.float32)

                # ── A3: full proprio sequence for window-center lookup ──────────
                proprio = np.concatenate(
                    [demo["obs"][k][:].reshape(T, -1) for k in self.proprio_keys],
                    axis=1,
                ).astype(np.float32)                              # (T, D_prop)

                # ── A4: rule-based phase labels ────────────────────────────────
                gripper_cmds = action[:, self.gripper_action_dim]
                object_z     = demo["obs"][self.object_key][:, 2]   # world z
                phases       = label_phases(gripper_cmds, object_z, self.lift_z_thresh)

                # ── Sliding windows (no cross-episode leakage) ─────────────────
                for start in range(T - self.horizon + 1):
                    end = start + self.horizon

                    # A2: K gripper commands before this window
                    hist_start = max(0, start - self.gripper_k)
                    hist       = action[hist_start:start, self.gripper_action_dim]
                    if len(hist) < self.gripper_k:
                        hist = np.pad(hist, (self.gripper_k - len(hist), 0))
                    hist = hist.astype(np.float32)                 # (K,)

                    # Phase: use center timestep as representative for window
                    center_t     = start + self.horizon // 2
                    center_phase = int(phases[center_t])

                    self._windows.append({
                        "state":           state[start:end].copy(),    # (H, D_s)
                        "action":          action[start:end].copy(),   # (H, D_a)
                        "object_pose_t0":  object_pose_t0,             # (D_obj,)
                        "gripper_history": hist,                       # (K,)
                        "proprio":         proprio[center_t].copy(),   # (D_prop,)
                        "phase":           np.int64(center_phase),
                    })

    # ── Normalization ──────────────────────────────────────────────────────────

    def _fit_normalizer(self) -> None:
        """Z-score stats fitted over all windows (not per-episode)."""
        states  = np.stack([w["state"]  for w in self._windows])   # (N, H, D_s)
        actions = np.stack([w["action"] for w in self._windows])   # (N, H, D_a)
        self.state_mean  = states.mean(axis=(0, 1))                # (D_s,)
        self.state_std   = states.std(axis=(0, 1)).clip(1e-6)
        self.action_mean = actions.mean(axis=(0, 1))               # (D_a,)
        self.action_std  = actions.std(axis=(0, 1)).clip(1e-6)

    def get_normalization_stats(self) -> dict:
        """Return stats dict for saving alongside model checkpoints."""
        return {
            "state_mean":  self.state_mean,
            "state_std":   self.state_std,
            "action_mean": self.action_mean,
            "action_std":  self.action_std,
        }

    # ── Dimension properties ───────────────────────────────────────────────────

    @property
    def state_dim(self) -> int:
        return self._windows[0]["state"].shape[-1]

    @property
    def action_dim(self) -> int:
        return self._windows[0]["action"].shape[-1]

    @property
    def object_dim(self) -> int:
        return self._windows[0]["object_pose_t0"].shape[-1]

    @property
    def proprio_dim(self) -> int:
        return self._windows[0]["proprio"].shape[-1]

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._windows)

    def __getitem__(self, idx: int) -> dict:
        w = self._windows[idx]

        state_clean  = w["state"].copy()
        action_clean = w["action"].copy()

        # Deployment noise augmentation in raw space (before normalization),
        # matching how eval corrupts observations before the normalizer.
        if self.noise_schedule == "asymmetric":
            # Beta(3,1) concentrates alpha_s near aug_alpha_s_max (mean ≈ 0.75 * max),
            # biasing toward the high-noise regime [0.03, 0.05] where deployment lives.
            # alpha_a stays uniform — action noise is already variable at deployment.
            alpha_s = np.random.beta(3, 1) * self.aug_alpha_s_max
        else:
            alpha_s = np.random.uniform(0.0, self.aug_alpha_s_max)
        alpha_a = np.random.uniform(0.0, self.aug_alpha_a_max)
        state_noisy  = state_clean  + np.random.randn(*state_clean.shape).astype(np.float32)  * alpha_s
        action_noisy = action_clean + np.random.randn(*action_clean.shape).astype(np.float32) * alpha_a

        if self.state_mean is not None:
            state_clean  = (state_clean  - self.state_mean)  / self.state_std
            action_clean = (action_clean - self.action_mean) / self.action_std
            state_noisy  = (state_noisy  - self.state_mean)  / self.state_std
            action_noisy = (action_noisy - self.action_mean) / self.action_std

        return {
            "state":           torch.from_numpy(state_clean),
            "action":          torch.from_numpy(action_clean),
            "state_noisy":     torch.from_numpy(state_noisy),
            "action_noisy":    torch.from_numpy(action_noisy),
            "object_pose_t0":  torch.from_numpy(w["object_pose_t0"]),
            "gripper_history": torch.from_numpy(w["gripper_history"]),
            "proprio":         torch.from_numpy(w["proprio"]),
            "phase":           torch.tensor(w["phase"], dtype=torch.long),
        }
