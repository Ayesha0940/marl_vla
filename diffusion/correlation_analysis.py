"""
Quantify state↔action structural coupling in a robomimic HDF5.

What it measures (in order of usefulness for joint denoising):
  1. R²(s_t -> a_t)            — how well current state predicts current action
  2. R²((s_t,a_t) -> Δs_{t+1}) — how well (state,action) predicts next-state delta
                                  (this is the *forward dynamics* the joint denoiser
                                   should be learning implicitly)
  3. CCA canonical correlations — spectrum of independent (s,a) coupling modes
  4. Per-action-dim probes      — which action dims are most/least predictable from s
                                  (cheap predictability → policy is reactive → joint
                                   denoising should help a lot on those dims)

A high R²((s,a) -> Δs) is the strongest signal that an auxiliary dynamics
head will help: it means the constraint exists and is learnable.

Usage:
    python -m diffusion.correlation_analysis \\
        --hdf5_path datasets/lift/ph/low_dim_v141.hdf5
    python -m diffusion.correlation_analysis \\
        --hdf5_path diffusion_data/lift_policy_rollouts.hdf5
"""

import argparse

import h5py
import numpy as np
from sklearn.cross_decomposition import CCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler


DEFAULT_OBS_KEYS = [
    "object", "robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos"
]


def _load_pairs(hdf5_path: str, obs_keys):
    """Return (S, A, S_next) concatenated across all demos."""
    states, actions, next_states = [], [], []
    with h5py.File(hdf5_path, "r") as f:
        for dk in sorted(f["data"].keys()):
            demo = f["data"][dk]
            T = demo["actions"].shape[0]
            if T < 2:
                continue
            s = np.concatenate(
                [demo["obs"][k][:].reshape(T, -1) for k in obs_keys], axis=1
            ).astype(np.float32)
            a = demo["actions"][:].astype(np.float32)
            states.append(s[:-1])
            actions.append(a[:-1])
            next_states.append(s[1:])
    return (np.concatenate(x, axis=0) for x in (states, actions, next_states))


def _r2(X, Y, ridge_alpha=1e-3):
    """Centered+scaled ridge fit, report uniform-average R² across outputs."""
    Xs = StandardScaler().fit_transform(X)
    Ys = StandardScaler().fit_transform(Y)
    return Ridge(alpha=ridge_alpha).fit(Xs, Ys).score(Xs, Ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdf5_path", required=True)
    ap.add_argument("--obs_keys", nargs="+", default=DEFAULT_OBS_KEYS)
    ap.add_argument("--n_cca", type=int, default=7,
                    help="Number of CCA components (≤ min(D_s, D_a))")
    args = ap.parse_args()

    print(f"Loading: {args.hdf5_path}")
    print(f"Obs keys: {args.obs_keys}")
    S, A, Sn = _load_pairs(args.hdf5_path, args.obs_keys)
    dS = Sn - S
    N, Ds = S.shape
    _, Da = A.shape
    print(f"Pairs: N={N}, D_s={Ds}, D_a={Da}\n")

    # --- 1. State -> action ----------------------------------------------------
    r2_sa = _r2(S, A)
    print(f"[1] R²(s_t  ->  a_t)                  = {r2_sa:.3f}")
    print(f"    interpretation: how reactive the policy is. High → joint")
    print(f"    denoising can recover a from a clean-enough s.\n")

    # --- 2. (s, a) -> Δs (forward dynamics) -----------------------------------
    SA = np.concatenate([S, A], axis=1)
    r2_dyn = _r2(SA, dS)
    print(f"[2] R²((s_t,a_t) -> Δs_{{t+1}})        = {r2_dyn:.3f}")
    print(f"    interpretation: linear forward-dynamics quality. This is the")
    print(f"    auxiliary loss the proposed architecture will use directly.\n")

    # --- 3. Inverse dynamics: (s, s') -> a ------------------------------------
    SS = np.concatenate([S, Sn], axis=1)
    r2_inv = _r2(SS, A)
    print(f"[3] R²((s_t,s_{{t+1}}) -> a_t)         = {r2_inv:.3f}")
    print(f"    interpretation: inverse-dynamics quality. High → action is")
    print(f"    redundant given clean state pair; joint denoiser can in")
    print(f"    principle reconstruct a from neighbouring states.\n")

    # --- 4. CCA spectrum ------------------------------------------------------
    k = min(args.n_cca, Ds, Da)
    cca = CCA(n_components=k, max_iter=2000).fit(S, A)
    U, V = cca.transform(S, A)
    canon = []
    for i in range(k):
        c = np.corrcoef(U[:, i], V[:, i])[0, 1]
        canon.append(float(c))
    print(f"[4] Canonical correlations (top {k})  = "
          f"{['%.2f' % c for c in canon]}")
    print(f"    interpretation: number of strong modes (>0.5) is the rank")
    print(f"    of structural coupling that an attention model can exploit.\n")

    # --- 5. Per-action-dim probes ---------------------------------------------
    print(f"[5] Per-action-dim R²(s -> a_i):")
    Xs = StandardScaler().fit_transform(S)
    for i in range(Da):
        Ys = StandardScaler().fit_transform(A[:, i:i+1])
        r2_i = Ridge(alpha=1e-3).fit(Xs, Ys).score(Xs, Ys)
        marker = "★" if r2_i > 0.5 else (" " if r2_i > 0.2 else "·")
        print(f"    {marker}  a[{i}]  R² = {r2_i:+.3f}")
    print(f"    ★ ≥0.5 (highly predictable from s),  · <0.2 (near-stochastic)\n")

    # --- 6. Summary recommendation --------------------------------------------
    print("=" * 60)
    print("RECOMMENDATION")
    print("=" * 60)
    if r2_dyn > 0.6:
        print(f"R²_dyn={r2_dyn:.2f} is high — auxiliary forward-dynamics loss")
        print("should give a clean signal to the architecture.")
    elif r2_dyn > 0.3:
        print(f"R²_dyn={r2_dyn:.2f} is moderate — dynamics loss should help")
        print("but expect diminishing returns at high noise.")
    else:
        print(f"R²_dyn={r2_dyn:.2f} is low — linear dynamics is weak. Either")
        print("the obs space is wrong (try adding velocities) or dynamics is")
        print("highly nonlinear and a small MLP head won't capture it.")

    if r2_sa > 0.5:
        print(f"R²(s->a)={r2_sa:.2f} is high — policy is reactive. Joint")
        print("denoising via cross-attention is the right structural choice.")
    else:
        print(f"R²(s->a)={r2_sa:.2f} is modest — action carries info not in")
        print("the current obs. Cross-attention still wins but anchors with")
        print("temporal context (gripper_history, phase) matter more.")


if __name__ == "__main__":
    main()