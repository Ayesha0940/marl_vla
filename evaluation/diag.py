#!/usr/bin/env python3
"""
Diagnostic script to figure out why UNet gets 10% success rate.

Checks:
1. Are best_model.pt and model_final.pt actually different weights?
   (If ema_model.module == model, they should be identical)
2. What does the model actually predict? Is noise prediction reasonable?
3. Does the generated action sequence look sane?
4. Compare MLP vs UNet checkpoint stats
"""

import sys
import os
import numpy as np
import torch

# Add parent directory to path so we can import diffusion module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def check_weights_identity(best_path, final_path):
    """Check if best_model.pt and model_final.pt have the same weights."""
    print("=" * 60)
    print("CHECK 1: Are best_model.pt and model_final.pt identical?")
    print("=" * 60)
    
    best = torch.load(best_path, map_location="cpu", weights_only=False)
    final = torch.load(final_path, map_location="cpu", weights_only=False)
    
    best_sd = best["model_state_dict"]
    final_sd = final["model_state_dict"]
    
    print(f"  best_model.pt  — epoch: {best.get('epoch', '?')}, loss: {best.get('loss', '?')}")
    print(f"  model_final.pt — epoch: {final.get('epoch', '?')}, loss: {final.get('loss', '?')}")
    
    all_same = True
    max_diff = 0.0
    for key in best_sd:
        if key in final_sd:
            diff = (best_sd[key].float() - final_sd[key].float()).abs().max().item()
            max_diff = max(max_diff, diff)
            if diff > 1e-6:
                all_same = False
    
    if all_same:
        print(f"  RESULT: Weights are IDENTICAL (max diff: {max_diff:.2e})")
        print("  This confirms ema_model.module == model (EMA bug)")
    else:
        print(f"  RESULT: Weights DIFFER (max diff: {max_diff:.2e})")
    print()
 
 
def check_noise_prediction(ckpt_path):
    """Check if noise predictions are reasonable."""
    print("=" * 60)
    print("CHECK 2: Noise prediction sanity check")
    print("=" * 60)
    
    from diffusion.lift_policy_unet import load_unet_checkpoint
    from diffusion.lift_policy import make_beta_schedule, q_sample
    
    model, ckpt, alphas, alphas_bar = load_unet_checkpoint(ckpt_path, torch.device("cpu"))
    model.eval()
    
    action_horizon = ckpt["action_horizon"]
    action_dim = ckpt["action_dim"]
    obs_horizon = ckpt["obs_horizon"]
    obs_dim = ckpt["obs_dim"]
    
    # Create a "clean" action sequence (zeros in normalized space)
    x0 = torch.zeros(1, action_horizon, action_dim)
    obs = torch.zeros(1, obs_horizon, obs_dim)
    
    # Add known noise at various timesteps
    for t_val in [0, 10, 25, 50, 75, 99]:
        t = torch.tensor([t_val])
        eps_true = torch.randn(1, action_horizon, action_dim)
        
        # Forward diffuse
        x_t = q_sample(x0, t, eps_true, alphas_bar)
        
        # Predict noise
        with torch.no_grad():
            eps_pred = model(x_t, t, obs)
        
        # Compare
        mse = torch.nn.functional.mse_loss(eps_pred, eps_true).item()
        pred_std = eps_pred.std().item()
        pred_mean = eps_pred.mean().item()
        
        print(f"  t={t_val:3d}: pred_noise MSE={mse:.4f}, "
              f"pred_mean={pred_mean:+.4f}, pred_std={pred_std:.4f}")
    
    print()
    print("  Expected: MSE should decrease for small t (less noise to predict)")
    print("  Expected: pred_std should be ~1.0 (predicting unit Gaussian noise)")
    print("  Red flag:  pred_std << 1 or >> 1 means model isn't predicting noise properly")
    print()
 
 
def check_generated_actions(ckpt_path):
    """Generate action sequences and check if they look reasonable."""
    print("=" * 60)
    print("CHECK 3: Generated action sequence quality")
    print("=" * 60)
    
    from diffusion.lift_policy_unet import load_unet_checkpoint
    from diffusion.lift_policy import sample_action_sequence
    
    model, ckpt, alphas, alphas_bar = load_unet_checkpoint(ckpt_path, torch.device("cpu"))
    
    obs_mean = np.asarray(ckpt["obs_mean"], dtype=np.float32)
    obs_std = np.asarray(ckpt["obs_std"], dtype=np.float32)
    action_mean = np.asarray(ckpt["action_mean"], dtype=np.float32)
    action_std = np.asarray(ckpt["action_std"], dtype=np.float32)
    obs_horizon = int(ckpt["obs_horizon"])
    obs_dim = int(ckpt["obs_dim"])
    diffusion_steps = int(ckpt["diffusion_steps"])
    
    print(f"  action_mean: {action_mean}")
    print(f"  action_std:  {action_std}")
    print()
    
    # Use zero obs (represents "mean" observation in unnormalized space)
    obs_history = np.tile(obs_mean, (obs_horizon, 1))  # (obs_horizon, obs_dim)
    
    # Generate 5 action sequences
    for i in range(5):
        torch.manual_seed(i)
        actions = sample_action_sequence(
            model=model,
            obs_history=obs_history,
            obs_mean=obs_mean,
            obs_std=obs_std,
            action_mean=action_mean,
            action_std=action_std,
            alphas=alphas,
            alphas_bar=alphas_bar,
            diffusion_steps=diffusion_steps,
            device=torch.device("cpu"),
        )
        
        print(f"  Sample {i}: shape={actions.shape}, "
              f"mean={actions.mean():.4f}, std={actions.std():.4f}, "
              f"min={actions.min():.4f}, max={actions.max():.4f}")
        
        # Check temporal coherence: difference between consecutive actions
        diffs = np.diff(actions, axis=0)
        print(f"           temporal_diff: mean={np.abs(diffs).mean():.4f}, "
              f"max={np.abs(diffs).max():.4f}")
        
        # Check gripper dimension (dim 6) - should be bimodal (-1 or 1)
        gripper = actions[:, 6]
        print(f"           gripper: {gripper.round(2)}")
    
    print()
    print("  Expected: actions in reasonable range (roughly [-1, 1])")
    print("  Expected: temporal coherence (small diffs between consecutive steps)")
    print("  Expected: gripper values near -1 or +1 (open/close)")
    print()
 
 
def check_normalization_stats(unet_ckpt_path, mlp_ckpt_path=None):
    """Compare normalization stats between UNet and MLP checkpoints."""
    print("=" * 60)
    print("CHECK 4: Normalization statistics")
    print("=" * 60)
    
    unet_ckpt = torch.load(unet_ckpt_path, map_location="cpu", weights_only=False)
    
    print(f"  UNet checkpoint:")
    print(f"    obs_mean shape:  {np.asarray(unet_ckpt['obs_mean']).shape}")
    print(f"    obs_std shape:   {np.asarray(unet_ckpt['obs_std']).shape}")
    print(f"    action_mean:     {np.asarray(unet_ckpt['action_mean'])}")
    print(f"    action_std:      {np.asarray(unet_ckpt['action_std'])}")
    print(f"    obs_keys:        {unet_ckpt.get('obs_keys', 'NOT SAVED')}")
    
    if mlp_ckpt_path:
        mlp_ckpt = torch.load(mlp_ckpt_path, map_location="cpu", weights_only=False)
        print(f"\n  MLP checkpoint:")
        print(f"    obs_mean shape:  {np.asarray(mlp_ckpt['obs_mean']).shape}")
        print(f"    obs_std shape:   {np.asarray(mlp_ckpt['obs_std']).shape}")
        print(f"    action_mean:     {np.asarray(mlp_ckpt['action_mean'])}")
        print(f"    action_std:      {np.asarray(mlp_ckpt['action_std'])}")
        print(f"    obs_keys:        {mlp_ckpt.get('obs_keys', 'NOT SAVED')}")
        
        # Check if they match
        obs_mean_match = np.allclose(
            np.asarray(unet_ckpt['obs_mean']), 
            np.asarray(mlp_ckpt['obs_mean']), 
            atol=1e-4
        )
        action_mean_match = np.allclose(
            np.asarray(unet_ckpt['action_mean']),
            np.asarray(mlp_ckpt['action_mean']),
            atol=1e-4
        )
        print(f"\n  obs_mean match:    {obs_mean_match}")
        print(f"  action_mean match: {action_mean_match}")
        if not obs_mean_match or not action_mean_match:
            print("  WARNING: Normalization stats differ between UNet and MLP!")
            print("  This could cause different behavior even with same architecture.")
    print()
 
 
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--unet_best", type=str, 
                   default="checkpoints/lift_diffusion_policy/best_model.pt")
    p.add_argument("--unet_final", type=str,
                   default="checkpoints/lift_diffusion_policy/model_final.pt")
    p.add_argument("--mlp_best", type=str, default=None,
                   help="Optional: MLP best checkpoint for comparison")
    args = p.parse_args()
    
    check_weights_identity(args.unet_best, args.unet_final)
    check_noise_prediction(args.unet_best)
    check_generated_actions(args.unet_best)
    check_normalization_stats(args.unet_best, args.mlp_best)
 