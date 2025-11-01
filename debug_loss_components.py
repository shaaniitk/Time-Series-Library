#!/usr/bin/env python3
"""Debug loss components to understand why directional accuracy improvement doesn't reduce loss."""

import torch
import torch.nn as nn
import numpy as np

def debug_hybrid_loss_components():
    """Simulate and debug the hybrid loss components."""
    
    print("🔍 DEBUGGING HYBRID MDN DIRECTIONAL LOSS COMPONENTS")
    print("=" * 60)
    
    # Simulate some realistic values
    batch_size, seq_len, num_targets = 2, 20, 4
    num_components = 8
    
    # Create mock MDN outputs
    means = torch.randn(batch_size, seq_len, num_targets, num_components)
    log_stds = torch.randn(batch_size, seq_len, num_targets, num_components) * 0.5
    log_weights = torch.randn(batch_size, seq_len, num_targets, num_components)
    
    # Create mock targets
    target = torch.randn(batch_size, seq_len, num_targets)
    
    # Extract mixture mean (as used in directional loss)
    weights = torch.softmax(log_weights, dim=-1)
    mixture_mean = torch.sum(weights * means, dim=-1)
    
    print(f"Data shapes:")
    print(f"  mixture_mean: {mixture_mean.shape}")
    print(f"  target: {target.shape}")
    print(f"  means: {means.shape}")
    print()
    
    # Simulate MDN NLL Loss (typical values)
    # NLL loss is usually positive and can be quite large
    simulated_nll_loss = torch.tensor(3.5)  # Typical NLL value
    
    # Compute directional accuracy components manually
    pred_diff = mixture_mean[:, 1:] - mixture_mean[:, :-1]
    target_diff = target[:, 1:] - target[:, :-1]
    
    # Smooth sign matching (as in DirectionalTrendLoss)
    smooth_tanh_scale = 10.0  # Default value
    pred_sign = torch.tanh(pred_diff * smooth_tanh_scale)
    target_sign = torch.tanh(target_diff * smooth_tanh_scale)
    sign_match = pred_sign * target_sign
    directional_loss = -sign_match.mean()
    
    # Compute trend correlation loss (simplified)
    # Flatten for correlation
    pred_diff_flat = pred_diff.reshape(-1, pred_diff.shape[1])
    target_diff_flat = target_diff.reshape(-1, target_diff.shape[1])
    
    # Simple correlation approximation
    pred_centered = pred_diff_flat - pred_diff_flat.mean(dim=1, keepdim=True)
    target_centered = target_diff_flat - target_diff_flat.mean(dim=1, keepdim=True)
    
    numerator = (pred_centered * target_centered).sum(dim=1)
    pred_std = pred_centered.std(dim=1)
    target_std = target_centered.std(dim=1)
    correlation = numerator / (pred_std * target_std + 1e-8)
    trend_loss = 1.0 - correlation.mean()
    
    # Compute magnitude loss (MSE)
    magnitude_loss = nn.functional.mse_loss(mixture_mean, target)
    
    # Current config weights
    nll_weight = 0.25
    direction_weight = 4.0
    trend_weight = 2.0
    magnitude_weight = 0.15
    
    print("🧮 LOSS COMPONENT ANALYSIS:")
    print(f"  Raw NLL Loss: {simulated_nll_loss:.6f}")
    print(f"  Raw Directional Loss: {directional_loss:.6f}")
    print(f"  Raw Trend Loss: {trend_loss:.6f}")
    print(f"  Raw Magnitude Loss: {magnitude_loss:.6f}")
    print()
    
    # Weighted components
    weighted_nll = nll_weight * simulated_nll_loss
    weighted_directional = direction_weight * directional_loss
    weighted_trend = trend_weight * trend_loss
    weighted_magnitude = magnitude_weight * magnitude_loss
    
    print("⚖️  WEIGHTED COMPONENTS:")
    print(f"  Weighted NLL: {nll_weight} × {simulated_nll_loss:.6f} = {weighted_nll:.6f}")
    print(f"  Weighted Directional: {direction_weight} × {directional_loss:.6f} = {weighted_directional:.6f}")
    print(f"  Weighted Trend: {trend_weight} × {trend_loss:.6f} = {weighted_trend:.6f}")
    print(f"  Weighted Magnitude: {magnitude_weight} × {magnitude_loss:.6f} = {weighted_magnitude:.6f}")
    print()
    
    # Total losses
    directional_trend_total = weighted_directional + weighted_trend + weighted_magnitude
    hybrid_total = weighted_nll + directional_trend_total
    
    print("📊 TOTAL LOSS BREAKDOWN:")
    print(f"  Directional+Trend+Magnitude: {directional_trend_total:.6f}")
    print(f"  Hybrid Total: {hybrid_total:.6f}")
    print()
    
    # Analyze component contributions
    total_abs = abs(weighted_nll) + abs(weighted_directional) + abs(weighted_trend) + abs(weighted_magnitude)
    
    print("📈 COMPONENT CONTRIBUTIONS (%):")
    print(f"  NLL: {abs(weighted_nll)/total_abs*100:.1f}%")
    print(f"  Directional: {abs(weighted_directional)/total_abs*100:.1f}%")
    print(f"  Trend: {abs(weighted_trend)/total_abs*100:.1f}%")
    print(f"  Magnitude: {abs(weighted_magnitude)/total_abs*100:.1f}%")
    print()
    
    # Simulate improvement scenarios
    print("🎯 IMPROVEMENT SCENARIOS:")
    
    # Scenario 1: Directional accuracy improves by 20%
    improved_directional = directional_loss * 0.8  # 20% better (more negative)
    new_weighted_directional = direction_weight * improved_directional
    new_total_1 = weighted_nll + new_weighted_directional + weighted_trend + weighted_magnitude
    
    print(f"  Scenario 1 - 20% better directional accuracy:")
    print(f"    Old total: {hybrid_total:.6f}")
    print(f"    New total: {new_total_1:.6f}")
    print(f"    Change: {new_total_1 - hybrid_total:.6f}")
    
    # Scenario 2: NLL increases while directional improves
    worse_nll = simulated_nll_loss * 1.1  # 10% worse NLL
    new_weighted_nll = nll_weight * worse_nll
    new_total_2 = new_weighted_nll + new_weighted_directional + weighted_trend + weighted_magnitude
    
    print(f"  Scenario 2 - 20% better directional + 10% worse NLL:")
    print(f"    Old total: {hybrid_total:.6f}")
    print(f"    New total: {new_total_2:.6f}")
    print(f"    Change: {new_total_2 - hybrid_total:.6f}")
    
    # Recommended weight adjustments
    print()
    print("💡 RECOMMENDED WEIGHT ADJUSTMENTS:")
    
    # Option 1: Increase directional weight
    new_direction_weight = 8.0
    new_weighted_directional_opt1 = new_direction_weight * directional_loss
    new_total_opt1 = weighted_nll + new_weighted_directional_opt1 + weighted_trend + weighted_magnitude
    
    print(f"  Option 1 - Increase direction_weight to {new_direction_weight}:")
    print(f"    New directional contribution: {new_weighted_directional_opt1:.6f}")
    print(f"    New total: {new_total_opt1:.6f}")
    
    # Option 2: Reduce NLL weight
    new_nll_weight = 0.1
    new_weighted_nll_opt2 = new_nll_weight * simulated_nll_loss
    new_total_opt2 = new_weighted_nll_opt2 + weighted_directional + weighted_trend + weighted_magnitude
    
    print(f"  Option 2 - Reduce nll_weight to {new_nll_weight}:")
    print(f"    New NLL contribution: {new_weighted_nll_opt2:.6f}")
    print(f"    New total: {new_total_opt2:.6f}")

if __name__ == "__main__":
    debug_hybrid_loss_components()