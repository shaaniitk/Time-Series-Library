#!/usr/bin/env python3
"""
Test to reproduce the batch_size > 1 shape bug
"""

import torch
import sys

# Test the exact scenario that's failing
print("=" * 60)
print("REPRODUCING BATCH SIZE > 1 SHAPE BUG")
print("=" * 60)

batch_size = 2
pred_len = 10
c_out = 4
num_components = 5

# Create MDN outputs matching the error
pi = torch.randn(batch_size, pred_len, c_out, num_components)
mu = torch.randn(batch_size, pred_len, c_out, num_components)
sigma = torch.abs(torch.randn(batch_size, pred_len, c_out, num_components)) + 0.1

print(f"\nMDN outputs shapes:")
print(f"  pi: {pi.shape}")
print(f"  mu: {mu.shape}")
print(f"  sigma: {sigma.shape}")

# Create targets
targets = torch.randn(batch_size, pred_len, c_out)
print(f"\nTargets shape: {targets.shape}")

# Import the loss
try:
    from layers.modular.losses.directional_trend_loss import HybridMDNDirectionalLoss
    
    loss_fn = HybridMDNDirectionalLoss(
        nll_weight=0.3,
        direction_weight=5.0,
        trend_weight=0.8,
        magnitude_weight=0.2,
    )
    
    print("\n✓ Loss function created")
    
    # Convert to log format as the handler does
    log_stds = torch.log(sigma.clamp(min=1e-6))
    log_weights = torch.log(pi.clamp(min=1e-8))
    mdn_params = (mu, log_stds, log_weights)
    
    print(f"\nConverted MDN params:")
    print(f"  mu: {mdn_params[0].shape}")
    print(f"  log_stds: {mdn_params[1].shape}")
    print(f"  log_weights: {mdn_params[2].shape}")
    
    # Try to compute loss
    print("\n🔍 Computing loss...")
    loss = loss_fn(mdn_params, targets)
    
    print(f"✅ SUCCESS: Loss computed = {loss.item():.6f}")
    
except Exception as e:
    print(f"\n❌ FAILURE: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("Testing with batch_size=1...")
print("=" * 60)

# Test with batch_size=1 to see if it works
batch_size = 1
pi1 = torch.randn(batch_size, pred_len, c_out, num_components)
mu1 = torch.randn(batch_size, pred_len, c_out, num_components)
sigma1 = torch.abs(torch.randn(batch_size, pred_len, c_out, num_components)) + 0.1
targets1 = torch.randn(batch_size, pred_len, c_out)

log_stds1 = torch.log(sigma1.clamp(min=1e-6))
log_weights1 = torch.log(pi1.clamp(min=1e-8))
mdn_params1 = (mu1, log_stds1, log_weights1)

try:
    loss1 = loss_fn(mdn_params1, targets1)
    print(f"✅ batch_size=1 works: Loss = {loss1.item():.6f}")
except Exception as e:
    print(f"❌ batch_size=1 ALSO fails: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🎉 TEST COMPLETE")
print("=" * 60)
