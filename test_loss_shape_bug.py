#!/usr/bin/env python3
"""
Test to reproduce the batch_size > 1 shape bug
"""

import torch
import sys

# Test the exact scenario that's failing with actual decoder
print("=" * 60)
print("TESTING WITH ACTUAL DECODER")
print("=" * 60)

# First test: check if the issue is in the decoder get_point_prediction
from layers.modular.decoder.sequential_mixture_decoder import SequentialMixtureDensityDecoder

# Create a simple config for testing
class SimpleConfig:
    def __init__(self):
        self.d_model = 780
        self.num_targets = 4
        self.mdn_components = 5
        self.pred_len = 10
        self.seq_len = 50
        
config = SimpleConfig()

try:
    decoder = SequentialMixtureDensityDecoder(
        d_model=config.d_model,
        num_targets=config.num_targets,
        num_components=config.mdn_components,
        nhead=8,
        num_decoder_layers=2,
        dropout=0.1
    )
    print("✓ Decoder created")
    
    # Test with batch_size=2
    batch_size = 2
    encoder_output = torch.randn(batch_size, config.seq_len, config.d_model)
    decoder_input = torch.randn(batch_size, config.pred_len, config.d_model)
    
    means, log_stds, log_weights = decoder(encoder_output, decoder_input)
    print(f"\nDecoder outputs with batch_size={batch_size}:")
    print(f"  means: {means.shape}")
    print(f"  log_stds: {log_stds.shape}")
    print(f"  log_weights: {log_weights.shape}")
    
    # Now test get_point_prediction
    point_pred = decoder.get_point_prediction((means, log_stds, log_weights))
    print(f"  point_pred: {point_pred.shape}")
    print("✅ get_point_prediction works with batch_size=2")
    
except Exception as e:
    print(f"❌ Decoder test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("TESTING LOSS FUNCTION WITH DECODER OUTPUTS")
print("=" * 60)

batch_size = 2
pred_len = 10
c_out = 4
num_components = 5

# BUT: What if log_weights has wrong shape?
# Let's test both shapes
mu = torch.randn(batch_size, pred_len, c_out, num_components)
log_stds = torch.log(torch.abs(torch.randn(batch_size, pred_len, c_out, num_components)) + 0.1)

# CASE 1: log_weights has shape [batch, seq_len, num_components] (correct for sequential decoder)
log_weights_3d = torch.randn(batch_size, pred_len, num_components)

print(f"\nCase 1: log_weights 3D (correct):")
print(f"  mu: {mu.shape}")
print(f"  log_stds: {log_stds.shape}")
print(f"  log_weights: {log_weights_3d.shape}")


try:
    from layers.modular.losses.directional_trend_loss import HybridMDNDirectionalLoss
    
    loss_fn = HybridMDNDirectionalLoss(
        nll_weight=0.3,
        direction_weight=5.0,
        trend_weight=0.8,
        magnitude_weight=0.2,
    )
    
    targets = torch.randn(batch_size, pred_len, c_out)
    
    mdn_params_3d = (mu, log_stds, log_weights_3d)
    loss_3d = loss_fn(mdn_params_3d, targets)
    print(f"✅ Loss with 3D log_weights: {loss_3d.item():.6f}")
    
except Exception as e:
    print(f"❌ FAILURE with 3D log_weights: {e}")
    import traceback
    traceback.print_exc()

# CASE 2: log_weights has shape [batch, seq_len, c_out, num_components] (if decoder has a bug)
log_weights_4d = torch.randn(batch_size, pred_len, c_out, num_components)

print(f"\nCase 2: log_weights 4D (if there's a bug):")
print(f"  mu: {mu.shape}")
print(f"  log_stds: {log_stds.shape}")
print(f"  log_weights: {log_weights_4d.shape}")

try:
    mdn_params_4d = (mu, log_stds, log_weights_4d)
    loss_4d = loss_fn(mdn_params_4d, targets)
    print(f"✅ Loss with 4D log_weights: {loss_4d.item():.6f}")
except Exception as e:
    print(f"❌ FAILURE with 4D log_weights: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("🎉 TEST COMPLETE")
print("=" * 60)
