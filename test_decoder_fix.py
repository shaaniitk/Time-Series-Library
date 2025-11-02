#!/usr/bin/env python3
"""
Minimal test to verify decoder num_components fix
WITHOUT loading the full model (to avoid OOM)
"""

import torch
import sys
import yaml
from pathlib import Path

# Load actual config file to avoid missing attributes
config_path = Path("configs/celestial_production_deep_ultimate_fixed.yaml")
with open(config_path) as f:
    config_dict = yaml.safe_load(f)

# Convert to namespace
from types import SimpleNamespace
def dict_to_namespace(d):
    """Recursively convert dict to SimpleNamespace"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: dict_to_namespace(v) for k, v in d.items()})
    elif isinstance(d, list):
        return [dict_to_namespace(item) for item in d]
    else:
        return d

config = dict_to_namespace(config_dict)

# Test 1: Verify decoder module reads config correctly
print("=" * 60)
print("TEST 1: Check decoder uses config.mdn_components")
print("=" * 60)

print(f"Config: mdn_components={config.mdn_components}")

try:
    from models.celestial_modules.decoder import DecoderModule
    
    decoder = DecoderModule(config)
    
    # Check if mixture_decoder exists and has correct num_components
    if hasattr(decoder, 'mixture_decoder'):
        # Access the actual num_components from the decoder
        num_comp = decoder.mixture_decoder.num_components
        print(f"✓ Decoder created with num_components={num_comp}")
        
        if num_comp == config.mdn_components:
            print(f"✅ SUCCESS: Decoder correctly uses config.mdn_components={config.mdn_components}")
        else:
            print(f"❌ FAILURE: Decoder has {num_comp} but config specifies {config.mdn_components}")
            sys.exit(1)
    else:
        print("❌ FAILURE: mixture_decoder not created")
        sys.exit(1)
        
except Exception as e:
    print(f"❌ FAILURE: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Verify loss handler compatibility
print("\n" + "=" * 60)
print("TEST 2: Check loss handler accepts use_mixture_decoder")
print("=" * 60)

try:
    from layers.modular.loss.loss_handler import create_loss_handler
    
    loss_config = {
        'type': 'hybrid_mdn_directional',
        'nll_weight': 0.3,
        'direction_weight': 5.0,
        'trend_weight': 0.8,
        'magnitude_weight': 0.2,
    }
    
    model_config = {
        'enable_mdn_decoder': False,
        'use_mixture_decoder': True,  # Should be accepted now
        'use_sequential_mixture_decoder': False,
        'c_out': 4,
    }
    
    print(f"Creating loss handler with use_mixture_decoder=True...")
    handler = create_loss_handler(loss_config, model_config)
    print(f"✅ SUCCESS: Loss handler created successfully")
    
except Exception as e:
    print(f"❌ FAILURE: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Quick forward pass test with minimal data
print("\n" + "=" * 60)
print("TEST 3: Minimal forward pass (memory check)")
print("=" * 60)

try:
    # Create minimal batch
    batch_size = 1  # Use 1 to minimize memory
    seq_len = 10    # Use short sequence
    
    x = torch.randn(batch_size, seq_len, config.d_model)
    
    print(f"Input shape: {x.shape}")
    print(f"Testing forward pass...")
    
    with torch.no_grad():  # No gradients = less memory
        output = decoder(x)
    
    print(f"Output shape: {output.shape}")
    
    # Check output is correct shape
    expected_shape = (batch_size, config.pred_len, config.c_out)
    if output.shape == expected_shape:
        print(f"✅ SUCCESS: Output shape correct {output.shape}")
    else:
        print(f"❌ FAILURE: Expected {expected_shape}, got {output.shape}")
        sys.exit(1)
    
except Exception as e:
    print(f"❌ FAILURE: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("🎉 ALL TESTS PASSED!")
print("=" * 60)
print("\nThe decoder fix is working correctly.")
print("Next step: Test with actual training but REDUCED batch size")
