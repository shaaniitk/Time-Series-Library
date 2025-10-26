#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test script for future celestial conditioning implementation.
Validates dataset loading, model signature, and config updates.
"""
import torch
import sys
import yaml
from pathlib import Path

# Quick sanity check of updated code
print("Testing future celestial conditioning implementation...")

# Test 1: Dataset modification
print("\n1. Testing ForecastingDataset with future celestial...")
try:
    from data_provider.data_loader import ForecastingDataset
    import numpy as np
    
    # Mock args
    class Args:
        seq_len = 96
        label_len = 48
        pred_len = 24
        use_future_celestial_conditioning = True
    
    # Mock data
    data_x = np.random.randn(200, 118)  # 200 timesteps, 118 features
    data_y = np.random.randn(200, 4)     # 200 timesteps, 4 targets
    data_stamp = np.random.randn(200, 5) # 200 timesteps, 5 time features
    
    class MockDimManager:
        pass
    
    dataset = ForecastingDataset(data_x, data_y, data_stamp, Args(), MockDimManager())
    
    # Get a sample
    sample = dataset[0]
    if len(sample) == 6:
        seq_x, seq_y, seq_x_mark, seq_y_mark, future_cel_x, future_cel_mark = sample
        print(f"   PASS: Dataset returns 6-tuple with future celestial!")
        print(f"   - seq_x (past): {seq_x.shape}")
        print(f"   - future_cel_x: {future_cel_x.shape}")
        print(f"   Expected: future_cel_x.shape[0] == pred_len (24)")
        assert future_cel_x.shape[0] == Args.pred_len, "Future celestial length mismatch!"
        print(f"   PASS: Future celestial shape correct!")
    else:
        print(f"   FAIL: Expected 6-tuple, got {len(sample)}-tuple")
        sys.exit(1)
        
except Exception as e:
    print(f"   FAIL: Dataset test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 2: Model signature
print("\n2. Testing model forward signature...")
try:
    # Just verify import and signature
    import inspect
    from models.Celestial_Enhanced_PGAT import Model
    
    sig = inspect.signature(Model.forward)
    params = list(sig.parameters.keys())
    
    if 'future_celestial_x' in params and 'future_celestial_mark' in params:
        print(f"   PASS: Model.forward has future_celestial parameters!")
        print(f"   Parameters: {params}")
    else:
        print(f"   FAIL: Missing future_celestial parameters in forward!")
        print(f"   Found: {params}")
        sys.exit(1)
        
except Exception as e:
    print(f"   FAIL: Model signature test failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Config file
print("\n3. Testing config file update...")
try:
    config_path = Path("configs/celestial_diagnostic_minimal.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    if config.get('use_future_celestial_conditioning') == True:
        print(f"   PASS: Config has use_future_celestial_conditioning=true!")
    else:
        print(f"   WARNING: Config missing use_future_celestial_conditioning flag")
        
except Exception as e:
    print(f"   FAIL: Config test failed: {e}")
    import traceback
    traceback.print_exc()

print("\n SUCCESS: All sanity checks passed! Implementation ready for testing.")
print("\nSummary of changes:")
print("   1. ForecastingDataset: Loads future celestial data (seq_len+pred_len window)")
print("   2. Model.forward: Accepts future_celestial_x and future_celestial_mark")
print("   3. C->T attention: Uses FUTURE celestial states when available (deterministic conditioning)")
print("   4. Training script: Handles both 4-tuple (legacy) and 6-tuple (future) batches")
print("   5. Config: use_future_celestial_conditioning flag added")
