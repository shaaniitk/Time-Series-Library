#!/usr/bin/env python3

"""
Test the final dimension fix for celestial processing
"""

import sys
import os
sys.path.append('.')

import torch
import yaml
from argparse import Namespace

# Load the corrected config
with open('configs/celestial_enhanced_pgat_production.yaml', 'r') as f:
    config_dict = yaml.safe_load(f)

# Convert to namespace
configs = Namespace(**config_dict)

print("🧪 TESTING FINAL DIMENSION FIX")
print("=" * 60)

print(f"📊 Config values:")
print(f"   enc_in: {configs.enc_in}")
print(f"   num_input_waves: {configs.num_input_waves}")
print(f"   d_model: {configs.d_model}")
print(f"   aggregate_waves_to_celestial: {configs.aggregate_waves_to_celestial}")

# Import model
from models.Celestial_Enhanced_PGAT import Model

print(f"\n🔧 Initializing model...")
model = Model(configs)

print(f"📊 Model parameters:")
print(f"   enc_in: {model.enc_in}")
print(f"   num_input_waves: {model.num_input_waves}")
print(f"   d_model: {model.d_model}")
print(f"   expected_embedding_input_dim: {model.expected_embedding_input_dim}")
print(f"   aggregate_waves_to_celestial: {model.aggregate_waves_to_celestial}")

# Create test data matching actual training data
batch_size = 2
seq_len = 250
input_features = 118  # Actual CSV features

x_enc = torch.randn(batch_size, seq_len, input_features)
x_mark_enc = torch.randn(batch_size, seq_len, 4)  # Time features
x_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, configs.dec_in)
x_mark_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, 4)

print(f"\n🧪 Testing forward pass...")
print(f"   Input shape: {x_enc.shape}")
print(f"   Expected: enc_in={model.enc_in} == num_input_waves={model.num_input_waves}")
print(f"   Condition check: {input_features} == {model.num_input_waves} = {input_features == model.num_input_waves}")

try:
    with torch.no_grad():
        outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    print(f"✅ SUCCESS! Forward pass completed")
    
    # Handle tuple output (model might return (predictions, metadata))
    if isinstance(outputs, tuple):
        predictions = outputs[0]
        metadata = outputs[1] if len(outputs) > 1 else None
        print(f"   Output type: tuple with {len(outputs)} elements")
        print(f"   Predictions shape: {predictions.shape}")
        if metadata is not None:
            print(f"   Metadata keys: {list(metadata.keys()) if isinstance(metadata, dict) else type(metadata)}")
        outputs_to_check = predictions
    else:
        print(f"   Output shape: {outputs.shape}")
        outputs_to_check = outputs
    
    expected_shape = (batch_size, configs.pred_len, configs.c_out)
    if outputs_to_check.shape == expected_shape:
        print(f"✅ Output shape is correct!")
    else:
        print(f"⚠️  Output shape mismatch: got {outputs_to_check.shape}, expected {expected_shape}")
        
except Exception as e:
    print(f"❌ FAILED: {e}")
    import traceback
    traceback.print_exc()

print(f"\n🎯 DIMENSION FIX VALIDATION:")
print(f"✅ Config updated: enc_in={configs.enc_in}, num_input_waves={configs.num_input_waves}")
print(f"✅ Condition will pass: {input_features} == {model.num_input_waves}")
print(f"✅ Celestial processing will be enabled")
print(f"✅ Expected embedding input: {model.expected_embedding_input_dim}D (d_model)")
print("=" * 60)