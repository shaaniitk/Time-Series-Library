#!/usr/bin/env python3
"""
Debug Forward Pass Issue

Isolate and fix the remaining forward pass error.
"""

import torch
import yaml
import traceback
from models.Celestial_Enhanced_PGAT import Model

def debug_forward_pass():
    """Debug the forward pass step by step."""
    
    print("🔍 DEBUGGING FORWARD PASS")
    print("="*50)
    
    # Load config
    with open('configs/celestial_enhanced_pgat_production.yaml', 'r') as f:
        config_dict = yaml.safe_load(f)
    
    class Config:
        def __init__(self, **kwargs):
            for key, value in kwargs.items():
                setattr(self, key, value)
    
    configs = Config(**config_dict)
    
    # Initialize model
    model = Model(configs)
    model.eval()
    
    print(f"Model configuration:")
    print(f"   d_model: {model.d_model}")
    print(f"   enc_in: {model.enc_in}")
    print(f"   expected_embedding_input_dim: {model.expected_embedding_input_dim}")
    
    # Create test data with correct sequence lengths
    batch_size = 2
    seq_len = configs.seq_len  # Use config seq_len (250)
    enc_in = configs.enc_in
    
    x_enc = torch.randn(batch_size, seq_len, enc_in)
    x_mark_enc = torch.randn(batch_size, seq_len, 4)
    x_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, configs.dec_in)
    x_mark_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, 4)
    
    print(f"\nInput shapes:")
    print(f"   x_enc: {x_enc.shape}")
    print(f"   x_mark_enc: {x_mark_enc.shape}")
    
    # Test phase-aware processing
    print(f"\n🌌 Testing phase-aware processing...")
    try:
        celestial_features, adjacency_matrix, metadata = model.phase_aware_processor(x_enc)
        print(f"✅ Phase processing successful:")
        print(f"   Celestial features: {celestial_features.shape}")
        print(f"   Adjacency matrix: {adjacency_matrix.shape}")
    except Exception as e:
        print(f"❌ Phase processing failed: {e}")
        traceback.print_exc()
        return False
    
    # Test celestial projection
    print(f"\n🔄 Testing celestial projection...")
    try:
        projected_features = model.celestial_projection(celestial_features)
        print(f"✅ Celestial projection successful:")
        print(f"   Input: {celestial_features.shape}")
        print(f"   Output: {projected_features.shape}")
        print(f"   Expected: [{batch_size}, {seq_len}, {model.d_model}]")
    except Exception as e:
        print(f"❌ Celestial projection failed: {e}")
        traceback.print_exc()
        return False
    
    # Test embedding layer
    print(f"\n📊 Testing embedding layer...")
    try:
        print(f"Embedding layer expects: {model.enc_embedding.value_embedding.tokenConv.in_channels} channels")
        print(f"Projected features have: {projected_features.shape[-1]} features")
        
        embedded_features = model.enc_embedding(projected_features, x_mark_enc)
        print(f"✅ Embedding successful:")
        print(f"   Input: {projected_features.shape}")
        print(f"   Output: {embedded_features.shape}")
    except Exception as e:
        print(f"❌ Embedding failed: {e}")
        traceback.print_exc()
        return False
    
    # Test full forward pass with verbose logging
    print(f"\n🚀 Testing full forward pass...")
    try:
        with torch.no_grad():
            outputs = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if isinstance(outputs, tuple):
            predictions, metadata = outputs
        else:
            predictions = outputs
        
        print(f"✅ Full forward pass successful!")
        print(f"   Output shape: {predictions.shape}")
        return True
        
    except Exception as e:
        print(f"❌ Full forward pass failed: {e}")
        traceback.print_exc()
        
        # Try to identify where it fails
        print(f"\n🔍 Detailed error analysis:")
        error_str = str(e)
        if "index" in error_str and "out of bounds" in error_str:
            print(f"   This is an indexing error")
            print(f"   Error: {error_str}")
            
            # Check if it's in the embedding layer
            if "tokenConv" in traceback.format_exc():
                print(f"   Error is in TokenEmbedding (Conv1d layer)")
                print(f"   Conv1d expects: in_channels={model.enc_embedding.value_embedding.tokenConv.in_channels}")
                print(f"   But got input with: {projected_features.shape[-1]} features")
        
        return False

if __name__ == "__main__":
    success = debug_forward_pass()
    
    if success:
        print(f"\n🎉 FORWARD PASS DEBUG: SUCCESS!")
    else:
        print(f"\n❌ FORWARD PASS DEBUG: FAILED!")
        print(f"   Check the error details above")