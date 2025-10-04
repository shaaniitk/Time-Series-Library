#!/usr/bin/env python3
"""
Find exactly where the validation error is happening
"""

import sys
import os
import torch
import traceback

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def find_validation_error():
    """Find exactly where the validation error occurs."""
    
    try:
        from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT
        
        class TestConfig:
            def __init__(self):
                self.seq_len = 96
                self.pred_len = 24
                self.enc_in = 7
                self.c_out = 3
                self.d_model = 512
                self.n_heads = 8
                self.dropout = 0.1
                self.use_mixture_density = True
                self.autocorr_factor = 1
                self.max_eigenvectors = 16
        
        config = TestConfig()
        model = SOTA_Temporal_PGAT(config, mode='probabilistic')
        
        # Create test inputs
        batch_size = 2
        wave_window = torch.randn(batch_size, config.seq_len, config.enc_in)
        target_window = torch.randn(batch_size, config.pred_len, config.enc_in)
        
        print("🔍 Running forward pass to catch exact error location...")
        
        try:
            output = model(wave_window, target_window, graph=None)
        except Exception as e:
            print(f"\n❌ Error caught: {e}")
            print(f"\n📍 Full traceback:")
            traceback.print_exc()
            
            # The error message tells us exactly what's wrong
            error_str = str(e)
            if "wave_features expected batch=2, nodes=7; received (2, 96, 512)" in error_str:
                print(f"\n💡 Analysis:")
                print(f"   - The error is in a validation that expects wave_features to have shape [batch=2, nodes=7, d_model]")
                print(f"   - But it's receiving shape [batch=2, seq_len=96, d_model=512]")
                print(f"   - This means somewhere in the code, the old temporal tensor is being passed")
                print(f"   - instead of the converted spatial tensor")
                
                print(f"\n🔧 Solution:")
                print(f"   - Find all places where wave_features, target_features are used")
                print(f"   - Make sure they use the converted spatial tensors")
                print(f"   - The validation function name should be in the traceback")
        
        return True
        
    except Exception as e:
        print(f"❌ Setup failed: {e}")
        return False

if __name__ == "__main__":
    find_validation_error()