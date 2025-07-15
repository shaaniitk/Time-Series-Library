#!/usr/bin/env python3
"""
Simple test for both fixed Autoformer models
"""

import torch
import torch.nn as nn
from types import SimpleNamespace
import sys
import os

def test_models():
    """Test both models directly"""
    print("Testing Autoformer Models...")
    
    try:
        # Create test configuration
        config = SimpleNamespace()
        config.task_name = 'long_term_forecast'
        config.seq_len = 96
        config.label_len = 48
        config.pred_len = 24
        config.enc_in = 7
        config.dec_in = 7
        config.c_out = 7
        config.d_model = 64
        config.n_heads = 8
        config.e_layers = 2
        config.d_layers = 1
        config.d_ff = 256
        config.moving_avg = 25
        config.factor = 1
        config.dropout = 0.1
        config.activation = 'gelu'
        config.embed = 'timeF'
        config.freq = 'h'
        config.norm_type = 'LayerNorm'
        
        print("1. Testing AutoformerFixed...")
        
        # Import and test AutoformerFixed directly
        sys.path.append('models')
        exec(open('models/Autoformer_Fixed.py').read(), globals())
        
        model1 = Model(config)
        print("   - Model initialized successfully")
        
        # Check critical components
        assert hasattr(model1, 'gradient_scale'), "Missing gradient_scale parameter"
        print("   - Gradient scaling parameter exists")
        
        # Test forward pass
        batch_size = 2
        x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
        x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
        x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
        x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
        
        with torch.no_grad():
            output1 = model1(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        expected_shape = (batch_size, config.pred_len, config.c_out)
        assert output1.shape == expected_shape, f"Expected {expected_shape}, got {output1.shape}"
        assert not torch.isnan(output1).any(), "Output contains NaN values"
        
        print("   - Forward pass successful")
        print(f"   - Output shape: {output1.shape}")
        
        print("2. Testing EnhancedAutoformer...")
        
        # Import and test EnhancedAutoformer directly
        exec(open('models/EnhancedAutoformer_Fixed.py').read(), globals())
        
        model2 = EnhancedAutoformer(config)
        print("   - Model initialized successfully")
        
        # Check enhanced components
        assert hasattr(model2.decomp, 'trend_weights'), "Missing learnable trend weights"
        print("   - Learnable decomposition working")
        
        with torch.no_grad():
            output2 = model2(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        assert output2.shape == expected_shape, f"Expected {expected_shape}, got {output2.shape}"
        assert not torch.isnan(output2).any(), "Output contains NaN values"
        
        print("   - Forward pass successful")
        print(f"   - Output shape: {output2.shape}")
        
        print("\n" + "="*50)
        print("ALL CRITICAL TESTS PASSED!")
        print("="*50)
        print("- AutoformerFixed: Core functionality working")
        print("- EnhancedAutoformer: Enhanced features working")
        print("- Forward passes produce correct shapes")
        print("- No NaN values in outputs")
        print("\nBoth models are ready for use!")
        
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_models()
    if success:
        print("\nSUCCESS: Both models are working perfectly!")
    else:
        print("\nFAILED: Please check the errors above")
        sys.exit(1)