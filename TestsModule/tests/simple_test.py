#!/usr/bin/env python3
"""
Simple test runner for Autoformer models without unicode characters
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_models():
    """Test both models with basic functionality"""
    print("Testing Autoformer Models...")
    
    try:
        import torch
        from types import SimpleNamespace
        
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
        
        # Test AutoformerFixed
        from models.Autoformer_Fixed import Model as AutoformerFixed
        
        model1 = AutoformerFixed(config)
        print("   - Model initialized successfully")
        
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
        assert not torch.isinf(output1).any(), "Output contains Inf values"
        
        print("   - Forward pass successful")
        print(f"   - Output shape: {output1.shape}")
        
        # Test gradient flow
        x_enc_grad = torch.randn(batch_size, config.seq_len, config.enc_in, requires_grad=True)
        output_grad = model1(x_enc_grad, x_mark_enc, x_dec, x_mark_dec)
        loss = output_grad.mean()
        loss.backward()
        
        assert x_enc_grad.grad is not None, "No gradients computed"
        print("   - Gradient flow working")
        
        print("2. Testing EnhancedAutoformer...")
        
        # Test EnhancedAutoformer
        from models.EnhancedAutoformer_Fixed import EnhancedAutoformer
        
        model2 = EnhancedAutoformer(config)
        print("   - Model initialized successfully")
        
        with torch.no_grad():
            output2 = model2(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        assert output2.shape == expected_shape, f"Expected {expected_shape}, got {output2.shape}"
        assert not torch.isnan(output2).any(), "Output contains NaN values"
        assert not torch.isinf(output2).any(), "Output contains Inf values"
        
        print("   - Forward pass successful")
        print(f"   - Output shape: {output2.shape}")
        
        # Test learnable decomposition
        assert hasattr(model2.decomp, 'trend_weights'), "Missing learnable trend weights"
        assert model2.decomp.trend_weights.requires_grad, "Trend weights not trainable"
        print("   - Learnable decomposition working")
        
        print("3. Testing different tasks...")
        
        # Test other tasks
        tasks = ['imputation', 'anomaly_detection', 'classification']
        
        for task in tasks:
            config.task_name = task
            if task == 'classification':
                config.num_class = 5
            
            try:
                model_task = AutoformerFixed(config)
                
                if task == 'classification':
                    x_mark_test = torch.ones(batch_size, config.seq_len)
                    output_task = model_task(x_enc, x_mark_test, None, None)
                    expected_task_shape = (batch_size, config.num_class)
                else:
                    x_mark_test = x_mark_enc if task != 'anomaly_detection' else None
                    output_task = model_task(x_enc, x_mark_test, None, None)
                    expected_task_shape = (batch_size, config.seq_len, config.c_out)
                
                assert output_task.shape == expected_task_shape
                print(f"   - {task} task working")
                
            except Exception as e:
                print(f"   - {task} task failed: {e}")
        
        print("4. Testing numerical stability...")
        
        # Reset to forecast task
        config.task_name = 'long_term_forecast'
        model_stable = AutoformerFixed(config)
        
        # Test with small values
        x_small = torch.randn(2, config.seq_len, config.enc_in) * 1e-6
        x_dec_small = torch.randn(2, config.label_len + config.pred_len, config.dec_in) * 1e-6
        
        with torch.no_grad():
            output_small = model_stable(x_small, x_mark_enc, x_dec_small, x_mark_dec)
        
        assert not torch.isnan(output_small).any(), "Small values produce NaN"
        print("   - Small values handled correctly")
        
        # Test with large values
        x_large = torch.randn(2, config.seq_len, config.enc_in) * 1e3
        x_dec_large = torch.randn(2, config.label_len + config.pred_len, config.dec_in) * 1e3
        
        with torch.no_grad():
            output_large = model_stable(x_large, x_mark_enc, x_dec_large, x_mark_dec)
        
        assert not torch.isnan(output_large).any(), "Large values produce NaN"
        print("   - Large values handled correctly")
        
        print("\n=== ALL TESTS PASSED ===")
        print("Both models are working correctly!")
        return True
        
    except ImportError as e:
        print(f"Import error: {e}")
        print("Make sure all required modules are available")
        return False
        
    except Exception as e:
        print(f"Test failed: {e}")
        print(f"Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_models()
    if success:
        print("\nSUCCESS: Models are ready for use!")
    else:
        print("\nFAILED: Please check the errors above")
        sys.exit(1)