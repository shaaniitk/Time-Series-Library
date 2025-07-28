#!/usr/bin/env python3
"""
Test script to validate the Enhanced Autoformer migration.
"""
import torch
import sys
import os

# Add the current directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class MockConfig:
    """Mock configuration for testing."""
    def __init__(self):
        self.task_name = 'long_term_forecast'
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 96
        self.enc_in = 7
        self.dec_in = 7
        self.c_out = 7
        self.c_out_evaluation = 7
        self.d_model = 512
        self.e_layers = 2
        self.d_layers = 1
        self.d_ff = 2048
        self.dropout = 0.1
        self.activation = 'gelu'
        self.embed = 'timeF'
        self.freq = 'h'
        self.norm_type = 'LayerNorm'
        self.factor = 1
        self.top_k = 5
        self.n_heads = 8
        self.moving_avg = 25
        self.autocorr = True

def test_enhanced_autoformer():
    """Test the Enhanced Autoformer migration."""
    print("Testing Enhanced Autoformer migration...")
    
    try:
        # Import the enhanced model
        from models.EnhancedAutoformer import EnhancedAutoformer, LearnableSeriesDecomp, StableSeriesDecomp
        
        # Test backward compatibility
        assert StableSeriesDecomp == LearnableSeriesDecomp, "Backward compatibility alias failed"
        print("✅ Backward compatibility alias working")
        
        # Create mock configuration
        configs = MockConfig()
        
        # Test model creation
        model = EnhancedAutoformer(configs)
        print("✅ Model creation successful")
        
        # Test LearnableSeriesDecomp
        decomp = LearnableSeriesDecomp(input_dim=7)
        test_input = torch.randn(32, 96, 7)
        seasonal, trend = decomp(test_input)
        
        assert seasonal.shape == test_input.shape, f"Seasonal shape mismatch: {seasonal.shape} != {test_input.shape}"
        assert trend.shape == test_input.shape, f"Trend shape mismatch: {trend.shape} != {test_input.shape}"
        print("✅ LearnableSeriesDecomp working correctly")
        
        # Test model forward pass
        batch_size = 4
        x_enc = torch.randn(batch_size, configs.seq_len, configs.enc_in)
        x_mark_enc = torch.randn(batch_size, configs.seq_len, 4)
        x_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, configs.dec_in)
        x_mark_dec = torch.randn(batch_size, configs.label_len + configs.pred_len, 4)
        
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
        expected_shape = (batch_size, configs.pred_len, configs.c_out)
        assert output.shape == expected_shape, f"Output shape mismatch: {output.shape} != {expected_shape}"
        print("✅ Model forward pass working correctly")
        
        print("🎉 All tests passed! Migration successful!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_enhanced_autoformer()
    sys.exit(0 if success else 1)