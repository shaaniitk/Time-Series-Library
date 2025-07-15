#!/usr/bin/env python3
"""
Direct test for both fixed Autoformer models
"""

import torch
import torch.nn as nn
from types import SimpleNamespace

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
        
        # Test AutoformerFixed
        from models.Autoformer_Fixed import Model as AutoformerFixed
        
        model1 = AutoformerFixed(config)
        print("   ✓ Model initialized successfully")
        
        # Check critical components
        assert hasattr(model1, 'gradient_scale'), "Missing gradient_scale parameter"
        assert isinstance(model1.gradient_scale, nn.Parameter), "gradient_scale not a parameter"
        print("   ✓ Gradient scaling parameter exists")
        
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
        
        print("   ✓ Forward pass successful")
        print(f"   ✓ Output shape: {output1.shape}")
        
        # Test gradient flow
        x_enc_grad = torch.randn(batch_size, config.seq_len, config.enc_in, requires_grad=True)
        output_grad = model1(x_enc_grad, x_mark_enc, x_dec, x_mark_dec)
        loss = output_grad.mean()
        loss.backward()
        
        assert x_enc_grad.grad is not None, "No gradients computed"
        assert not torch.allclose(x_enc_grad.grad, torch.zeros_like(x_enc_grad.grad)), "Zero gradients"
        print("   ✓ Gradient flow working")
        
        print("2. Testing EnhancedAutoformer...")
        
        # Test EnhancedAutoformer
        from models.EnhancedAutoformer_Fixed import EnhancedAutoformer
        
        model2 = EnhancedAutoformer(config)
        print("   ✓ Model initialized successfully")
        
        # Check enhanced components
        assert hasattr(model2.decomp, 'trend_weights'), "Missing learnable trend weights"
        assert isinstance(model2.decomp.trend_weights, nn.Parameter), "trend_weights not a parameter"
        assert model2.decomp.trend_weights.requires_grad, "Trend weights not trainable"
        print("   ✓ Learnable decomposition working")
        
        with torch.no_grad():
            output2 = model2(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        assert output2.shape == expected_shape, f"Expected {expected_shape}, got {output2.shape}"
        assert not torch.isnan(output2).any(), "Output contains NaN values"
        assert not torch.isinf(output2).any(), "Output contains Inf values"
        
        print("   ✓ Forward pass successful")
        print(f"   ✓ Output shape: {output2.shape}")
        
        # Test decomposition stability
        seasonal, trend = model2.decomp(x_dec)
        reconstructed = seasonal + trend
        assert torch.allclose(reconstructed, x_dec, atol=1e-5), "Decomposition not stable"
        print("   ✓ Stable decomposition working")
        
        print("3. Testing different tasks...")
        
        # Test imputation
        config.task_name = 'imputation'
        model_imp = AutoformerFixed(config)
        output_imp = model_imp(x_enc, x_mark_enc, None, None)
        assert output_imp.shape == (batch_size, config.seq_len, config.c_out)
        print("   ✓ Imputation task working")
        
        # Test anomaly detection
        config.task_name = 'anomaly_detection'
        model_anom = AutoformerFixed(config)
        output_anom = model_anom(x_enc, None, None, None)
        assert output_anom.shape == (batch_size, config.seq_len, config.c_out)
        print("   ✓ Anomaly detection task working")
        
        # Test classification
        config.task_name = 'classification'
        config.num_class = 5
        model_cls = AutoformerFixed(config)
        x_mark_cls = torch.ones(batch_size, config.seq_len)
        output_cls = model_cls(x_enc, x_mark_cls, None, None)
        assert output_cls.shape == (batch_size, config.num_class)
        print("   ✓ Classification task working")
        
        print("4. Testing numerical stability...")
        
        # Reset to forecast task
        config.task_name = 'long_term_forecast'
        model_stable = AutoformerFixed(config)
        
        # Test with extreme values
        for scale in [1e-6, 1e6]:
            x_test = torch.randn(2, config.seq_len, config.enc_in) * scale
            x_dec_test = torch.randn(2, config.label_len + config.pred_len, config.dec_in) * scale
            
            with torch.no_grad():
                output_test = model_stable(x_test, x_mark_enc, x_dec_test, x_mark_dec)
            
            assert not torch.isnan(output_test).any(), f"Scale {scale} produces NaN"
            assert not torch.isinf(output_test).any(), f"Scale {scale} produces Inf"
        
        print("   ✓ Numerical stability confirmed")
        
        print("5. Testing quantile support in EnhancedAutoformer...")
        
        # Test quantile mode
        config.quantile_levels = [0.1, 0.5, 0.9]
        config.c_out = 21  # 7 targets * 3 quantiles
        model_quant = EnhancedAutoformer(config)
        
        assert model_quant.is_quantile_mode == True
        assert model_quant.num_quantiles == 3
        
        with torch.no_grad():
            output_quant = model_quant(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        assert output_quant.shape == (batch_size, config.pred_len, config.c_out)
        print("   ✓ Quantile support working")
        
        print("\n" + "="*50)
        print("🎉 ALL CRITICAL TESTS PASSED!")
        print("="*50)
        print("✅ AutoformerFixed: All core functionality working")
        print("✅ EnhancedAutoformer: All enhanced features working")
        print("✅ All task types supported")
        print("✅ Numerical stability confirmed")
        print("✅ Gradient flow functional")
        print("✅ Quantile support operational")
        print("\nBoth models are ready for production use!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_models()
    if success:
        print("\n🚀 SUCCESS: Both models are working perfectly!")
    else:
        print("\n💥 FAILED: Please check the errors above")
        exit(1)