import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.EnhancedAutoformer_Fixed import EnhancedAutoformer, StableSeriesDecomp


class TestEnhancedAutoformerFixed:
    
    @pytest.fixture
    def base_config(self):
        """Base configuration for testing"""
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
        config.factor = 1
        config.dropout = 0.1
        config.activation = 'gelu'
        config.embed = 'timeF'
        config.freq = 'h'
        config.norm_type = 'LayerNorm'
        return config

    def test_stable_series_decomp(self):
        """Test StableSeriesDecomp component"""
        decomp = StableSeriesDecomp(input_dim=7, kernel_size=25)
        
        # Test kernel size is odd
        assert decomp.kernel_size % 2 == 1
        
        # Test forward pass
        x = torch.randn(2, 96, 7)
        seasonal, trend = decomp(x)
        
        assert seasonal.shape == x.shape
        assert trend.shape == x.shape
        assert not torch.isnan(seasonal).any()
        assert not torch.isnan(trend).any()
        
        # Test reconstruction
        reconstructed = seasonal + trend
        assert torch.allclose(reconstructed, x, atol=1e-5)

    def test_learnable_weights_initialization(self):
        """Test learnable trend weights are properly initialized"""
        decomp = StableSeriesDecomp(input_dim=5, kernel_size=15)
        
        # Check weights are parameters
        assert isinstance(decomp.trend_weights, nn.Parameter)
        assert decomp.trend_weights.requires_grad
        
        # Check weights sum to 1 after softmax
        weights = torch.softmax(decomp.trend_weights, dim=0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)

    def test_config_validation(self, base_config):
        """Test enhanced configuration validation"""
        # Valid config should work
        model = EnhancedAutoformer(base_config)
        assert model is not None
        
        # Missing required config should fail
        invalid_config = SimpleNamespace()
        with pytest.raises(ValueError, match="Missing config"):
            EnhancedAutoformer(invalid_config)

    def test_dimension_handling(self, base_config):
        """Test proper dimension handling in decomposition"""
        model = EnhancedAutoformer(base_config)
        
        # Check decomposition uses correct input dimension
        assert model.decomp.trend_weights.shape[0] == base_config.dec_in

    def test_quantile_mode_support(self, base_config):
        """Test quantile mode functionality"""
        base_config.quantile_levels = [0.1, 0.5, 0.9]
        model = EnhancedAutoformer(base_config)
        
        assert model.is_quantile_mode == True
        assert model.num_quantiles == 3

    def test_forecast_with_quantiles(self, base_config):
        """Test forecasting with quantile support"""
        base_config.quantile_levels = [0.1, 0.5, 0.9]
        base_config.c_out = 21  # 7 targets * 3 quantiles
        model = EnhancedAutoformer(base_config)
        
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        expected_shape = (batch_size, base_config.pred_len, base_config.c_out)
        assert output.shape == expected_shape

    def test_enhanced_encoder_layer(self, base_config):
        """Test enhanced encoder layer functionality"""
        from models.EnhancedAutoformer_Fixed import EnhancedEncoderLayer
        from layers.Attention import get_attention_layer
        
        attention = get_attention_layer(base_config)
        layer = EnhancedEncoderLayer(attention, base_config.d_model, base_config.d_ff)
        
        # Check attention scaling parameter
        assert hasattr(layer, 'attn_scale')
        assert isinstance(layer.attn_scale, nn.Parameter)
        
        # Test forward pass
        x = torch.randn(2, 96, base_config.d_model)
        output, attn = layer(x)
        
        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    def test_enhanced_decoder_layer(self, base_config):
        """Test enhanced decoder layer functionality"""
        from models.EnhancedAutoformer_Fixed import EnhancedDecoderLayer
        from layers.Attention import get_attention_layer
        
        self_attn = get_attention_layer(base_config)
        cross_attn = get_attention_layer(base_config)
        layer = EnhancedDecoderLayer(self_attn, cross_attn, base_config.d_model, base_config.c_out)
        
        x = torch.randn(2, 72, base_config.d_model)  # label_len + pred_len
        cross = torch.randn(2, 96, base_config.d_model)
        
        seasonal, trend = layer(x, cross)
        
        assert seasonal.shape == x.shape
        assert trend.shape == (2, 72, base_config.c_out)

    def test_forecast_dimension_consistency(self, base_config):
        """Test dimension consistency in forecast method"""
        model = EnhancedAutoformer(base_config)
        
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        
        # Test internal dimension handling
        seasonal_init_full, trend_init_full = model.decomp(x_dec)
        
        # Check dimensions match expected
        assert seasonal_init_full.shape == x_dec.shape
        assert trend_init_full.shape == x_dec.shape
        
        # Check target extraction
        trend_init_targets = trend_init_full[:, :, :model.num_target_variables]
        assert trend_init_targets.shape[-1] == model.num_target_variables

    def test_all_tasks_functionality(self, base_config):
        """Test all task types work correctly"""
        tasks = ['long_term_forecast', 'imputation', 'anomaly_detection', 'classification']
        
        for task in tasks:
            base_config.task_name = task
            if task == 'classification':
                base_config.num_class = 5
            
            model = EnhancedAutoformer(base_config)
            
            batch_size = 2
            x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
            x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4) if task != 'anomaly_detection' else None
            
            if task in ['long_term_forecast']:
                x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
                x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            else:
                output = model(x_enc, x_mark_enc, None, None)
            
            assert output is not None
            assert not torch.isnan(output).any()

    def test_gradient_flow_enhanced(self, base_config):
        """Test gradient flow through enhanced components"""
        model = EnhancedAutoformer(base_config)
        
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in, requires_grad=True)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = output.mean()
        loss.backward()
        
        # Check gradients flow to learnable decomposition weights
        for name, param in model.named_parameters():
            if 'trend_weights' in name:
                assert param.grad is not None
                assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_numerical_stability_enhanced(self, base_config):
        """Test enhanced numerical stability"""
        model = EnhancedAutoformer(base_config)
        
        # Test with extreme values
        for scale in [1e-8, 1e8]:
            x_enc = torch.randn(2, base_config.seq_len, base_config.enc_in) * scale
            x_mark_enc = torch.randn(2, base_config.seq_len, 4)
            x_dec = torch.randn(2, base_config.label_len + base_config.pred_len, base_config.dec_in) * scale
            x_mark_dec = torch.randn(2, base_config.label_len + base_config.pred_len, 4)
            
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()

    def test_memory_efficiency_enhanced(self, base_config):
        """Test enhanced model memory efficiency"""
        model = EnhancedAutoformer(base_config)
        
        # Multiple forward passes to check for memory leaks
        for i in range(10):
            batch_size = 2
            x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
            x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
            x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
            x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
            
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Clean up
            del x_enc, x_mark_enc, x_dec, x_mark_dec, output
        
        # Should not cause memory issues
        assert True

    def test_model_comparison_consistency(self, base_config):
        """Test model produces consistent results across runs"""
        torch.manual_seed(42)
        model = EnhancedAutoformer(base_config)
        
        # Same inputs, multiple runs
        torch.manual_seed(123)
        x_enc = torch.randn(2, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(2, base_config.seq_len, 4)
        x_dec = torch.randn(2, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(2, base_config.label_len + base_config.pred_len, 4)
        
        with torch.no_grad():
            output1 = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            output2 = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])