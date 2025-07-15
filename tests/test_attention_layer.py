import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.Attention import get_attention_layer


class TestAttentionLayer:
    """Test attention layer selection and functionality"""
    
    @pytest.fixture
    def base_config(self):
        config = SimpleNamespace()
        config.factor = 1
        config.dropout = 0.1
        config.d_model = 64
        config.n_heads = 8
        return config
    
    def test_default_attention_selection(self, base_config):
        """Test default attention type selection"""
        # No attention_type specified - should default to standard AutoCorrelation
        attention_layer = get_attention_layer(base_config)
        
        assert attention_layer is not None
        assert hasattr(attention_layer, 'inner_correlation')
        assert hasattr(attention_layer, 'query_projection')
        assert hasattr(attention_layer, 'key_projection')
        assert hasattr(attention_layer, 'value_projection')
        assert hasattr(attention_layer, 'out_projection')
    
    def test_standard_autocorrelation_selection(self, base_config):
        """Test explicit standard AutoCorrelation selection"""
        base_config.attention_type = 'autocorrelation'
        attention_layer = get_attention_layer(base_config)
        
        # Should be standard AutoCorrelationLayer
        assert attention_layer.__class__.__name__ == 'AutoCorrelationLayer'
        assert attention_layer.inner_correlation.__class__.__name__ == 'AutoCorrelation'
    
    def test_adaptive_autocorrelation_selection(self, base_config):
        """Test adaptive AutoCorrelation selection"""
        base_config.attention_type = 'adaptive_autocorrelation'
        attention_layer = get_attention_layer(base_config)
        
        # Should be enhanced AdaptiveAutoCorrelationLayer
        assert attention_layer.__class__.__name__ == 'AdaptiveAutoCorrelationLayer'
        assert attention_layer.inner_correlation.__class__.__name__ == 'AdaptiveAutoCorrelation'
        
        # Check enhanced features are enabled
        assert attention_layer.inner_correlation.adaptive_k == True
        assert attention_layer.inner_correlation.multi_scale == True
        assert attention_layer.inner_correlation.scales == [1, 2, 4]
    
    def test_attention_layer_forward_pass(self, base_config):
        """Test forward pass through attention layers"""
        batch_size = 2
        seq_len = 48
        
        for attention_type in [None, 'autocorrelation', 'adaptive_autocorrelation']:
            if attention_type:
                base_config.attention_type = attention_type
            elif hasattr(base_config, 'attention_type'):
                delattr(base_config, 'attention_type')
            
            attention_layer = get_attention_layer(base_config)
            
            # Test forward pass
            x = torch.randn(batch_size, seq_len, base_config.d_model)
            output, attn = attention_layer(x, x, x, None)
            
            # Check output properties
            assert output.shape == (batch_size, seq_len, base_config.d_model)
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
    
    def test_attention_layer_gradient_flow(self, base_config):
        """Test gradient flow through attention layers"""
        batch_size = 2
        seq_len = 24
        
        for attention_type in ['autocorrelation', 'adaptive_autocorrelation']:
            base_config.attention_type = attention_type
            attention_layer = get_attention_layer(base_config)
            
            x = torch.randn(batch_size, seq_len, base_config.d_model, requires_grad=True)
            output, _ = attention_layer(x, x, x, None)
            
            loss = output.mean()
            loss.backward()
            
            # Check gradients exist
            assert x.grad is not None
            assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
            
            # Check layer parameters have gradients
            for param in attention_layer.parameters():
                if param.requires_grad:
                    assert param.grad is not None
    
    def test_attention_layer_parameter_counts(self, base_config):
        """Test parameter counts for different attention types"""
        standard_layer = get_attention_layer(base_config)
        
        base_config.attention_type = 'adaptive_autocorrelation'
        adaptive_layer = get_attention_layer(base_config)
        
        # Count parameters
        std_params = sum(p.numel() for p in standard_layer.parameters())
        adp_params = sum(p.numel() for p in adaptive_layer.parameters())
        
        # Adaptive should have more parameters due to enhanced features
        assert adp_params > std_params
        
        print(f"Standard params: {std_params}, Adaptive params: {adp_params}")
    
    def test_attention_layer_different_configs(self, base_config):
        """Test attention layers with different configurations"""
        configs = [
            {'d_model': 32, 'n_heads': 4},
            {'d_model': 128, 'n_heads': 16},
            {'d_model': 256, 'n_heads': 8}
        ]
        
        for config_update in configs:
            for key, value in config_update.items():
                setattr(base_config, key, value)
            
            # Test both attention types
            for attention_type in ['autocorrelation', 'adaptive_autocorrelation']:
                base_config.attention_type = attention_type
                attention_layer = get_attention_layer(base_config)
                
                # Quick forward pass test
                x = torch.randn(1, 24, base_config.d_model)
                output, _ = attention_layer(x, x, x, None)
                
                assert output.shape == x.shape
                assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])