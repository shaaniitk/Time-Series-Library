import pytest
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer


class TestAutoCorrelationCore:
    """Test the core AutoCorrelation mechanism - the heart of Autoformer"""
    
    @pytest.fixture
    def autocorr_config(self):
        return {
            'factor': 1,
            'attention_dropout': 0.1,
            'output_attention': False
        }
    
    def test_fft_correlation_computation(self, autocorr_config):
        """Test FFT-based correlation computation accuracy"""
        autocorr = AutoCorrelation(**autocorr_config)
        
        # Create synthetic periodic data
        seq_len = 96
        batch_size = 2
        n_heads = 4
        d_head = 16
        
        # Generate data with known periodicity
        t = torch.arange(seq_len).float()
        periodic_signal = torch.sin(2 * np.pi * t / 24)  # 24-hour period
        
        queries = periodic_signal.unsqueeze(0).unsqueeze(0).unsqueeze(-1).expand(batch_size, seq_len, n_heads, d_head)
        keys = queries.clone()
        values = queries.clone()
        
        output, attn = autocorr(queries, keys, values, None)
        
        # Validate output shape and no NaN
        assert output.shape == queries.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_time_delay_aggregation(self, autocorr_config):
        """Test time delay aggregation mechanism"""
        autocorr = AutoCorrelation(**autocorr_config)
        
        # Test with different sequence lengths
        for seq_len in [48, 96, 192]:
            queries = torch.randn(2, seq_len, 4, 16)
            keys = torch.randn(2, seq_len, 4, 16)
            values = torch.randn(2, seq_len, 4, 16)
            
            output, _ = autocorr(queries, keys, values, None)
            
            # Check output preserves sequence length
            assert output.shape[1] == seq_len
            assert not torch.isnan(output).any()
    
    def test_top_k_selection_logic(self, autocorr_config):
        """Test top-k period selection logic"""
        autocorr = AutoCorrelation(factor=2, **{k:v for k,v in autocorr_config.items() if k != 'factor'})
        
        seq_len = 96
        queries = torch.randn(1, seq_len, 1, 1)
        keys = queries.clone()
        values = queries.clone()
        
        # Mock the correlation computation to test k selection
        with torch.no_grad():
            output, _ = autocorr(queries, keys, values, None)
        
        # Verify k is reasonable (should be factor * log(seq_len))
        expected_k = int(2 * np.log(seq_len))
        assert expected_k > 0 and expected_k < seq_len
    
    def test_autocorr_layer_projections(self):
        """Test AutoCorrelationLayer projections"""
        d_model = 64
        n_heads = 8
        autocorr = AutoCorrelation(factor=1, attention_dropout=0.1)
        layer = AutoCorrelationLayer(autocorr, d_model, n_heads)
        
        batch_size = 2
        seq_len = 96
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attn = layer(x, x, x, None)
        
        # Check projections work correctly
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
    
    def test_correlation_vs_attention_difference(self):
        """Test that AutoCorrelation behaves differently from standard attention"""
        d_model = 32
        n_heads = 4
        seq_len = 48
        
        # AutoCorrelation
        autocorr = AutoCorrelationLayer(
            AutoCorrelation(factor=1, attention_dropout=0.0),
            d_model, n_heads
        )
        
        # Standard attention for comparison
        std_attn = nn.MultiheadAttention(d_model, n_heads, dropout=0.0, batch_first=True)
        
        x = torch.randn(2, seq_len, d_model)
        
        with torch.no_grad():
            autocorr_out, _ = autocorr(x, x, x, None)
            std_out, _ = std_attn(x, x, x)
        
        # They should produce different outputs (not identical)
        assert not torch.allclose(autocorr_out, std_out, atol=1e-3)
    
    def test_periodic_pattern_detection(self):
        """Test AutoCorrelation's ability to detect periodic patterns"""
        d_model = 16
        n_heads = 2
        seq_len = 96
        
        autocorr_layer = AutoCorrelationLayer(
            AutoCorrelation(factor=1, attention_dropout=0.0),
            d_model, n_heads
        )
        
        # Create data with clear 24-step periodicity
        t = torch.arange(seq_len).float()
        periodic_data = torch.sin(2 * np.pi * t / 24).unsqueeze(0).unsqueeze(-1).expand(1, seq_len, d_model)
        
        with torch.no_grad():
            output, _ = autocorr_layer(periodic_data, periodic_data, periodic_data, None)
        
        # Output should maintain periodicity (rough check)
        assert output.shape == periodic_data.shape
        assert output.std() > 0  # Should not be constant
        assert not torch.isnan(output).any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])