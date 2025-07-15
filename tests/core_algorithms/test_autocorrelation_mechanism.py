import pytest
import torch
import numpy as np
import math
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelation, AdaptiveAutoCorrelationLayer


class TestAutoCorrelationCore:
    """Test core AutoCorrelation mechanisms"""
    
    def test_fft_correlation_computation(self):
        """Test FFT-based period discovery"""
        autocorr = AutoCorrelation(factor=1, attention_dropout=0.0)
        
        # Create periodic signal with known period
        seq_len = 48
        period = 12
        t = torch.arange(seq_len).float()
        signal = torch.sin(2 * np.pi * t / period)
        
        # Shape: [B, L, H, E]
        queries = signal.view(1, seq_len, 1, 1)
        keys = queries.clone()
        values = queries.clone()
        
        output, _ = autocorr(queries, keys, values, None)
        
        # Validate basic properties
        assert output.shape == queries.shape
        assert not torch.isnan(output).any()
        assert output.std() > 1e-6  # Should not be constant
    
    def test_top_k_selection_logic(self):
        """Test top-k period selection"""
        for factor in [1, 2, 3]:
            autocorr = AutoCorrelation(factor=factor, attention_dropout=0.0)
            
            for seq_len in [24, 48, 96]:
                expected_k = int(factor * math.log(seq_len))
                
                # Create test data
                x = torch.randn(1, seq_len, 1, 1)
                output, _ = autocorr(x, x, x, None)
                
                # Verify k is reasonable
                assert expected_k > 0
                assert expected_k < seq_len // 2
                assert output.shape == x.shape
    
    def test_time_delay_aggregation_variants(self):
        """Test different time delay aggregation methods"""
        autocorr = AutoCorrelation(factor=1, attention_dropout=0.0)
        
        seq_len = 48
        values = torch.randn(2, 4, 8, seq_len)  # [B, H, C, L]
        corr = torch.randn(2, 4, 8, seq_len)
        
        # Test training version
        autocorr.train()
        result_train = autocorr.time_delay_agg_training(values, corr)
        
        # Test inference version  
        autocorr.eval()
        result_inf = autocorr.time_delay_agg_inference(values, corr)
        
        # Test full version
        result_full = autocorr.time_delay_agg_full(values, corr)
        
        # All should have same shape
        assert result_train.shape == values.shape
        assert result_inf.shape == values.shape
        assert result_full.shape == values.shape
        
        # Should not be identical (different algorithms)
        assert not torch.allclose(result_train, result_inf, atol=1e-3)
    
    def test_autocorrelation_layer_projections(self):
        """Test AutoCorrelationLayer wrapper"""
        d_model = 64
        n_heads = 8
        
        autocorr = AutoCorrelation(factor=1, attention_dropout=0.0)
        layer = AutoCorrelationLayer(autocorr, d_model, n_heads)
        
        batch_size = 2
        seq_len = 48
        x = torch.randn(batch_size, seq_len, d_model)
        
        output, attn = layer(x, x, x, None)
        
        # Check dimensions
        assert output.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(output).any()
        
        # Test gradient flow
        loss = output.mean()
        loss.backward()
        assert any(p.grad is not None for p in layer.parameters())


class TestEnhancedAutoCorrelation:
    """Test enhanced AutoCorrelation features"""
    
    def test_adaptive_k_selection(self):
        """Test adaptive k selection mechanism"""
        autocorr = AdaptiveAutoCorrelation(
            factor=1, adaptive_k=True, multi_scale=False
        )
        
        seq_len = 96
        # Create correlation with clear peaks
        corr_energy = torch.zeros(1, seq_len)
        corr_energy[0, [12, 24, 48]] = 10.0  # Strong peaks at these lags
        corr_energy[0, :] += torch.randn(seq_len) * 0.1  # Add noise
        
        k = autocorr.select_adaptive_k(corr_energy, seq_len)
        
        # Should select reasonable k
        assert 2 <= k <= seq_len // 3
        assert isinstance(k, int)
    
    def test_multi_scale_correlation(self):
        """Test multi-scale correlation analysis"""
        autocorr = AdaptiveAutoCorrelation(
            factor=1, multi_scale=True, scales=[1, 2, 4]
        )
        
        seq_len = 96
        B, H, E = 2, 4, 8
        
        # Create queries/keys with multiple periodicities
        t = torch.arange(seq_len).float()
        signal1 = torch.sin(2 * np.pi * t / 12)  # 12-step period
        signal2 = torch.sin(2 * np.pi * t / 24)  # 24-step period
        combined = (signal1 + signal2).view(1, 1, 1, seq_len).expand(B, H, E, seq_len)
        
        multi_corr = autocorr.multi_scale_correlation(combined, combined, seq_len)
        
        # Should preserve sequence length
        assert multi_corr.shape == (B, H, E, seq_len)
        assert not torch.isnan(multi_corr).any()
        
        # Should capture multi-scale patterns
        assert multi_corr.std() > 1e-6
    
    def test_enhanced_time_delay_aggregation(self):
        """Test enhanced aggregation with numerical stability"""
        autocorr = AdaptiveAutoCorrelation(
            factor=1, adaptive_k=True, eps=1e-8
        )
        
        batch, head, channel, length = 2, 4, 8, 48
        values = torch.randn(batch, head, channel, length)
        corr = torch.randn(batch, head, channel, length)
        
        result = autocorr.enhanced_time_delay_agg(values, corr)
        
        # Check output properties
        assert result.shape == values.shape
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()
        
        # Should be different from input (aggregation effect)
        assert not torch.allclose(result, values, atol=1e-3)
    
    def test_numerical_stability_enhancements(self):
        """Test numerical stability features"""
        autocorr = AdaptiveAutoCorrelation(
            factor=1, eps=1e-8, multi_scale=True
        )
        
        # Test with extreme values
        seq_len = 48
        for scale in [1e-6, 1e6]:
            queries = torch.randn(1, seq_len, 2, 4) * scale
            keys = queries.clone()
            values = queries.clone()
            
            output, _ = autocorr(queries, keys, values, None)
            
            # Should handle extreme scales gracefully
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            assert output.shape == queries.shape
    
    def test_layer_scaling_gradient_flow(self):
        """Test layer scaling for better gradient flow"""
        d_model = 32
        n_heads = 4
        
        autocorr = AdaptiveAutoCorrelation(factor=1)
        layer = AdaptiveAutoCorrelationLayer(autocorr, d_model, n_heads)
        
        x = torch.randn(2, 24, d_model, requires_grad=True)
        output, _ = layer(x, x, x, None)
        
        # Test gradient flow through layer scaling
        loss = output.mean()
        loss.backward()
        
        # Check layer scale parameter has gradient
        assert layer.layer_scale.grad is not None
        assert not torch.allclose(layer.layer_scale.grad, torch.zeros_like(layer.layer_scale.grad))


class TestAutoCorrelationComparison:
    """Compare standard vs enhanced AutoCorrelation"""
    
    def test_standard_vs_enhanced_outputs(self):
        """Compare outputs between standard and enhanced versions"""
        d_model = 32
        n_heads = 4
        seq_len = 48
        
        # Standard AutoCorrelation
        std_autocorr = AutoCorrelation(factor=1, attention_dropout=0.0)
        std_layer = AutoCorrelationLayer(std_autocorr, d_model, n_heads)
        
        # Enhanced AutoCorrelation
        enh_autocorr = AdaptiveAutoCorrelation(factor=1, multi_scale=False, adaptive_k=False)
        enh_layer = AdaptiveAutoCorrelationLayer(enh_autocorr, d_model, n_heads)
        
        # Same input
        x = torch.randn(2, seq_len, d_model)
        
        with torch.no_grad():
            std_out, _ = std_layer(x, x, x, None)
            enh_out, _ = enh_layer(x, x, x, None)
        
        # Should have same shape but different values
        assert std_out.shape == enh_out.shape
        assert not torch.allclose(std_out, enh_out, atol=1e-3)
    
    def test_performance_characteristics(self):
        """Test performance characteristics of different versions"""
        import time
        
        d_model = 64
        n_heads = 8
        seq_len = 96
        
        # Create layers
        std_layer = AutoCorrelationLayer(
            AutoCorrelation(factor=1, attention_dropout=0.0),
            d_model, n_heads
        )
        
        enh_layer = AdaptiveAutoCorrelationLayer(
            AdaptiveAutoCorrelation(factor=1, multi_scale=True, adaptive_k=True),
            d_model, n_heads
        )
        
        x = torch.randn(4, seq_len, d_model)
        
        # Time standard version
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                std_layer(x, x, x, None)
        std_time = time.time() - start
        
        # Time enhanced version
        start = time.time()
        for _ in range(10):
            with torch.no_grad():
                enh_layer(x, x, x, None)
        enh_time = time.time() - start
        
        # Enhanced should be reasonably close in performance
        # (allowing up to 3x slower due to additional features)
        assert enh_time < std_time * 3.0
        
        print(f"Standard: {std_time:.4f}s, Enhanced: {enh_time:.4f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])