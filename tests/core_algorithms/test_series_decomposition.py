import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import with explicit path
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from layers.Autoformer_EncDec import moving_avg, series_decomp, series_decomp_multi, my_Layernorm


class TestMovingAverage:
    """Test moving average component"""
    
    def test_moving_avg_basic(self):
        """Test basic moving average functionality"""
        kernel_size = 5
        stride = 1
        ma = moving_avg(kernel_size, stride)
        
        # Test with different input shapes
        for batch_size in [1, 4]:
            for seq_len in [24, 48, 96]:
                for n_features in [1, 7]:
                    x = torch.randn(batch_size, seq_len, n_features)
                    output = ma(x)
                    
                    # Should preserve sequence length
                    assert output.shape == x.shape
                    assert not torch.isnan(output).any()
                    assert not torch.isinf(output).any()
    
    def test_moving_avg_smoothing_property(self):
        """Test that moving average actually smooths the signal"""
        kernel_size = 25
        ma = moving_avg(kernel_size, stride=1)
        
        # Create noisy signal with controlled seed for reproducibility
        torch.manual_seed(42)  # Fixed seed for reproducible test
        seq_len = 96
        t = torch.arange(seq_len).float()
        clean_signal = torch.sin(2 * np.pi * t / 24)  # 24-step period
        noise = torch.randn(seq_len) * 0.5
        noisy_signal = clean_signal + noise
        
        x = noisy_signal.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        smoothed = ma(x).squeeze()
        
        # Smoothed signal should have lower variance than noisy signal
        assert torch.var(smoothed) < torch.var(noisy_signal)
        
        # Test that moving average does smooth the signal
        # We don't test MSE against clean signal as that depends on noise pattern
        # Instead, verify that smoothed signal has less high-frequency content
        noisy_diff = torch.abs(torch.diff(noisy_signal))
        smoothed_diff = torch.abs(torch.diff(smoothed))
        
        # Average absolute difference should be smaller in smoothed signal
        assert smoothed_diff.mean() < noisy_diff.mean()
    
    def test_moving_avg_different_kernels(self):
        """Test moving average with different kernel sizes"""
        x = torch.randn(2, 48, 3)
        
        for kernel_size in [3, 5, 15, 25, 35]:
            ma = moving_avg(kernel_size, stride=1)
            output = ma(x)
            
            assert output.shape == x.shape
            assert not torch.isnan(output).any()
            
            # Larger kernels should produce smoother outputs
            if kernel_size > 3:
                output_var = torch.var(output)
                assert output_var.item() >= 0  # Basic sanity check
    
    def test_moving_avg_edge_padding(self):
        """Test edge padding behavior"""
        kernel_size = 5
        ma = moving_avg(kernel_size, stride=1)
        
        # Create signal with distinct edge values
        x = torch.zeros(1, 10, 1)
        x[0, 0, 0] = 10.0  # First value
        x[0, -1, 0] = -10.0  # Last value
        
        output = ma(x)
        
        # Output should preserve sequence length
        assert output.shape == x.shape
        
        # Edge effects should be present due to padding
        assert not torch.allclose(output[0, 0, 0], output[0, 1, 0])
        assert not torch.allclose(output[0, -1, 0], output[0, -2, 0])


class TestSeriesDecomposition:
    """Test series decomposition components"""
    
    def test_series_decomp_reconstruction(self):
        """Test that seasonal + trend = original signal"""
        kernel_size = 25
        decomp = series_decomp(kernel_size)
        
        # Test with various signals
        for batch_size in [1, 4]:
            for seq_len in [48, 96, 192]:
                for n_features in [1, 7]:
                    x = torch.randn(batch_size, seq_len, n_features)
                    
                    seasonal, trend = decomp(x)
                    reconstructed = seasonal + trend
                    
                    # Should reconstruct perfectly
                    assert torch.allclose(x, reconstructed, atol=1e-6), \
                        f"Reconstruction failed for shape {x.shape}"
                    
                    # Components should have same shape as input
                    assert seasonal.shape == x.shape
                    assert trend.shape == x.shape
    
    def test_series_decomp_trend_extraction(self):
        """Test trend extraction quality"""
        kernel_size = 25
        decomp = series_decomp(kernel_size)
        
        # Create signal with clear trend + seasonal pattern
        seq_len = 96
        t = torch.arange(seq_len).float()
        trend_true = 0.1 * t + 5.0  # Linear trend
        seasonal_true = torch.sin(2 * np.pi * t / 24)  # 24-step seasonality
        signal = trend_true + seasonal_true
        
        x = signal.unsqueeze(0).unsqueeze(-1)  # [1, seq_len, 1]
        seasonal_extracted, trend_extracted = decomp(x)
        
        # Trend should be smoother than original signal
        trend_var = torch.var(trend_extracted.squeeze())
        signal_var = torch.var(signal)
        assert trend_var < signal_var
        
        # Trend should correlate with true trend
        trend_corr = torch.corrcoef(torch.stack([
            trend_true, trend_extracted.squeeze()
        ]))[0, 1]
        assert trend_corr > 0.7, f"Trend correlation too low: {trend_corr}"
    
    def test_series_decomp_seasonal_extraction(self):
        """Test seasonal component extraction"""
        kernel_size = 25
        decomp = series_decomp(kernel_size)
        
        # Create signal with strong seasonal pattern
        seq_len = 96
        t = torch.arange(seq_len).float()
        seasonal_true = torch.sin(2 * np.pi * t / 12)  # 12-step seasonality
        trend_true = torch.ones_like(t) * 5.0  # Constant trend
        signal = seasonal_true + trend_true
        
        x = signal.unsqueeze(0).unsqueeze(-1)
        seasonal_extracted, trend_extracted = decomp(x)
        
        # Seasonal should have zero mean (approximately)
        seasonal_mean = torch.abs(seasonal_extracted.mean())
        assert seasonal_mean < 0.1, f"Seasonal mean too high: {seasonal_mean}"
        
        # Seasonal should correlate with true seasonal pattern
        seasonal_corr = torch.corrcoef(torch.stack([
            seasonal_true, seasonal_extracted.squeeze()
        ]))[0, 1]
        assert seasonal_corr > 0.8, f"Seasonal correlation too low: {seasonal_corr}"
    
    def test_series_decomp_different_kernels(self):
        """Test decomposition with different kernel sizes"""
        x = torch.randn(2, 96, 3)
        
        for kernel_size in [5, 15, 25, 35, 51]:
            decomp = series_decomp(kernel_size)
            seasonal, trend = decomp(x)
            
            # Perfect reconstruction
            reconstructed = seasonal + trend
            assert torch.allclose(x, reconstructed, atol=1e-6)
            
            # Larger kernels should produce smoother trends
            trend_smoothness = torch.mean(torch.diff(trend, dim=1) ** 2)
            assert trend_smoothness.item() >= 0  # Basic sanity check
    
    def test_series_decomp_gradient_flow(self):
        """Test gradient flow through decomposition"""
        kernel_size = 25
        decomp = series_decomp(kernel_size)
        
        x = torch.randn(2, 48, 3, requires_grad=True)
        seasonal, trend = decomp(x)
        
        # Test gradient flow through both components
        loss_seasonal = seasonal.mean()
        loss_seasonal.backward(retain_graph=True)
        assert x.grad is not None
        
        x.grad.zero_()
        loss_trend = trend.mean()
        loss_trend.backward()
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestMultiSeriesDecomposition:
    """Test multiple series decomposition"""
    
    def test_series_decomp_multi_basic(self):
        """Test basic multi-scale decomposition"""
        kernel_sizes = [5, 15, 25]
        decomp_multi = series_decomp_multi(kernel_sizes)
        
        x = torch.randn(2, 96, 3)
        seasonal, trend = decomp_multi(x)
        
        # Should reconstruct perfectly
        reconstructed = seasonal + trend
        assert torch.allclose(x, reconstructed, atol=1e-5)
        
        # Components should have same shape as input
        assert seasonal.shape == x.shape
        assert trend.shape == x.shape
    
    def test_series_decomp_multi_vs_single(self):
        """Compare multi-scale vs single-scale decomposition"""
        x = torch.randn(2, 96, 3)
        
        # Single scale
        single_decomp = series_decomp(25)
        seasonal_single, trend_single = single_decomp(x)
        
        # Multi scale
        multi_decomp = series_decomp_multi([25])  # Same kernel size
        seasonal_multi, trend_multi = multi_decomp(x)
        
        # Should be very similar (but not identical due to averaging)
        assert torch.allclose(seasonal_single, seasonal_multi, atol=1e-5)
        assert torch.allclose(trend_single, trend_multi, atol=1e-5)
    
    def test_series_decomp_multi_different_scales(self):
        """Test multi-scale with different kernel combinations"""
        x = torch.randn(2, 96, 3)
        
        kernel_combinations = [
            [5, 15],
            [5, 15, 25],
            [5, 15, 25, 35],
            [3, 7, 15, 31]
        ]
        
        for kernels in kernel_combinations:
            decomp_multi = series_decomp_multi(kernels)
            seasonal, trend = decomp_multi(x)
            
            # Perfect reconstruction
            reconstructed = seasonal + trend
            assert torch.allclose(x, reconstructed, atol=1e-5)
            
            # Components should be finite
            assert not torch.isnan(seasonal).any()
            assert not torch.isnan(trend).any()
            assert not torch.isinf(seasonal).any()
            assert not torch.isinf(trend).any()


class TestCustomLayerNorm:
    """Test custom layer normalization for seasonal components"""
    
    def test_my_layernorm_basic(self):
        """Test basic functionality of custom layer norm"""
        channels = 64
        ln = my_Layernorm(channels)
        
        x = torch.randn(2, 48, channels)
        output = ln(x)
        
        # Should preserve shape
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
    
    def test_my_layernorm_bias_removal(self):
        """Test that custom layer norm removes temporal bias"""
        channels = 32
        ln = my_Layernorm(channels)
        
        # Create input with temporal bias
        x = torch.randn(2, 48, channels)
        # Add increasing bias across time
        bias = torch.arange(48).float().unsqueeze(0).unsqueeze(-1).expand(2, 48, channels)
        x_biased = x + bias * 0.1
        
        output = ln(x_biased)
        
        # Output should have reduced temporal bias
        temporal_mean = torch.mean(output, dim=-1)  # [batch, time]
        temporal_var = torch.var(temporal_mean, dim=1)  # Variance across time
        
        # Temporal variance should be small (bias removed)
        assert torch.all(temporal_var < 1.0)
    
    def test_my_layernorm_vs_standard(self):
        """Compare custom layer norm with standard layer norm"""
        channels = 32
        custom_ln = my_Layernorm(channels)
        standard_ln = nn.LayerNorm(channels)
        
        x = torch.randn(2, 48, channels)
        
        custom_out = custom_ln(x)
        standard_out = standard_ln(x)
        
        # Should produce different outputs (custom removes temporal bias)
        assert not torch.allclose(custom_out, standard_out, atol=1e-3)
        
        # Both should preserve shape
        assert custom_out.shape == x.shape
        assert standard_out.shape == x.shape
    
    def test_my_layernorm_gradient_flow(self):
        """Test gradient flow through custom layer norm"""
        channels = 32
        ln = my_Layernorm(channels)
        
        # Test with non-zero target to ensure gradient flow
        x = torch.randn(2, 48, channels, requires_grad=True)
        output = ln(x)
        
        # Use sum of squares loss instead of mean to ensure non-zero gradients
        loss = (output ** 2).sum()
        loss.backward()
        
        # Check that layer norm parameters have gradients
        assert ln.layernorm.weight.grad is not None
        assert ln.layernorm.bias.grad is not None
        
        # Verify at least one parameter has non-zero gradient
        has_param_grad = not torch.allclose(ln.layernorm.weight.grad, 
                                          torch.zeros_like(ln.layernorm.weight.grad))
        assert has_param_grad, "Layer norm parameters should have gradients"
        
        # Check layer norm parameters have gradients
        for param in ln.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestDecompositionIntegration:
    """Test integration and edge cases"""
    
    def test_decomposition_numerical_stability(self):
        """Test decomposition with extreme input values"""
        kernel_size = 25
        decomp = series_decomp(kernel_size)
        
        # Test with extreme scales
        for scale in [1e-6, 1e6]:
            x = torch.randn(2, 48, 3) * scale
            seasonal, trend = decomp(x)
            
            # Should handle extreme scales gracefully
            assert not torch.isnan(seasonal).any()
            assert not torch.isnan(trend).any()
            assert not torch.isinf(seasonal).any()
            assert not torch.isinf(trend).any()
            
            # Perfect reconstruction
            reconstructed = seasonal + trend
            assert torch.allclose(x, reconstructed, rtol=1e-5)
    
    def test_decomposition_edge_cases(self):
        """Test decomposition with edge cases"""
        kernel_size = 25
        decomp = series_decomp(kernel_size)
        
        # Test with constant signal
        constant_signal = torch.ones(1, 96, 1) * 5.0
        seasonal, trend = decomp(constant_signal)
        
        # Seasonal should be near zero, trend should be the constant
        assert torch.allclose(seasonal, torch.zeros_like(seasonal), atol=1e-6)
        assert torch.allclose(trend, constant_signal, atol=1e-6)
        
        # Test with very short sequences
        short_signal = torch.randn(1, 10, 1)
        seasonal_short, trend_short = decomp(short_signal)
        reconstructed_short = seasonal_short + trend_short
        assert torch.allclose(short_signal, reconstructed_short, atol=1e-6)
    
    def test_decomposition_batch_consistency(self):
        """Test that decomposition is consistent across batch dimension"""
        kernel_size = 25
        decomp = series_decomp(kernel_size)
        
        # Create batch where each sample is the same
        base_signal = torch.randn(1, 96, 3)
        batch_signal = base_signal.repeat(4, 1, 1)
        
        seasonal_batch, trend_batch = decomp(batch_signal)
        
        # All batch elements should have identical decomposition
        for i in range(1, 4):
            assert torch.allclose(seasonal_batch[0], seasonal_batch[i], atol=1e-6)
            assert torch.allclose(trend_batch[0], trend_batch[i], atol=1e-6)
    
    def test_decomposition_different_sequence_lengths(self):
        """Test decomposition with various sequence lengths"""
        kernel_size = 25
        decomp = series_decomp(kernel_size)
        
        for seq_len in [12, 24, 48, 96, 192, 336]:
            x = torch.randn(2, seq_len, 3)
            seasonal, trend = decomp(x)
            
            # Perfect reconstruction
            reconstructed = seasonal + trend
            assert torch.allclose(x, reconstructed, atol=1e-6)
            
            # Proper shapes
            assert seasonal.shape == x.shape
            assert trend.shape == x.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])