import pytest
import torch
import torch.nn as nn
import numpy as np
from types import SimpleNamespace
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.Autoformer_Fixed import Model as AutoformerFixed


class TestRobustness:
    """Test model robustness to various data conditions"""
    
    @pytest.fixture
    def base_config(self):
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
        return config
    
    def test_missing_values_handling(self, base_config):
        """Test model robustness to missing values"""
        model = AutoformerFixed(base_config)
        model.eval()
        
        batch_size = 4
        seq_len = base_config.seq_len
        
        # Create clean data
        x_enc_clean = torch.randn(batch_size, seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        
        # Create data with missing values (NaNs)
        x_enc_missing = x_enc_clean.clone()
        
        # Add missing values at random positions (10% of values)
        mask = torch.rand_like(x_enc_missing) < 0.1
        x_enc_missing[mask] = float('nan')
        
        # Replace NaNs with zeros (common strategy)
        x_enc_missing_filled = torch.nan_to_num(x_enc_missing, nan=0.0)
        
        # Forward pass with clean data
        with torch.no_grad():
            output_clean = model(x_enc_clean, x_mark_enc, x_dec, x_mark_dec)
        
        # Forward pass with missing values filled
        with torch.no_grad():
            output_missing = model(x_enc_missing_filled, x_mark_enc, x_dec, x_mark_dec)
        
        # Model should produce valid outputs for both cases
        assert not torch.isnan(output_clean).any()
        assert not torch.isnan(output_missing).any()
        
        # Outputs should be different but not drastically
        output_diff = torch.abs(output_clean - output_missing).mean()
        assert output_diff > 0, "Outputs should be different"
        assert output_diff < output_clean.abs().mean(), "Difference should be reasonable"
    
    def test_outlier_robustness(self, base_config):
        """Test model robustness to outliers"""
        model = AutoformerFixed(base_config)
        model.eval()
        
        batch_size = 4
        seq_len = base_config.seq_len
        
        # Create clean data
        x_enc_clean = torch.randn(batch_size, seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        
        # Create data with outliers
        x_enc_outliers = x_enc_clean.clone()
        
        # Add outliers at random positions (5% of values)
        mask = torch.rand_like(x_enc_outliers) < 0.05
        x_enc_outliers[mask] = x_enc_outliers[mask] * 10  # 10x multiplier for outliers
        
        # Forward pass with clean data
        with torch.no_grad():
            output_clean = model(x_enc_clean, x_mark_enc, x_dec, x_mark_dec)
        
        # Forward pass with outliers
        with torch.no_grad():
            output_outliers = model(x_enc_outliers, x_mark_enc, x_dec, x_mark_dec)
        
        # Model should produce valid outputs for both cases
        assert not torch.isnan(output_clean).any()
        assert not torch.isnan(output_outliers).any()
        
        # Outputs should be different but not drastically
        output_diff = torch.abs(output_clean - output_outliers).mean()
        assert output_diff > 0, "Outputs should be different"
        assert output_diff < output_clean.abs().mean() * 2, "Difference should be reasonable"
    
    def test_different_distributions(self, base_config):
        """Test model robustness to different data distributions"""
        model = AutoformerFixed(base_config)
        model.eval()
        
        batch_size = 4
        seq_len = base_config.seq_len
        
        # Create data from different distributions
        distributions = [
            torch.randn,  # Normal distribution
            lambda *args: torch.rand(*args) * 2 - 1,  # Uniform [-1, 1]
            lambda *args: torch.exp(torch.randn(*args)),  # Log-normal
            lambda *args: torch.pow(torch.rand(*args), 0.5)  # Square root of uniform
        ]
        
        outputs = []
        
        for dist_func in distributions:
            # Create data from this distribution
            x_enc = dist_func(batch_size, seq_len, base_config.enc_in)
            x_mark_enc = torch.randn(batch_size, seq_len, 4)
            x_dec = dist_func(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
            x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
            
            # Forward pass
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Model should produce valid outputs
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            
            outputs.append(output)
        
        # Outputs should be different for different distributions
        for i in range(len(outputs)):
            for j in range(i+1, len(outputs)):
                assert not torch.allclose(outputs[i], outputs[j], atol=1e-3)
    
    def test_noise_robustness(self, base_config):
        """Test model robustness to different noise levels"""
        model = AutoformerFixed(base_config)
        model.eval()
        
        batch_size = 4
        seq_len = base_config.seq_len
        
        # Create base data
        x_enc_base = torch.randn(batch_size, seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        
        # Test with different noise levels
        noise_levels = [0.01, 0.05, 0.1, 0.2, 0.5]
        outputs = []
        
        for noise in noise_levels:
            # Add noise to base data
            noise_tensor = torch.randn_like(x_enc_base) * noise
            x_enc_noisy = x_enc_base + noise_tensor
            
            # Forward pass
            with torch.no_grad():
                output = model(x_enc_noisy, x_mark_enc, x_dec, x_mark_dec)
            
            # Model should produce valid outputs
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            
            outputs.append(output)
        
        # Output should change gradually with noise level
        for i in range(len(outputs) - 1):
            diff_curr = torch.abs(outputs[i] - outputs[i+1]).mean()
            assert diff_curr > 0, "Outputs should be different for different noise levels"
            
            # Higher noise should generally cause larger differences
            if i < len(outputs) - 2:
                diff_next = torch.abs(outputs[i+1] - outputs[i+2]).mean()
                # This might not always hold, so we use a soft assertion
                if diff_next < diff_curr:
                    print(f"Warning: Difference decreased from noise {noise_levels[i+1]} to {noise_levels[i+2]}")
    
    def test_sequence_length_robustness(self, base_config):
        """Test model robustness to different sequence lengths"""
        model = AutoformerFixed(base_config)
        model.eval()
        
        batch_size = 4
        
        # Test with different sequence lengths
        seq_lengths = [48, 96, 192]
        
        for seq_len in seq_lengths:
            # Update config
            base_config.seq_len = seq_len
            
            # Create inputs
            x_enc = torch.randn(batch_size, seq_len, base_config.enc_in)
            x_mark_enc = torch.randn(batch_size, seq_len, 4)
            x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
            x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
            
            # Forward pass
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Model should produce valid outputs
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            assert output.shape == (batch_size, base_config.pred_len, base_config.c_out)
    
    def test_feature_dimension_robustness(self, base_config):
        """Test model robustness to different feature dimensions"""
        feature_dims = [1, 3, 7, 12]
        
        for dim in feature_dims:
            # Update config
            config = base_config
            config.enc_in = dim
            config.dec_in = dim
            config.c_out = dim
            
            # Create model
            model = AutoformerFixed(config)
            model.eval()
            
            # Create inputs
            batch_size = 4
            seq_len = config.seq_len
            x_enc = torch.randn(batch_size, seq_len, dim)
            x_mark_enc = torch.randn(batch_size, seq_len, 4)
            x_dec = torch.randn(batch_size, config.label_len + config.pred_len, dim)
            x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
            
            # Forward pass
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Model should produce valid outputs
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            assert output.shape == (batch_size, config.pred_len, dim)
    
    def test_numerical_stability(self, base_config):
        """Test model numerical stability with extreme values"""
        model = AutoformerFixed(base_config)
        model.eval()
        
        batch_size = 4
        seq_len = base_config.seq_len
        
        # Test with extreme scales
        scales = [1e-6, 1e-3, 1.0, 1e3, 1e6]
        
        for scale in scales:
            # Create scaled inputs
            x_enc = torch.randn(batch_size, seq_len, base_config.enc_in) * scale
            x_mark_enc = torch.randn(batch_size, seq_len, 4)  # Time marks not scaled
            x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in) * scale
            x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
            
            # Forward pass
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Model should produce valid outputs
            assert not torch.isnan(output).any(), f"NaN output with scale {scale}"
            assert not torch.isinf(output).any(), f"Inf output with scale {scale}"
            
            # Output scale should be related to input scale
            output_scale = output.abs().mean().item()
            assert output_scale > 0, f"Zero output with scale {scale}"
            
            # Very rough check that output scale is reasonable
            if scale >= 1e-3 and scale <= 1e3:
                assert 1e-6 < output_scale < 1e6, f"Unreasonable output scale: {output_scale} with input scale {scale}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])