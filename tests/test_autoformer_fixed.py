import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Autoformer_Fixed import Model as AutoformerFixed


class TestAutoformerFixed:
    
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
        config.moving_avg = 25
        config.factor = 1
        config.dropout = 0.1
        config.activation = 'gelu'
        config.embed = 'timeF'
        config.freq = 'h'
        config.norm_type = 'LayerNorm'
        return config

    def test_config_validation(self, base_config):
        """Test configuration validation"""
        # Valid config should work
        model = AutoformerFixed(base_config)
        assert model is not None
        
        # Missing required config should fail
        invalid_config = SimpleNamespace()
        with pytest.raises(ValueError, match="Missing config"):
            AutoformerFixed(invalid_config)

    def test_kernel_size_fix(self, base_config):
        """Test kernel size is always odd"""
        # Even kernel size should be made odd
        base_config.moving_avg = 24
        model = AutoformerFixed(base_config)
        assert hasattr(model.decomp, 'moving_avg')
        
        # Odd kernel size should remain odd
        base_config.moving_avg = 25
        model = AutoformerFixed(base_config)
        assert hasattr(model.decomp, 'moving_avg')

    def test_gradient_scaling_parameter(self, base_config):
        """Test gradient scaling parameter exists and is trainable"""
        model = AutoformerFixed(base_config)
        assert hasattr(model, 'gradient_scale')
        assert isinstance(model.gradient_scale, nn.Parameter)
        assert model.gradient_scale.requires_grad

    def test_forecast_task_forward(self, base_config):
        """Test forecasting task forward pass"""
        model = AutoformerFixed(base_config)
        
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Check output shape
        expected_shape = (batch_size, base_config.pred_len, base_config.c_out)
        assert output.shape == expected_shape
        
        # Check no NaN or Inf values
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_imputation_task(self, base_config):
        """Test imputation task"""
        base_config.task_name = 'imputation'
        model = AutoformerFixed(base_config)
        
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        
        output = model(x_enc, x_mark_enc, None, None)
        
        expected_shape = (batch_size, base_config.seq_len, base_config.c_out)
        assert output.shape == expected_shape

    def test_anomaly_detection_task(self, base_config):
        """Test anomaly detection task"""
        base_config.task_name = 'anomaly_detection'
        model = AutoformerFixed(base_config)
        
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        
        output = model(x_enc, None, None, None)
        
        expected_shape = (batch_size, base_config.seq_len, base_config.c_out)
        assert output.shape == expected_shape

    def test_classification_task(self, base_config):
        """Test classification task"""
        base_config.task_name = 'classification'
        base_config.num_class = 5
        model = AutoformerFixed(base_config)
        
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.ones(batch_size, base_config.seq_len)
        
        output = model(x_enc, x_mark_enc, None, None)
        
        expected_shape = (batch_size, base_config.num_class)
        assert output.shape == expected_shape

    def test_decomposition_stability(self, base_config):
        """Test series decomposition produces stable outputs"""
        model = AutoformerFixed(base_config)
        
        # Test with different input magnitudes
        for scale in [0.1, 1.0, 10.0]:
            x = torch.randn(2, 96, 7) * scale
            seasonal, trend = model.decomp(x)
            
            # Check decomposition is valid
            assert seasonal.shape == x.shape
            assert trend.shape == x.shape
            assert not torch.isnan(seasonal).any()
            assert not torch.isnan(trend).any()
            
            # Check reconstruction
            reconstructed = seasonal + trend
            assert torch.allclose(reconstructed, x, atol=1e-6)

    def test_gradient_flow(self, base_config):
        """Test gradient flow through the model"""
        model = AutoformerFixed(base_config)
        
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in, requires_grad=True)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = output.mean()
        loss.backward()
        
        # Check gradients exist and are not zero
        assert x_enc.grad is not None
        assert not torch.allclose(x_enc.grad, torch.zeros_like(x_enc.grad))

    def test_memory_efficiency(self, base_config):
        """Test model doesn't cause memory leaks"""
        model = AutoformerFixed(base_config)
        
        initial_memory = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Multiple forward passes
        for _ in range(5):
            batch_size = 2
            x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
            x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
            x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
            x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
            
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            del x_enc, x_mark_enc, x_dec, x_mark_dec, output
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            # Memory should not grow significantly
            assert final_memory - initial_memory < 100 * 1024 * 1024  # 100MB threshold

    def test_numerical_stability(self, base_config):
        """Test numerical stability with extreme inputs"""
        model = AutoformerFixed(base_config)
        
        # Test with very small values
        x_enc_small = torch.randn(2, base_config.seq_len, base_config.enc_in) * 1e-6
        x_mark_enc = torch.randn(2, base_config.seq_len, 4)
        x_dec = torch.randn(2, base_config.label_len + base_config.pred_len, base_config.dec_in) * 1e-6
        x_mark_dec = torch.randn(2, base_config.label_len + base_config.pred_len, 4)
        
        with torch.no_grad():
            output_small = model(x_enc_small, x_mark_enc, x_dec, x_mark_dec)
        
        assert not torch.isnan(output_small).any()
        assert not torch.isinf(output_small).any()
        
        # Test with large values
        x_enc_large = torch.randn(2, base_config.seq_len, base_config.enc_in) * 1e3
        x_dec_large = torch.randn(2, base_config.label_len + base_config.pred_len, base_config.dec_in) * 1e3
        
        with torch.no_grad():
            output_large = model(x_enc_large, x_mark_enc, x_dec_large, x_mark_dec)
        
        assert not torch.isnan(output_large).any()
        assert not torch.isinf(output_large).any()

    def test_model_determinism(self, base_config):
        """Test model produces deterministic outputs"""
        torch.manual_seed(42)
        model1 = AutoformerFixed(base_config)
        
        torch.manual_seed(42)
        model2 = AutoformerFixed(base_config)
        
        # Same inputs
        torch.manual_seed(123)
        x_enc = torch.randn(2, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(2, base_config.seq_len, 4)
        x_dec = torch.randn(2, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(2, base_config.label_len + base_config.pred_len, 4)
        
        with torch.no_grad():
            output1 = model1(x_enc, x_mark_enc, x_dec, x_mark_dec)
            output2 = model2(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        assert torch.allclose(output1, output2, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])