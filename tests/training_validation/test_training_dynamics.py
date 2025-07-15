import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from types import SimpleNamespace
import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.Autoformer_Fixed import Model as AutoformerFixed
from layers.Autoformer_EncDec import series_decomp


class TestTrainingDynamics:
    """Test training dynamics and convergence behavior"""
    
    @pytest.fixture
    def base_config(self):
        config = SimpleNamespace()
        config.task_name = 'long_term_forecast'
        config.seq_len = 48
        config.label_len = 24
        config.pred_len = 12
        config.enc_in = 3
        config.dec_in = 3
        config.c_out = 3
        config.d_model = 32
        config.n_heads = 4
        config.e_layers = 1
        config.d_layers = 1
        config.d_ff = 64
        config.moving_avg = 25
        config.factor = 1
        config.dropout = 0.1
        config.activation = 'gelu'
        config.embed = 'timeF'
        config.freq = 'h'
        config.norm_type = 'LayerNorm'
        return config
    
    def test_gradient_flow(self, base_config):
        """Test gradient flow through all model components"""
        model = AutoformerFixed(base_config)
        
        # Create synthetic data
        batch_size = 4
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        target = torch.randn(batch_size, base_config.pred_len, base_config.c_out)
        
        # Forward pass
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist for all parameters
        total_params = 0
        params_with_grad = 0
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                total_params += 1
                if param.grad is not None:
                    params_with_grad += 1
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                    assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
        
        # All parameters should have gradients
        assert params_with_grad == total_params, f"Only {params_with_grad}/{total_params} parameters have gradients"
    
    def test_component_gradient_magnitudes(self, base_config):
        """Test gradient magnitudes across different components"""
        model = AutoformerFixed(base_config)
        
        # Create synthetic data
        batch_size = 4
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        target = torch.randn(batch_size, base_config.pred_len, base_config.c_out)
        
        # Forward pass
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = nn.MSELoss()(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check gradient magnitudes by component
        component_grads = {
            'encoder': [],
            'decoder': [],
            'embedding': [],
            'projection': []
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                if 'encoder' in name:
                    component_grads['encoder'].append(param.grad.norm().item())
                elif 'decoder' in name:
                    component_grads['decoder'].append(param.grad.norm().item())
                elif 'embed' in name:
                    component_grads['embedding'].append(param.grad.norm().item())
                elif 'projection' in name:
                    component_grads['projection'].append(param.grad.norm().item())
        
        # Calculate average gradient norm for each component
        for component, grads in component_grads.items():
            if grads:
                avg_grad = sum(grads) / len(grads)
                assert 1e-6 < avg_grad < 1e3, f"Unusual gradient magnitude in {component}: {avg_grad}"
    
    def test_convergence_on_synthetic_data(self, base_config):
        """Test model convergence on synthetic data"""
        model = AutoformerFixed(base_config)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Create synthetic data with clear patterns
        batch_size = 4
        seq_len = base_config.seq_len
        pred_len = base_config.pred_len
        
        # Generate synthetic time series with trend and seasonality
        t_enc = torch.arange(0, seq_len).float() / seq_len
        t_dec = torch.arange(seq_len - base_config.label_len, seq_len + pred_len).float() / seq_len
        
        # Create features with different patterns
        features = []
        for i in range(base_config.enc_in):
            # Mix of trend and seasonal patterns
            trend = 0.1 * i * t_enc
            seasonal = 0.5 * torch.sin(2 * np.pi * (i + 1) * t_enc)
            feature = trend + seasonal + torch.randn_like(t_enc) * 0.1  # Add small noise
            features.append(feature)
        
        x_enc = torch.stack(features, dim=1).unsqueeze(0).repeat(batch_size, 1, 1)
        x_enc = x_enc.transpose(1, 2)  # [batch, seq_len, features]
        
        # Create decoder input (shifted target)
        x_dec_features = []
        for i in range(base_config.dec_in):
            trend_dec = 0.1 * i * t_dec
            seasonal_dec = 0.5 * torch.sin(2 * np.pi * (i + 1) * t_dec)
            feature_dec = trend_dec + seasonal_dec + torch.randn_like(t_dec) * 0.1
            x_dec_features.append(feature_dec)
        
        x_dec = torch.stack(x_dec_features, dim=1).unsqueeze(0).repeat(batch_size, 1, 1)
        x_dec = x_dec.transpose(1, 2)  # [batch, label_len+pred_len, features]
        
        # Target is the last pred_len steps of decoder input
        target = x_dec[:, -pred_len:, :]
        
        # Simple time features
        x_mark_enc = torch.randn(batch_size, seq_len, 4)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + pred_len, 4)
        
        # Train for several epochs
        losses = []
        model.train()
        
        for epoch in range(20):
            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should decrease significantly
        initial_loss = np.mean(losses[:3])
        final_loss = np.mean(losses[-3:])
        
        assert final_loss < initial_loss * 0.5, "Model should converge on synthetic data"
    
    def test_learning_rate_sensitivity(self, base_config):
        """Test model sensitivity to learning rate"""
        learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
        final_losses = []
        
        # Fixed data for consistent comparison
        torch.manual_seed(42)
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        target = torch.randn(batch_size, base_config.pred_len, base_config.c_out)
        
        for lr in learning_rates:
            model = AutoformerFixed(base_config)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            # Train for fixed number of steps
            model.train()
            for step in range(10):
                optimizer.zero_grad()
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            final_losses.append(loss.item())
        
        # Different learning rates should produce different results
        loss_std = np.std(final_losses)
        assert loss_std > 1e-3, "Learning rate should affect training dynamics"
    
    def test_decomposition_impact(self, base_config):
        """Test impact of series decomposition on training"""
        # Create two identical models
        model_with_decomp = AutoformerFixed(base_config)
        
        # Create model without decomposition by replacing decomp layers
        model_no_decomp = AutoformerFixed(base_config)
        
        # Replace decomposition in encoder layers with identity
        for i, layer in enumerate(model_no_decomp.encoder.attn_layers):
            # Create dummy decomposition that returns input and zero trend
            layer.decomp1 = lambda x: (x, torch.zeros_like(x))
            layer.decomp2 = lambda x: (x, torch.zeros_like(x))
        
        # Create synthetic data
        batch_size = 4
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        target = torch.randn(batch_size, base_config.pred_len, base_config.c_out)
        
        # Compare outputs
        with torch.no_grad():
            output_with_decomp = model_with_decomp(x_enc, x_mark_enc, x_dec, x_mark_dec)
            output_no_decomp = model_no_decomp(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Outputs should be different due to decomposition
        assert not torch.allclose(output_with_decomp, output_no_decomp, atol=1e-3)
    
    def test_training_speed(self, base_config):
        """Test training speed with different configurations"""
        model = AutoformerFixed(base_config)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Create synthetic data
        batch_size = 4
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        target = torch.randn(batch_size, base_config.pred_len, base_config.c_out)
        
        # Measure training time
        start_time = time.time()
        model.train()
        
        for step in range(10):
            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        training_time = time.time() - start_time
        
        # Just a sanity check that training doesn't take too long
        assert training_time < 60, f"Training too slow: {training_time:.2f}s for 10 steps"
        print(f"Training time for 10 steps: {training_time:.2f}s")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])