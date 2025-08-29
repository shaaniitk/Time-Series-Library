import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from types import SimpleNamespace
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Autoformer_Fixed import Model as AutoformerFixed
from models.EnhancedAutoformer_Fixed import EnhancedAutoformer


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
    
    def test_gradient_flow_quality(self, base_config):
        """Test gradient flow through all model components"""
        model = AutoformerFixed(base_config)
        
        # Create synthetic data
        batch_size = 4
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in, requires_grad=True)
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
        total_grad_norm = 0
        param_count = 0
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                grad_norm = param.grad.norm().item()
                assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
                total_grad_norm += grad_norm
                param_count += 1
        
        # Total gradient norm should be reasonable
        avg_grad_norm = total_grad_norm / param_count
        assert 1e-6 < avg_grad_norm < 1e3, f"Unusual gradient magnitude: {avg_grad_norm}"
    
    def test_loss_convergence_behavior(self, base_config):
        """Test that model can overfit to small dataset (convergence test)"""
        model = AutoformerFixed(base_config)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Small synthetic dataset
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        target = torch.randn(batch_size, base_config.pred_len, base_config.c_out)
        
        losses = []
        model.train()
        
        # Train for several steps
        for step in range(50):
            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should decrease (overfitting to small dataset)
        initial_loss = np.mean(losses[:5])
        final_loss = np.mean(losses[-5:])
        assert final_loss < initial_loss * 0.8, "Model should overfit to small dataset"
        
        # Loss should be finite throughout training
        assert all(np.isfinite(l) for l in losses), "Loss became non-finite during training"
    
    def test_parameter_update_magnitudes(self, base_config):
        """Test parameter update magnitudes are reasonable"""
        model = AutoformerFixed(base_config)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        
        # Store initial parameters
        initial_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                initial_params[name] = param.data.clone()
        
        # Single training step
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        target = torch.randn(batch_size, base_config.pred_len, base_config.c_out)
        
        optimizer.zero_grad()
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        
        # Check parameter update magnitudes
        for name, param in model.named_parameters():
            if param.requires_grad and name in initial_params:
                update_magnitude = (param.data - initial_params[name]).norm().item()
                param_magnitude = param.data.norm().item()
                
                # Update should be small relative to parameter magnitude
                relative_update = update_magnitude / (param_magnitude + 1e-8)
                assert relative_update < 0.1, f"Large parameter update in {name}: {relative_update}"
    
    def test_learning_rate_sensitivity(self, base_config):
        """Test model behavior with different learning rates"""
        learning_rates = [1e-5, 1e-4, 1e-3, 1e-2]
        final_losses = []
        
        for lr in learning_rates:
            model = AutoformerFixed(base_config)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.MSELoss()
            
            # Fixed dataset
            torch.manual_seed(42)
            batch_size = 2
            x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
            x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
            x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
            x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
            target = torch.randn(batch_size, base_config.pred_len, base_config.c_out)
            
            # Train for 20 steps
            model.train()
            for step in range(20):
                optimizer.zero_grad()
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            
            final_losses.append(loss.item())
        
        # Should see some variation in final losses with different LRs
        loss_std = np.std(final_losses)
        assert loss_std > 1e-6, "Learning rate should affect final loss"
        
        # Very high LR might cause instability
        if final_losses[-1] > final_losses[0] * 10:
            print(f"Warning: High LR ({learning_rates[-1]}) may cause instability")
    
    def test_enhanced_model_training_stability(self, base_config):
        """Test enhanced model training stability"""
        model = EnhancedAutoformer(base_config)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Check learnable decomposition weights are updated
        initial_weights = model.decomp.trend_weights.data.clone()
        
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        target = torch.randn(batch_size, base_config.pred_len, base_config.c_out)
        
        # Training steps
        for step in range(10):
            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
        
        # Learnable weights should have changed
        final_weights = model.decomp.trend_weights.data
        assert not torch.allclose(initial_weights, final_weights, atol=1e-6), \
            "Learnable decomposition weights should update during training"
    
    def test_gradient_clipping_necessity(self, base_config):
        """Test if gradient clipping is necessary"""
        model = AutoformerFixed(base_config)
        
        # Use larger learning rate to potentially cause gradient explosion
        optimizer = optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.MSELoss()
        
        batch_size = 4
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        target = torch.randn(batch_size, base_config.pred_len, base_config.c_out)
        
        max_grad_norms = []
        
        for step in range(10):
            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(output, target)
            loss.backward()
            
            # Calculate gradient norm before clipping
            total_norm = 0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm = param.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** (1. / 2)
            max_grad_norms.append(total_norm)
            
            optimizer.step()
        
        max_grad_norm = max(max_grad_norms)
        
        # If gradient norm exceeds threshold, clipping might be beneficial
        if max_grad_norm > 10.0:
            print(f"Warning: Large gradient norm detected ({max_grad_norm:.2f}). Consider gradient clipping.")


if __name__ == "__main__":
    import numpy as np
    pytest.main([__file__, "-v"])