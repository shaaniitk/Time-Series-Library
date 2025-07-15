import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from types import SimpleNamespace
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Direct import to avoid __init__.py issues
import importlib.util
spec = importlib.util.spec_from_file_location("autoformer_fixed", 
    os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                 "models", "Autoformer_Fixed.py"))
autoformer_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(autoformer_module)
AutoformerFixed = autoformer_module.Model


class TestSimpleTraining:
    """Simple training tests for Autoformer"""
    
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
    
    def test_model_creation(self, base_config):
        """Test that model can be created"""
        model = AutoformerFixed(base_config)
        assert model is not None
        assert hasattr(model, 'forward')
    
    def test_forward_pass(self, base_config):
        """Test basic forward pass"""
        model = AutoformerFixed(base_config)
        
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        assert output is not None
        assert output.shape == (batch_size, base_config.pred_len, base_config.c_out)
    
    def test_gradient_flow(self, base_config):
        """Test gradient flow"""
        model = AutoformerFixed(base_config)
        
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        target = torch.randn(batch_size, base_config.pred_len, base_config.c_out)
        
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        
        # Check that some parameters have gradients
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients, "Model should have gradients after backward pass"
    
    def test_basic_training_step(self, base_config):
        """Test basic training step"""
        model = AutoformerFixed(base_config)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        batch_size = 2
        x_enc = torch.randn(batch_size, base_config.seq_len, base_config.enc_in)
        x_mark_enc = torch.randn(batch_size, base_config.seq_len, 4)
        x_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, base_config.dec_in)
        x_mark_dec = torch.randn(batch_size, base_config.label_len + base_config.pred_len, 4)
        target = torch.randn(batch_size, base_config.pred_len, base_config.c_out)
        
        # Training step
        model.train()
        optimizer.zero_grad()
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        assert loss.item() > 0, "Loss should be positive"
        assert not torch.isnan(loss), "Loss should not be NaN"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])