import pytest
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from types import SimpleNamespace
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.Autoformer_Fixed import Model as AutoformerFixed
from models.EnhancedAutoformer_Fixed import EnhancedAutoformer


class TestEndToEndWorkflows:
    """Test end-to-end workflows for Autoformer models"""
    
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
    
    def generate_synthetic_data(self, config, batch_size=16):
        """Generate synthetic data with trend and seasonality"""
        seq_len = config.seq_len
        pred_len = config.pred_len
        label_len = config.label_len
        features = config.enc_in
        
        # Generate time indices
        t_enc = torch.arange(0, seq_len).float() / seq_len
        t_dec = torch.arange(seq_len - label_len, seq_len + pred_len).float() / seq_len
        
        # Create features with different patterns
        x_enc_features = []
        for i in range(features):
            # Mix of trend and seasonal patterns
            trend = 0.1 * i * t_enc
            seasonal = 0.5 * torch.sin(2 * np.pi * (i + 1) * t_enc)
            feature = trend + seasonal + torch.randn_like(t_enc) * 0.1  # Add small noise
            x_enc_features.append(feature)
        
        x_enc = torch.stack(x_enc_features, dim=1).unsqueeze(0).repeat(batch_size, 1, 1)
        x_enc = x_enc.transpose(1, 2)  # [batch, seq_len, features]
        
        # Create decoder input (shifted target)
        x_dec_features = []
        for i in range(features):
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
        x_mark_dec = torch.randn(batch_size, label_len + pred_len, 4)
        
        return x_enc, x_mark_enc, x_dec, x_mark_dec, target
    
    def test_forecasting_workflow(self, base_config):
        """Test complete forecasting workflow"""
        # Set task to forecasting
        base_config.task_name = 'long_term_forecast'
        
        # Create model
        model = AutoformerFixed(base_config)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Generate data
        x_enc, x_mark_enc, x_dec, x_mark_dec, target = self.generate_synthetic_data(base_config)
        
        # Training loop
        model.train()
        losses = []
        
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should decrease
        assert losses[-1] < losses[0], "Training loss should decrease"
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            eval_output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        eval_loss = criterion(eval_output, target)
        assert eval_loss < losses[0], "Evaluation loss should be lower than initial training loss"
        
        # Prediction shape should match target
        assert eval_output.shape == target.shape
    
    def test_imputation_workflow(self, base_config):
        """Test imputation workflow"""
        # Set task to imputation
        base_config.task_name = 'imputation'
        base_config.mask_rate = 0.25  # 25% masking rate
        
        # Create model
        model = AutoformerFixed(base_config)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Generate data
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = self.generate_synthetic_data(base_config)
        
        # Create masked input for imputation
        batch_size, seq_len, features = x_enc.shape
        mask = torch.rand(batch_size, seq_len, features) > base_config.mask_rate
        masked_x_enc = x_enc.clone()
        masked_x_enc[~mask] = 0  # Mask values
        
        # Target is the original unmasked data
        target = x_enc
        
        # Training loop
        model.train()
        losses = []
        
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(masked_x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # For imputation, output should match the entire input sequence
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should decrease
        assert losses[-1] < losses[0], "Training loss should decrease"
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            eval_output = model(masked_x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Calculate loss only on masked values
        eval_loss_masked = criterion(eval_output[~mask], target[~mask])
        assert eval_loss_masked < losses[0], "Imputation loss should be lower than initial training loss"
        
        # Prediction shape should match target
        assert eval_output.shape == target.shape
    
    def test_anomaly_detection_workflow(self, base_config):
        """Test anomaly detection workflow"""
        # Set task to anomaly detection
        base_config.task_name = 'anomaly_detection'
        
        # Create model
        model = AutoformerFixed(base_config)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        # Generate data
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = self.generate_synthetic_data(base_config)
        
        # Create anomalies
        batch_size, seq_len, features = x_enc.shape
        anomaly_mask = torch.rand(batch_size, seq_len, features) < 0.05  # 5% anomalies
        anomalous_x_enc = x_enc.clone()
        anomalous_x_enc[anomaly_mask] = anomalous_x_enc[anomaly_mask] * 5  # Scale anomalies
        
        # Target is the original clean data
        target = x_enc
        
        # Training loop
        model.train()
        losses = []
        
        for epoch in range(10):
            optimizer.zero_grad()
            output = model(anomalous_x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # For anomaly detection, output should reconstruct the normal pattern
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should decrease
        assert losses[-1] < losses[0], "Training loss should decrease"
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            eval_output = model(anomalous_x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Calculate reconstruction error
        reconstruction_error = torch.abs(eval_output - anomalous_x_enc)
        
        # Anomalies should have higher reconstruction error
        normal_error = reconstruction_error[~anomaly_mask].mean()
        anomaly_error = reconstruction_error[anomaly_mask].mean()
        
        assert anomaly_error > normal_error, "Anomalies should have higher reconstruction error"
    
    def test_classification_workflow(self, base_config):
        """Test classification workflow"""
        # Set task to classification
        base_config.task_name = 'classification'
        base_config.num_classes = 5
        
        # Create model
        model = AutoformerFixed(base_config)
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        # Generate data
        batch_size = 16
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = self.generate_synthetic_data(base_config, batch_size)
        
        # Create class labels (random for testing)
        target = torch.randint(0, base_config.num_classes, (batch_size,))
        
        # Training loop
        model.train()
        losses = []
        
        for epoch in range(20):
            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # For classification, output should be class logits
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        # Loss should decrease
        assert losses[-1] < losses[0], "Training loss should decrease"
        
        # Evaluation
        model.eval()
        with torch.no_grad():
            eval_output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Output should be class logits
        assert eval_output.shape == (batch_size, base_config.num_classes)
        
        # Calculate accuracy
        _, predicted = torch.max(eval_output, 1)
        accuracy = (predicted == target).sum().item() / batch_size
        
        # With random labels, accuracy should be better than random guessing after training
        assert accuracy > 1.0 / base_config.num_classes, "Accuracy should be better than random guessing"
    
    def test_model_saving_loading(self, base_config):
        """Test model saving and loading"""
        import tempfile
        
        # Create model
        model = AutoformerFixed(base_config)
        
        # Generate data
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = self.generate_synthetic_data(base_config)
        
        # Get initial output
        model.eval()
        with torch.no_grad():
            initial_output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Save model
        with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp:
            model_path = tmp.name
            torch.save(model.state_dict(), model_path)
        
        # Create new model and load state
        new_model = AutoformerFixed(base_config)
        new_model.load_state_dict(torch.load(model_path))
        new_model.eval()
        
        # Get output from loaded model
        with torch.no_grad():
            loaded_output = new_model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Outputs should be identical
        assert torch.allclose(initial_output, loaded_output, atol=1e-6)
        
        # Clean up
        os.remove(model_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])