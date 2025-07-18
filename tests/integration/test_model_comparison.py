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
from models.EnhancedAutoformer_Fixed import EnhancedAutoformer
from models.Transformer import Model as VanillaTransformer
from models.Informer import Model as Informer


class TestModelComparison:
    """Compare different time series models"""
    
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
    
    def train_model(self, model, data, epochs=10):
        """Train model and return losses and time"""
        x_enc, x_mark_enc, x_dec, x_mark_dec, target = data
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.MSELoss()
        
        model.train()
        losses = []
        
        start_time = time.time()
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        
        training_time = time.time() - start_time
        
        return losses, training_time
    
    def evaluate_model(self, model, data):
        """Evaluate model and return metrics"""
        x_enc, x_mark_enc, x_dec, x_mark_dec, target = data
        
        criterion = nn.MSELoss()
        
        model.eval()
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Calculate metrics
        mse = criterion(output, target).item()
        mae = nn.L1Loss()(output, target).item()
        
        # Calculate normalized metrics
        target_mean = target.mean()
        target_std = target.std()
        normalized_output = (output - target_mean) / target_std
        normalized_target = (target - target_mean) / target_std
        
        rmse = torch.sqrt(torch.mean((output - target) ** 2)).item()
        nrmse = torch.sqrt(torch.mean((normalized_output - normalized_target) ** 2)).item()
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'nrmse': nrmse
        }
    
    def test_model_accuracy_comparison(self, base_config):
        """Compare accuracy of different models"""
        # Create models
        models = {
            'Autoformer': AutoformerFixed(base_config),
            'EnhancedAutoformer': EnhancedAutoformer(base_config),
            'Transformer': VanillaTransformer(base_config),
            'Informer': Informer(base_config)
        }
        
        # Generate data
        data = self.generate_synthetic_data(base_config)
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            losses, training_time = self.train_model(model, data)
            
            print(f"Evaluating {name}...")
            metrics = self.evaluate_model(model, data)
            
            results[name] = {
                'initial_loss': losses[0],
                'final_loss': losses[-1],
                'training_time': training_time,
                **metrics
            }
            
            print(f"{name} results:")
            print(f"  Training time: {training_time:.2f}s")
            print(f"  MSE: {metrics['mse']:.6f}")
            print(f"  RMSE: {metrics['rmse']:.6f}")
            print(f"  MAE: {metrics['mae']:.6f}")
            print(f"  NRMSE: {metrics['nrmse']:.6f}")
        
        # All models should converge
        for name, result in results.items():
            assert result['final_loss'] < result['initial_loss'], f"{name} did not converge"
        
        # Compare models
        model_names = list(results.keys())
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                print(f"\nComparing {name1} vs {name2}:")
                
                # Compare MSE
                mse_diff = results[name1]['mse'] - results[name2]['mse']
                print(f"  MSE difference: {mse_diff:.6f}")
                
                # Compare training time
                time_ratio = results[name1]['training_time'] / results[name2]['training_time']
                print(f"  Training time ratio: {time_ratio:.2f}")
    
    def test_model_parameter_efficiency(self, base_config):
        """Compare parameter efficiency of different models"""
        # Create models
        models = {
            'Autoformer': AutoformerFixed(base_config),
            'EnhancedAutoformer': EnhancedAutoformer(base_config),
            'Transformer': VanillaTransformer(base_config),
            'Informer': Informer(base_config)
        }
        
        # Count parameters
        param_counts = {}
        
        for name, model in models.items():
            param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
            param_counts[name] = param_count
            print(f"{name} parameter count: {param_count:,}")
        
        # Generate data
        data = self.generate_synthetic_data(base_config)
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            losses, training_time = self.train_model(model, data, epochs=5)
            
            print(f"Evaluating {name}...")
            metrics = self.evaluate_model(model, data)
            
            results[name] = {
                'param_count': param_counts[name],
                'mse': metrics['mse'],
                'training_time': training_time
            }
        
        # Calculate parameter efficiency (MSE per 10k parameters)
        for name, result in results.items():
            efficiency = result['mse'] / (result['param_count'] / 10000)
            result['efficiency'] = efficiency
            print(f"{name} efficiency (MSE per 10k params): {efficiency:.6f}")
        
        # Compare efficiency
        model_names = list(results.keys())
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                eff_ratio = results[name1]['efficiency'] / results[name2]['efficiency']
                print(f"{name1} is {eff_ratio:.2f}x as efficient as {name2}")
    
    def test_model_inference_speed(self, base_config):
        """Compare inference speed of different models"""
        # Create models
        models = {
            'Autoformer': AutoformerFixed(base_config),
            'EnhancedAutoformer': EnhancedAutoformer(base_config),
            'Transformer': VanillaTransformer(base_config),
            'Informer': Informer(base_config)
        }
        
        # Generate data
        x_enc, x_mark_enc, x_dec, x_mark_dec, _ = self.generate_synthetic_data(base_config)
        
        # Measure inference time
        inference_times = {}
        
        for name, model in models.items():
            model.eval()
            
            # Warmup
            with torch.no_grad():
                _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Measure time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(10):
                    _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            end_time = time.time()
            
            avg_time = (end_time - start_time) / 10
            inference_times[name] = avg_time
            
            print(f"{name} inference time: {avg_time:.6f}s")
        
        # Compare inference times
        model_names = list(inference_times.keys())
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                time_ratio = inference_times[name1] / inference_times[name2]
                print(f"{name1} is {time_ratio:.2f}x as fast as {name2}")
    
    def test_model_scaling_with_sequence_length(self, base_config):
        """Test how models scale with sequence length"""
        # Create models
        models = {
            'Autoformer': AutoformerFixed,
            'Transformer': VanillaTransformer
        }
        
        seq_lengths = [48, 96, 192, 384]
        scaling_results = {name: [] for name in models}
        
        for seq_len in seq_lengths:
            # Update config
            config = base_config
            config.seq_len = seq_len
            
            for name, model_class in models.items():
                # Create model
                model = model_class(config)
                model.eval()
                
                # Generate data
                batch_size = 4
                x_enc, x_mark_enc, x_dec, x_mark_dec, _ = self.generate_synthetic_data(config, batch_size)
                
                # Measure inference time
                start_time = time.time()
                with torch.no_grad():
                    for _ in range(5):
                        _ = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 5
                scaling_results[name].append(avg_time)
                
                print(f"{name} with seq_len={seq_len}: {avg_time:.6f}s")
        
        # Calculate scaling factors
        for name, times in scaling_results.items():
            print(f"\n{name} scaling factors:")
            for i in range(1, len(seq_lengths)):
                time_ratio = times[i] / times[0]
                seq_ratio = seq_lengths[i] / seq_lengths[0]
                print(f"  {seq_lengths[0]}  {seq_lengths[i]}: {time_ratio:.2f}x time, {seq_ratio:.2f}x length")
                
                # Check if scaling is better than quadratic
                if name == 'Autoformer':
                    assert time_ratio < seq_ratio ** 2, f"Autoformer scaling should be better than quadratic, got {time_ratio:.2f}x for {seq_ratio:.2f}x length"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])