import pytest
import torch
import torch.nn as nn
from types import SimpleNamespace
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.Autoformer_Fixed import Model as AutoformerFixed
from models.EnhancedAutoformer_Fixed import EnhancedAutoformer


class TestModelIntegration:
    """Integration tests to ensure both models work as expected in realistic scenarios"""
    
    @pytest.fixture
    def realistic_config(self):
        """Realistic configuration similar to actual usage"""
        config = SimpleNamespace()
        config.task_name = 'long_term_forecast'
        config.seq_len = 96
        config.label_len = 48
        config.pred_len = 24
        config.enc_in = 7
        config.dec_in = 7
        config.c_out = 7
        config.d_model = 512
        config.n_heads = 8
        config.e_layers = 2
        config.d_layers = 1
        config.d_ff = 2048
        config.moving_avg = 25
        config.factor = 1
        config.dropout = 0.1
        config.activation = 'gelu'
        config.embed = 'timeF'
        config.freq = 'h'
        config.norm_type = 'LayerNorm'
        return config

    def test_autoformer_fixed_end_to_end(self, realistic_config):
        """Test AutoformerFixed end-to-end functionality"""
        model = AutoformerFixed(realistic_config)
        model.eval()
        
        # Realistic batch size and data
        batch_size = 8
        x_enc = torch.randn(batch_size, realistic_config.seq_len, realistic_config.enc_in)
        x_mark_enc = torch.randn(batch_size, realistic_config.seq_len, 4)
        x_dec = torch.randn(batch_size, realistic_config.label_len + realistic_config.pred_len, realistic_config.dec_in)
        x_mark_dec = torch.randn(batch_size, realistic_config.label_len + realistic_config.pred_len, 4)
        
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Validate output
        assert output.shape == (batch_size, realistic_config.pred_len, realistic_config.c_out)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        
        # Check output is reasonable (not all zeros or constant)
        assert output.std() > 1e-6
        assert not torch.allclose(output, torch.zeros_like(output))

    def test_enhanced_autoformer_end_to_end(self, realistic_config):
        """Test EnhancedAutoformer end-to-end functionality"""
        model = EnhancedAutoformer(realistic_config)
        model.eval()
        
        batch_size = 8
        x_enc = torch.randn(batch_size, realistic_config.seq_len, realistic_config.enc_in)
        x_mark_enc = torch.randn(batch_size, realistic_config.seq_len, 4)
        x_dec = torch.randn(batch_size, realistic_config.label_len + realistic_config.pred_len, realistic_config.dec_in)
        x_mark_dec = torch.randn(batch_size, realistic_config.label_len + realistic_config.pred_len, 4)
        
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Validate output
        assert output.shape == (batch_size, realistic_config.pred_len, realistic_config.c_out)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert output.std() > 1e-6

    def test_training_simulation(self, realistic_config):
        """Simulate training process for both models"""
        models = {
            'AutoformerFixed': AutoformerFixed(realistic_config),
            'EnhancedAutoformer': EnhancedAutoformer(realistic_config)
        }
        
        for model_name, model in models.items():
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            criterion = nn.MSELoss()
            
            # Simulate training steps
            for step in range(5):
                batch_size = 4
                x_enc = torch.randn(batch_size, realistic_config.seq_len, realistic_config.enc_in)
                x_mark_enc = torch.randn(batch_size, realistic_config.seq_len, 4)
                x_dec = torch.randn(batch_size, realistic_config.label_len + realistic_config.pred_len, realistic_config.dec_in)
                x_mark_dec = torch.randn(batch_size, realistic_config.label_len + realistic_config.pred_len, 4)
                target = torch.randn(batch_size, realistic_config.pred_len, realistic_config.c_out)
                
                optimizer.zero_grad()
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                loss = criterion(output, target)
                loss.backward()
                
                # Check gradients exist
                grad_norm = 0
                for param in model.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item()
                
                assert grad_norm > 0, f"{model_name}: No gradients computed"
                
                optimizer.step()
                
                # Loss should be finite
                assert torch.isfinite(loss).all(), f"{model_name}: Loss is not finite"

    def test_different_sequence_lengths(self, realistic_config):
        """Test models with different sequence lengths"""
        seq_lengths = [48, 96, 192, 336]
        pred_lengths = [12, 24, 48, 96]
        
        for seq_len, pred_len in zip(seq_lengths, pred_lengths):
            config = realistic_config
            config.seq_len = seq_len
            config.pred_len = pred_len
            config.label_len = pred_len // 2
            
            for ModelClass in [AutoformerFixed, EnhancedAutoformer]:
                model = ModelClass(config)
                model.eval()
                
                batch_size = 2
                x_enc = torch.randn(batch_size, seq_len, config.enc_in)
                x_mark_enc = torch.randn(batch_size, seq_len, 4)
                x_dec = torch.randn(batch_size, config.label_len + pred_len, config.dec_in)
                x_mark_dec = torch.randn(batch_size, config.label_len + pred_len, 4)
                
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                assert output.shape == (batch_size, pred_len, config.c_out)

    def test_multivariate_scenarios(self, realistic_config):
        """Test models with different numbers of variables"""
        variable_counts = [1, 7, 21, 50]
        
        for n_vars in variable_counts:
            config = realistic_config
            config.enc_in = n_vars
            config.dec_in = n_vars
            config.c_out = n_vars
            
            for ModelClass in [AutoformerFixed, EnhancedAutoformer]:
                model = ModelClass(config)
                model.eval()
                
                batch_size = 2
                x_enc = torch.randn(batch_size, config.seq_len, n_vars)
                x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
                x_dec = torch.randn(batch_size, config.label_len + config.pred_len, n_vars)
                x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
                
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                assert output.shape == (batch_size, config.pred_len, n_vars)

    def test_performance_comparison(self, realistic_config):
        """Compare performance characteristics of both models"""
        models = {
            'AutoformerFixed': AutoformerFixed(realistic_config),
            'EnhancedAutoformer': EnhancedAutoformer(realistic_config)
        }
        
        batch_size = 4
        x_enc = torch.randn(batch_size, realistic_config.seq_len, realistic_config.enc_in)
        x_mark_enc = torch.randn(batch_size, realistic_config.seq_len, 4)
        x_dec = torch.randn(batch_size, realistic_config.label_len + realistic_config.pred_len, realistic_config.dec_in)
        x_mark_dec = torch.randn(batch_size, realistic_config.label_len + realistic_config.pred_len, 4)
        
        results = {}
        
        for model_name, model in models.items():
            model.eval()
            
            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            # Measure inference time
            import time
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(10):
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            avg_time = (time.time() - start_time) / 10
            
            results[model_name] = {
                'total_params': total_params,
                'trainable_params': trainable_params,
                'avg_inference_time': avg_time,
                'output_shape': output.shape
            }
        
        # Both models should produce same output shape
        assert results['AutoformerFixed']['output_shape'] == results['EnhancedAutoformer']['output_shape']
        
        # Enhanced model might have more parameters due to learnable components
        print(f"AutoformerFixed params: {results['AutoformerFixed']['total_params']:,}")
        print(f"EnhancedAutoformer params: {results['EnhancedAutoformer']['total_params']:,}")

    def test_edge_cases(self, realistic_config):
        """Test edge cases and boundary conditions"""
        # Test with minimum sequence length
        config = realistic_config
        config.seq_len = 12
        config.pred_len = 1
        config.label_len = 1
        
        for ModelClass in [AutoformerFixed, EnhancedAutoformer]:
            model = ModelClass(config)
            
            x_enc = torch.randn(1, config.seq_len, config.enc_in)
            x_mark_enc = torch.randn(1, config.seq_len, 4)
            x_dec = torch.randn(1, config.label_len + config.pred_len, config.dec_in)
            x_mark_dec = torch.randn(1, config.label_len + config.pred_len, 4)
            
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            assert output.shape == (1, config.pred_len, config.c_out)

    def test_quantile_integration(self, realistic_config):
        """Test quantile functionality using ModularAutoformer with proper quantile configuration"""
        # Import the quantile bayesian config and ModularAutoformer
        from configs.autoformer.quantile_bayesian_config import get_quantile_bayesian_autoformer_config
        from models.modular_autoformer import ModularAutoformer
        
        # Create proper quantile bayesian configuration with 5 quantiles
        config = get_quantile_bayesian_autoformer_config(
            num_targets=7,
            num_covariates=3,
            seq_len=96,
            pred_len=24,
            label_len=48,
            quantile_levels=[0.1, 0.25, 0.5, 0.75, 0.9]
        )
        
        print(f"=== Quantile Integration Test ===")
        print(f"Loss function: {config.loss_function_type}")
        print(f"Output head type: {config.output_head_type}")
        print(f"Quantiles: {config.quantile_levels}")
        print(f"c_out (model output): {config.c_out}")
        print(f"c_out_evaluation (targets): {config.c_out_evaluation}")
        
        model = ModularAutoformer(config)
        model.eval()
        
        batch_size = 2
        x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
        x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
        x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
        x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
        
        with torch.no_grad():
            output = model(x_enc=x_enc, x_mark_enc=x_mark_enc, x_dec=x_dec, x_mark_dec=x_mark_dec)
        
        # For quantile models, output should be [batch, pred_len, num_features * num_quantiles]
        expected_shape = (batch_size, config.pred_len, config.c_out)
        print(f"Expected shape: {expected_shape}")
        print(f"Actual shape: {output.shape}")
        
        assert output.shape == expected_shape, f"Expected {expected_shape}, got {output.shape}"
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert not torch.isinf(output).any(), "Output contains Inf values"
        
        # Verify the model is properly configured for quantiles
        assert config.loss_function_type == 'bayesian_quantile'
        assert config.output_head_type == 'quantile'
        assert len(config.quantile_levels) == 5
        assert config.c_out == config.c_out_evaluation * len(config.quantile_levels)  # 7 * 5 = 35


if __name__ == "__main__":
    pytest.main([__file__, "-v"])