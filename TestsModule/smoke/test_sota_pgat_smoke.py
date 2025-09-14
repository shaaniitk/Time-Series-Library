"""Smoke tests for SOTA_Temporal_PGAT model.

Smoke tests provide quick validation that the model can be instantiated,
run basic operations, and doesn't have obvious failures. These tests are
designed to run fast and catch major regressions.
"""

# Suppress optional dependency warnings
from TestsModule.suppress_warnings import suppress_optional_dependency_warnings
suppress_optional_dependency_warnings()

import pytest
import torch
import numpy as np
from unittest.mock import MagicMock
import warnings

# Import the model
try:
    from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT as SOTATemporalPGAT
except ImportError:
    pytest.skip("SOTA_Temporal_PGAT model not available", allow_module_level=True)


class TestSOTATemporalPGATSmoke:
    """Smoke test suite for SOTA_Temporal_PGAT model."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Setup basic test configuration."""
        # Minimal configuration for smoke tests
        self.configs = MagicMock()
        self.configs.seq_len = 96
        self.configs.label_len = 48
        self.configs.pred_len = 96
        self.configs.enc_in = 7
        self.configs.dec_in = 7
        self.configs.c_out = 7
        self.configs.d_model = 128  # Smaller for faster tests
        self.configs.n_heads = 4
        self.configs.e_layers = 2
        self.configs.d_layers = 1
        self.configs.d_ff = 512
        self.configs.dropout = 0.1
        self.configs.activation = 'gelu'
        self.configs.output_attention = False
        self.configs.mix = True
        
        # Graph-specific configurations
        self.configs.graph_dim = 16  # Smaller for faster tests
        self.configs.num_nodes = self.configs.enc_in
        self.configs.graph_alpha = 0.2
        self.configs.graph_beta = 0.1
        
        # Test data dimensions
        self.batch_size = 2  # Small batch for speed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def create_minimal_test_data(self):
        """Create minimal test data for smoke tests."""
        x_enc = torch.randn(self.batch_size, self.configs.seq_len, self.configs.enc_in, device=self.device)
        x_mark_enc = torch.randn(self.batch_size, self.configs.seq_len, 4, device=self.device)
        x_dec = torch.randn(self.batch_size, self.configs.label_len + self.configs.pred_len, self.configs.dec_in, device=self.device)
        x_mark_dec = torch.randn(self.batch_size, self.configs.label_len + self.configs.pred_len, 4, device=self.device)
        
        return x_enc, x_mark_enc, x_dec, x_mark_dec
    
    def test_model_instantiation(self):
        """Test that model can be instantiated without errors."""
        # Should not raise any exceptions
        model = SOTATemporalPGAT(self.configs)
        
        # Basic checks
        assert model is not None
        assert hasattr(model, 'forward')
        assert hasattr(model, 'parameters')
        
        # Check that model has expected components
        assert hasattr(model, 'enc_embedding') or hasattr(model, 'encoder')
        assert hasattr(model, 'decoder') or hasattr(model, 'projection')
        
        print(f"✓ Model instantiated successfully")
        print(f"  Model type: {type(model).__name__}")
        print(f"  Device: {next(model.parameters()).device}")
    
    def test_model_forward_pass(self):
        """Test that model can perform forward pass without errors."""
        model = SOTATemporalPGAT(self.configs).to(self.device)
        model.eval()
        
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_minimal_test_data()
        
        # Should not raise any exceptions
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Basic output validation
        assert output is not None
        assert isinstance(output, torch.Tensor)
        assert output.shape[0] == self.batch_size  # Batch dimension preserved
        assert output.shape[-1] == self.configs.c_out  # Output features correct
        assert not torch.isnan(output).any(), "Output contains NaN values"
        assert torch.isfinite(output).all(), "Output contains infinite values"
        
        print(f"✓ Forward pass completed successfully")
        print(f"  Input shape: {x_enc.shape}")
        print(f"  Output shape: {output.shape}")
        print(f"  Output range: [{output.min().item():.3f}, {output.max().item():.3f}]")
    
    def test_model_training_mode(self):
        """Test that model can switch between training and evaluation modes."""
        model = SOTATemporalPGAT(self.configs).to(self.device)
        
        # Test training mode
        model.train()
        assert model.training
        
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_minimal_test_data()
        
        # Forward pass in training mode
        output_train = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        assert output_train is not None
        
        # Test evaluation mode
        model.eval()
        assert not model.training
        
        # Forward pass in evaluation mode
        with torch.no_grad():
            output_eval = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        assert output_eval is not None
        
        # Outputs should have same shape
        assert output_train.shape == output_eval.shape
        
        print(f"✓ Training/evaluation mode switching works")
        print(f"  Training mode output shape: {output_train.shape}")
        print(f"  Evaluation mode output shape: {output_eval.shape}")
    
    def test_gradient_computation(self):
        """Test that gradients can be computed without errors."""
        model = SOTATemporalPGAT(self.configs).to(self.device)
        model.train()
        
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_minimal_test_data()
        
        # Forward pass
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Compute simple loss
        loss = output.mean()
        
        # Backward pass should not raise exceptions
        loss.backward()
        
        # Check that gradients were computed
        grad_count = 0
        total_params = 0
        
        for name, param in model.named_parameters():
            total_params += 1
            if param.grad is not None:
                grad_count += 1
                assert torch.isfinite(param.grad).all(), f"Gradient for {name} contains non-finite values"
        
        assert grad_count > 0, "No gradients were computed"
        
        print(f"✓ Gradient computation successful")
        print(f"  Parameters with gradients: {grad_count}/{total_params}")
        print(f"  Loss value: {loss.item():.6f}")
    
    def test_different_batch_sizes(self):
        """Test that model works with different batch sizes."""
        model = SOTATemporalPGAT(self.configs).to(self.device)
        model.eval()
        
        batch_sizes = [1, 2, 4]
        
        for batch_size in batch_sizes:
            # Create data with specific batch size
            x_enc = torch.randn(batch_size, self.configs.seq_len, self.configs.enc_in, device=self.device)
            x_mark_enc = torch.randn(batch_size, self.configs.seq_len, 4, device=self.device)
            x_dec = torch.randn(batch_size, self.configs.label_len + self.configs.pred_len, self.configs.dec_in, device=self.device)
            x_mark_dec = torch.randn(batch_size, self.configs.label_len + self.configs.pred_len, 4, device=self.device)
            
            # Forward pass should work
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Check output shape
            assert output.shape[0] == batch_size
            assert not torch.isnan(output).any()
            
            print(f"  ✓ Batch size {batch_size}: output shape {output.shape}")
        
        print(f"✓ Different batch sizes handled correctly")
    
    def test_model_parameters(self):
        """Test basic properties of model parameters."""
        model = SOTATemporalPGAT(self.configs)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        # Basic sanity checks
        assert total_params > 0, "Model has no parameters"
        assert trainable_params > 0, "Model has no trainable parameters"
        assert trainable_params <= total_params, "More trainable than total parameters"
        
        # Check parameter initialization
        param_stats = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_mean = param.data.mean().item()
                param_std = param.data.std().item()
                param_stats.append((name, param_mean, param_std))
                
                # Parameters should be initialized (not all zeros)
                assert not torch.allclose(param.data, torch.zeros_like(param.data)), f"Parameter {name} is all zeros"
                
                # Parameters should be finite
                assert torch.isfinite(param.data).all(), f"Parameter {name} contains non-finite values"
        
        print(f"✓ Model parameters look healthy")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Parameter groups: {len(param_stats)}")
    
    def test_model_device_compatibility(self):
        """Test that model works on available devices."""
        devices_to_test = [torch.device('cpu')]
        
        if torch.cuda.is_available():
            devices_to_test.append(torch.device('cuda'))
        
        for device in devices_to_test:
            model = SOTATemporalPGAT(self.configs).to(device)
            model.eval()
            
            # Create data on the same device
            x_enc = torch.randn(2, self.configs.seq_len, self.configs.enc_in, device=device)
            x_mark_enc = torch.randn(2, self.configs.seq_len, 4, device=device)
            x_dec = torch.randn(2, self.configs.label_len + self.configs.pred_len, self.configs.dec_in, device=device)
            x_mark_dec = torch.randn(2, self.configs.label_len + self.configs.pred_len, 4, device=device)
            
            # Forward pass should work
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Output should be on the same device
            assert output.device == device
            assert not torch.isnan(output).any()
            
            print(f"  ✓ Device {device}: output shape {output.shape}")
        
        print(f"✓ Device compatibility verified")
    
    def test_model_deterministic_output(self):
        """Test that model produces deterministic output when using same seed."""
        # Set seeds for reproducibility
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        np.random.seed(42)
        
        model1 = SOTATemporalPGAT(self.configs).to(self.device)
        model1.eval()
        
        # Reset seeds
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(42)
        np.random.seed(42)
        
        model2 = SOTATemporalPGAT(self.configs).to(self.device)
        model2.eval()
        
        # Create same input data
        torch.manual_seed(123)
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_minimal_test_data()
        
        # Both models should produce same output
        with torch.no_grad():
            output1 = model1(x_enc, x_mark_enc, x_dec, x_mark_dec)
            output2 = model2(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Outputs should be identical (or very close due to floating point)
        assert torch.allclose(output1, output2, atol=1e-6), "Models with same seed produce different outputs"
        
        print(f"✓ Deterministic output verified")
        print(f"  Max difference: {(output1 - output2).abs().max().item():.2e}")
    
    def test_model_memory_cleanup(self):
        """Test that model doesn't cause obvious memory leaks."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for memory testing")
        
        # Clear cache
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()
        
        # Create and use model multiple times
        for i in range(5):
            model = SOTATemporalPGAT(self.configs).to(self.device)
            model.eval()
            
            x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_minimal_test_data()
            
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Delete model and output
            del model, output
            
            # Force garbage collection
            torch.cuda.empty_cache()
        
        final_memory = torch.cuda.memory_allocated()
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be minimal
        assert memory_increase < 100 * 1024 * 1024, f"Potential memory leak: {memory_increase / 1024**2:.1f}MB increase"
        
        print(f"✓ Memory cleanup verified")
        print(f"  Initial memory: {initial_memory / 1024**2:.1f}MB")
        print(f"  Final memory: {final_memory / 1024**2:.1f}MB")
        print(f"  Memory increase: {memory_increase / 1024**2:.1f}MB")
    
    def test_model_with_different_configs(self):
        """Test model with various configuration changes."""
        base_configs = self.configs
        
        # Test different configurations
        config_variations = [
            {'d_model': 64, 'n_heads': 2},  # Smaller model
            {'seq_len': 48, 'pred_len': 48},  # Different sequence lengths
            {'enc_in': 3, 'dec_in': 3, 'c_out': 3, 'num_nodes': 3},  # Different feature dimensions
            {'dropout': 0.0},  # No dropout
            {'activation': 'relu'},  # Different activation
        ]
        
        for i, config_changes in enumerate(config_variations):
            # Create modified config
            test_configs = MagicMock()
            
            # Copy base config
            for attr in dir(base_configs):
                if not attr.startswith('_'):
                    setattr(test_configs, attr, getattr(base_configs, attr))
            
            # Apply changes
            for key, value in config_changes.items():
                setattr(test_configs, key, value)
            
            try:
                # Create model with modified config
                model = SOTATemporalPGAT(test_configs).to(self.device)
                model.eval()
                
                # Create appropriate test data
                batch_size = 2
                x_enc = torch.randn(batch_size, test_configs.seq_len, test_configs.enc_in, device=self.device)
                x_mark_enc = torch.randn(batch_size, test_configs.seq_len, 4, device=self.device)
                x_dec = torch.randn(batch_size, test_configs.label_len + test_configs.pred_len, test_configs.dec_in, device=self.device)
                x_mark_dec = torch.randn(batch_size, test_configs.label_len + test_configs.pred_len, 4, device=self.device)
                
                # Test forward pass
                with torch.no_grad():
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                # Basic validation
                assert output.shape[0] == batch_size
                assert output.shape[-1] == test_configs.c_out
                assert not torch.isnan(output).any()
                
                print(f"  ✓ Config variation {i+1}: {config_changes}")
                
            except Exception as e:
                pytest.fail(f"Config variation {i+1} failed: {config_changes}, Error: {str(e)}")
        
        print(f"✓ Different configurations handled correctly")
    
    def test_model_import_and_dependencies(self):
        """Test that all required dependencies are available."""
        # Test that model can be imported
        try:
            from models.SOTA_Temporal_PGAT import Model as SOTATemporalPGAT
            assert SOTATemporalPGAT is not None
        except ImportError as e:
            pytest.fail(f"Failed to import SOTA_Temporal_PGAT: {e}")
        
        # Test that model components can be accessed
        model = SOTATemporalPGAT(self.configs)
        
        # Check for expected attributes/methods
        expected_attributes = ['forward', 'parameters', 'train', 'eval']
        for attr in expected_attributes:
            assert hasattr(model, attr), f"Model missing expected attribute: {attr}"
        
        print(f"✓ Import and dependencies verified")
        print(f"  Model class: {SOTATemporalPGAT.__name__}")
        print(f"  Model module: {SOTATemporalPGAT.__module__}")
    
    def test_quick_integration_check(self):
        """Quick integration test to ensure model works end-to-end."""
        model = SOTATemporalPGAT(self.configs).to(self.device)
        
        # Test full training step
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_minimal_test_data()
        
        # Forward pass
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Compute loss (simple MSE with target)
        target = torch.randn_like(output)
        loss = torch.nn.functional.mse_loss(output, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Test evaluation
        model.eval()
        with torch.no_grad():
            eval_output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Basic checks
        assert not torch.isnan(loss).any(), "Loss contains NaN"
        assert loss.item() > 0, "Loss should be positive"
        assert eval_output.shape == output.shape, "Eval output shape differs from training output"
        
        print(f"✓ Quick integration test passed")
        print(f"  Training loss: {loss.item():.6f}")
        print(f"  Output shape: {output.shape}")
        print(f"  Evaluation output shape: {eval_output.shape}")


if __name__ == "__main__":
    # Run smoke tests directly
    pytest.main([__file__, "-v"])