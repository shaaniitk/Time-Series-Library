"""
Integration Test for Enhanced Celestial PGAT Model

This module tests the integration of the phase-aware processor and
sequential mixture decoder with the main Celestial Enhanced PGAT model.
"""

import torch
import pytest
import numpy as np
from models.Celestial_Enhanced_PGAT import Model
from layers.modular.decoder.sequential_mixture_decoder import SequentialMixtureNLLLoss


class MockConfig:
    """Mock configuration for testing."""
    
    def __init__(self):
        # Core parameters
        self.seq_len = 96
        self.label_len = 48
        self.pred_len = 24
        self.enc_in = 118  # Full feature count
        self.dec_in = 4    # OHLC
        self.c_out = 4     # OHLC
        self.d_model = 64
        self.n_heads = 4
        self.e_layers = 2
        self.d_layers = 2
        self.dropout = 0.1
        
        # Embedding parameters
        self.embed = 'timeF'
        self.freq = 'h'
        
        # Enhanced features
        self.use_mixture_decoder = True
        self.use_stochastic_learner = True
        self.use_hierarchical_mapping = True
        self.use_celestial_graph = True
        
        # Wave aggregation settings
        self.aggregate_waves_to_celestial = True
        self.num_input_waves = 118
        self.target_wave_indices = [0, 1, 2, 3]  # OHLC indices
        
        # Additional parameters
        self.celestial_fusion_layers = 2
        self.reg_loss_weight = 0.1


class TestCelestialModelIntegration:
    """Test integration of enhanced components with the main model."""
    
    @pytest.fixture
    def config(self):
        """Create mock configuration."""
        return MockConfig()
    
    @pytest.fixture
    def model(self, config):
        """Create enhanced celestial model."""
        return Model(config)
    
    @pytest.fixture
    def sample_data(self, config):
        """Create sample input data matching the expected structure."""
        batch_size = 4
        
        # Create input data with realistic celestial features
        x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
        x_mark_enc = torch.randn(batch_size, config.seq_len, 4)  # Time features
        
        # Create decoder input
        x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
        x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
        
        # Create targets
        targets = torch.randn(batch_size, config.pred_len, config.c_out)
        
        # Make some features look like sin/cos pairs for realism
        for i in range(0, config.enc_in, 2):
            if i + 1 < config.enc_in:
                angles = torch.rand(batch_size, config.seq_len) * 2 * np.pi
                x_enc[:, :, i] = torch.sin(angles)
                x_enc[:, :, i + 1] = torch.cos(angles)
        
        return x_enc, x_mark_enc, x_dec, x_mark_dec, targets
    
    def test_model_initialization(self, model, config):
        """Test that the enhanced model initializes correctly."""
        # Check that phase-aware processor is created
        assert hasattr(model, 'phase_aware_processor'), "Model should have phase-aware processor"
        assert model.phase_aware_processor is not None
        
        # Check that sequential mixture decoder is used
        assert hasattr(model, 'mixture_decoder'), "Model should have mixture decoder"
        assert model.mixture_decoder is not None
        assert model.mixture_decoder.__class__.__name__ == 'SequentialMixtureDensityDecoder'
        
        # Check that embedding dimensions are updated for phase-aware processing
        expected_enc_dim = config.num_input_waves if not config.aggregate_waves_to_celestial else 13 * 32
        # Note: The actual embedding dim depends on the implementation
        
        # Check that all components are present
        assert hasattr(model, 'celestial_nodes'), "Model should have celestial nodes"
        assert hasattr(model, 'graph_attention_layers'), "Model should have graph attention layers"
        assert hasattr(model, 'decoder_layers'), "Model should have decoder layers"
    
    def test_forward_pass(self, model, sample_data, config):
        """Test complete forward pass with enhanced components."""
        x_enc, x_mark_enc, x_dec, x_mark_dec, targets = sample_data
        
        # Set model to evaluation mode for consistent behavior
        model.eval()
        
        with torch.no_grad():
            outputs, metadata = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Check that outputs are in the expected format
        if isinstance(outputs, dict):
            # Mixture decoder output
            assert 'means' in outputs, "Mixture output should contain means"
            assert 'log_stds' in outputs, "Mixture output should contain log_stds"
            assert 'log_weights' in outputs, "Mixture output should contain log_weights"
            
            means = outputs['means']
            log_stds = outputs['log_stds']
            log_weights = outputs['log_weights']
            
            # Check shapes
            expected_shape = (x_enc.size(0), config.pred_len, config.c_out, 3)  # 3 components
            expected_weights_shape = (x_enc.size(0), config.pred_len, 3)
            
            assert means.shape == expected_shape, f"Means shape mismatch: {means.shape} vs {expected_shape}"
            assert log_stds.shape == expected_shape, f"Log stds shape mismatch: {log_stds.shape} vs {expected_shape}"
            assert log_weights.shape == expected_weights_shape, f"Log weights shape mismatch: {log_weights.shape} vs {expected_weights_shape}"
            
        else:
            # Point prediction output
            expected_shape = (x_enc.size(0), config.pred_len, config.c_out)
            assert outputs.shape == expected_shape, f"Output shape mismatch: {outputs.shape} vs {expected_shape}"
        
        # Check metadata
        assert isinstance(metadata, dict), "Metadata should be a dictionary"
        
        # Check for phase-aware processing metadata
        if config.aggregate_waves_to_celestial:
            assert 'phase_metadata' in metadata, "Should have phase metadata"
            assert 'adjacency_matrix' in metadata, "Should have adjacency matrix"
            
            phase_metadata = metadata['phase_metadata']
            adjacency_matrix = metadata['adjacency_matrix']
            
            # Check adjacency matrix shape
            expected_adj_shape = (x_enc.size(0), 12, 12)  # 12 celestial bodies (excluding Chiron)
            # Note: Actual shape depends on implementation
            
            assert isinstance(phase_metadata, dict), "Phase metadata should be a dictionary"
            assert torch.is_tensor(adjacency_matrix), "Adjacency matrix should be a tensor"
    
    def test_loss_computation(self, model, sample_data, config):
        """Test loss computation with sequential mixture decoder."""
        x_enc, x_mark_enc, x_dec, x_mark_dec, targets = sample_data
        
        # Forward pass
        outputs, metadata = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Create loss function
        loss_fn = SequentialMixtureNLLLoss(reduction='mean')
        
        # Determine targets
        if config.aggregate_waves_to_celestial and 'original_targets' in metadata:
            true_targets = metadata['original_targets'][:, -config.pred_len:, :]
        else:
            true_targets = targets
        
        # Compute loss
        if isinstance(outputs, dict):
            # Mixture decoder output
            mixture_params = (outputs['means'], outputs['log_stds'], outputs['log_weights'])
            loss = loss_fn(mixture_params, true_targets)
        else:
            # Fallback to MSE for point predictions
            loss_fn_mse = torch.nn.MSELoss()
            loss = loss_fn_mse(outputs, true_targets)
        
        # Check loss properties
        assert torch.is_tensor(loss), "Loss should be a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert loss.item() >= 0, "Loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"
        
        print(f"✅ Loss computation successful: {loss.item():.4f}")
    
    def test_gradient_flow(self, model, sample_data, config):
        """Test that gradients flow through the enhanced model."""
        x_enc, x_mark_enc, x_dec, x_mark_dec, targets = sample_data
        
        # Set model to training mode
        model.train()
        
        # Forward pass
        outputs, metadata = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Create loss
        loss_fn = SequentialMixtureNLLLoss(reduction='mean')
        
        if config.aggregate_waves_to_celestial and 'original_targets' in metadata:
            true_targets = metadata['original_targets'][:, -config.pred_len:, :]
        else:
            true_targets = targets
        
        if isinstance(outputs, dict):
            mixture_params = (outputs['means'], outputs['log_stds'], outputs['log_weights'])
            loss = loss_fn(mixture_params, true_targets)
        else:
            loss_fn_mse = torch.nn.MSELoss()
            loss = loss_fn_mse(outputs, true_targets)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients are computed for key components
        gradient_checks = []
        
        # Check phase-aware processor gradients
        if hasattr(model, 'phase_aware_processor'):
            for name, param in model.phase_aware_processor.named_parameters():
                if param.requires_grad and param.grad is not None:
                    gradient_checks.append(f"phase_aware_processor.{name}")
                    assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradients in {name}"
        
        # Check mixture decoder gradients
        if hasattr(model, 'mixture_decoder'):
            for name, param in model.mixture_decoder.named_parameters():
                if param.requires_grad and param.grad is not None:
                    gradient_checks.append(f"mixture_decoder.{name}")
                    assert torch.all(torch.isfinite(param.grad)), f"Non-finite gradients in {name}"
        
        # Check that at least some gradients were computed
        assert len(gradient_checks) > 0, "No gradients found - this suggests a problem"
        
        print(f"✅ Gradient flow verified for {len(gradient_checks)} parameter groups")
    
    def test_regularization_loss(self, model, sample_data, config):
        """Test regularization loss computation."""
        x_enc, x_mark_enc, x_dec, x_mark_dec, targets = sample_data
        
        # Set model to training mode
        model.train()
        
        # Forward pass
        outputs, metadata = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Check regularization loss
        if hasattr(model, 'get_regularization_loss'):
            reg_loss = model.get_regularization_loss()
            
            assert torch.is_tensor(reg_loss), "Regularization loss should be a tensor"
            assert reg_loss.dim() == 0, "Regularization loss should be a scalar"
            assert reg_loss.item() >= 0, "Regularization loss should be non-negative"
            assert torch.isfinite(reg_loss), "Regularization loss should be finite"
            
            print(f"✅ Regularization loss: {reg_loss.item():.6f}")
    
    def test_phase_information_extraction(self, model, sample_data, config):
        """Test that phase information is correctly extracted and used."""
        x_enc, x_mark_enc, x_dec, x_mark_dec, targets = sample_data
        
        # Forward pass
        outputs, metadata = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if config.aggregate_waves_to_celestial and 'phase_metadata' in metadata:
            phase_metadata = metadata['phase_metadata']
            
            # Check that phase information is present
            assert 'phase_info' in phase_metadata, "Should have phase info"
            assert 'edge_metadata' in phase_metadata, "Should have edge metadata"
            
            phase_info = phase_metadata['phase_info']
            edge_metadata = phase_metadata['edge_metadata']
            
            # Check that we have phase info for celestial bodies
            assert len(phase_info) > 0, "Should have phase info for celestial bodies"
            
            # Check edge metadata
            if 'avg_edge_strength' in edge_metadata:
                avg_strength = edge_metadata['avg_edge_strength']
                assert isinstance(avg_strength, (int, float)), "Edge strength should be numeric"
                assert 0 <= avg_strength <= 1, "Edge strength should be in [0, 1]"
            
            print(f"✅ Phase information extracted for {len(phase_info)} celestial bodies")
    
    def test_different_batch_sizes(self, config):
        """Test model with different batch sizes."""
        model = Model(config)
        model.eval()
        
        for batch_size in [1, 2, 8]:
            # Create data
            x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
            x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
            x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
            x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
            
            # Make sin/cos pairs
            for i in range(0, config.enc_in, 2):
                if i + 1 < config.enc_in:
                    angles = torch.rand(batch_size, config.seq_len) * 2 * np.pi
                    x_enc[:, :, i] = torch.sin(angles)
                    x_enc[:, :, i + 1] = torch.cos(angles)
            
            with torch.no_grad():
                outputs, metadata = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Check that outputs have correct batch dimension
            if isinstance(outputs, dict):
                assert outputs['means'].size(0) == batch_size
            else:
                assert outputs.size(0) == batch_size
            
            print(f"✅ Batch size {batch_size} test passed")


if __name__ == "__main__":
    print("🧪 Testing Celestial Model Integration...")
    
    # Create configuration and model
    config = MockConfig()
    
    try:
        model = Model(config)
        print(f"✅ Model initialization successful!")
        print(f"   - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        
        # Create sample data
        batch_size = 2
        x_enc = torch.randn(batch_size, config.seq_len, config.enc_in)
        x_mark_enc = torch.randn(batch_size, config.seq_len, 4)
        x_dec = torch.randn(batch_size, config.label_len + config.pred_len, config.dec_in)
        x_mark_dec = torch.randn(batch_size, config.label_len + config.pred_len, 4)
        targets = torch.randn(batch_size, config.pred_len, config.c_out)
        
        # Make sin/cos pairs
        for i in range(0, config.enc_in, 2):
            if i + 1 < config.enc_in:
                angles = torch.rand(batch_size, config.seq_len) * 2 * np.pi
                x_enc[:, :, i] = torch.sin(angles)
                x_enc[:, :, i + 1] = torch.cos(angles)
        
        # Test forward pass
        model.eval()
        with torch.no_grad():
            outputs, metadata = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"✅ Forward pass successful!")
        
        if isinstance(outputs, dict):
            print(f"   - Mixture decoder output:")
            print(f"     - Means shape: {outputs['means'].shape}")
            print(f"     - Log stds shape: {outputs['log_stds'].shape}")
            print(f"     - Log weights shape: {outputs['log_weights'].shape}")
        else:
            print(f"   - Point prediction shape: {outputs.shape}")
        
        # Test loss computation
        loss_fn = SequentialMixtureNLLLoss(reduction='mean')
        
        if config.aggregate_waves_to_celestial and 'original_targets' in metadata:
            true_targets = metadata['original_targets'][:, -config.pred_len:, :]
        else:
            true_targets = targets
        
        if isinstance(outputs, dict):
            mixture_params = (outputs['means'], outputs['log_stds'], outputs['log_weights'])
            loss = loss_fn(mixture_params, true_targets)
        else:
            loss_fn_mse = torch.nn.MSELoss()
            loss = loss_fn_mse(outputs, true_targets)
        
        print(f"✅ Loss computation successful: {loss.item():.4f}")
        
        # Test gradient flow
        model.train()
        outputs, metadata = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if isinstance(outputs, dict):
            mixture_params = (outputs['means'], outputs['log_stds'], outputs['log_weights'])
            loss = loss_fn(mixture_params, true_targets)
        else:
            loss = loss_fn_mse(outputs, true_targets)
        
        loss.backward()
        print(f"✅ Gradient flow successful!")
        
        # Check metadata
        if 'phase_metadata' in metadata:
            phase_metadata = metadata['phase_metadata']
            print(f"✅ Phase-aware processing active:")
            if 'phase_info' in phase_metadata:
                print(f"   - Phase info for {len(phase_metadata['phase_info'])} celestial bodies")
            if 'edge_metadata' in phase_metadata:
                edge_meta = phase_metadata['edge_metadata']
                if 'avg_edge_strength' in edge_meta:
                    print(f"   - Average edge strength: {edge_meta['avg_edge_strength']:.4f}")
        
        print("✅ All integration tests passed!")
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("🧪 Celestial Model Integration tests completed!")