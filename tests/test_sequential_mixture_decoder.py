"""
Test Sequential Mixture Decoder

This module tests the sequential mixture density decoder that preserves
temporal structure throughout the decoding process.
"""

import torch
import pytest
import numpy as np
import math
from layers.modular.decoder.sequential_mixture_decoder import (
    SequentialMixtureDensityDecoder,
    SequentialMixtureNLLLoss,
    PositionalEncoding
)


class TestSequentialMixtureDensityDecoder:
    """Test the sequential mixture density decoder."""
    
    @pytest.fixture
    def decoder_params(self):
        """Standard decoder parameters."""
        return {
            'd_model': 64,
            'pred_len': 24,
            'num_components': 3,
            'num_targets': 4,  # OHLC
            'num_decoder_layers': 2,
            'num_heads': 4,
            'dropout': 0.1
        }
    
    @pytest.fixture
    def decoder(self, decoder_params):
        """Create a sequential mixture decoder."""
        return SequentialMixtureDensityDecoder(**decoder_params)
    
    @pytest.fixture
    def sample_data(self, decoder_params):
        """Create sample encoder and decoder data."""
        batch_size = 8
        enc_seq_len = 96
        dec_seq_len = decoder_params['pred_len']
        d_model = decoder_params['d_model']
        
        encoder_output = torch.randn(batch_size, enc_seq_len, d_model)
        decoder_input = torch.randn(batch_size, dec_seq_len, d_model)
        
        return encoder_output, decoder_input
    
    def test_initialization(self, decoder, decoder_params):
        """Test decoder initialization."""
        assert decoder.d_model == decoder_params['d_model']
        assert decoder.pred_len == decoder_params['pred_len']
        assert decoder.num_components == decoder_params['num_components']
        assert decoder.num_targets == decoder_params['num_targets']
        
        # Check that components are initialized
        assert decoder.pos_encoding is not None
        assert decoder.transformer_decoder is not None
        assert len(decoder.mixture_heads) == 3  # means, log_stds, log_weights
        assert decoder.pre_mixture_norm is not None
    
    def test_forward_pass(self, decoder, sample_data, decoder_params):
        """Test forward pass with sample data."""
        encoder_output, decoder_input = sample_data
        
        means, log_stds, log_weights = decoder(encoder_output, decoder_input)
        
        batch_size = encoder_output.size(0)
        pred_len = decoder_params['pred_len']
        num_targets = decoder_params['num_targets']
        num_components = decoder_params['num_components']
        
        # Check output shapes
        expected_means_shape = (batch_size, pred_len, num_targets, num_components)
        expected_log_stds_shape = (batch_size, pred_len, num_targets, num_components)
        expected_log_weights_shape = (batch_size, pred_len, num_components)
        
        assert means.shape == expected_means_shape, f"Expected {expected_means_shape}, got {means.shape}"
        assert log_stds.shape == expected_log_stds_shape, f"Expected {expected_log_stds_shape}, got {log_stds.shape}"
        assert log_weights.shape == expected_log_weights_shape, f"Expected {expected_log_weights_shape}, got {log_weights.shape}"
        
        # Check that log_stds are bounded (for numerical stability)
        assert torch.all(log_stds >= -10), "log_stds should be >= -10"
        assert torch.all(log_stds <= 2), "log_stds should be <= 2"
        
        # Check that outputs are finite
        assert torch.all(torch.isfinite(means)), "Means should be finite"
        assert torch.all(torch.isfinite(log_stds)), "Log stds should be finite"
        assert torch.all(torch.isfinite(log_weights)), "Log weights should be finite"
    
    def test_causal_mask(self, decoder, sample_data):
        """Test that causal mask is properly applied."""
        encoder_output, decoder_input = sample_data
        
        # Create a custom decoder input where later timesteps have distinct patterns
        batch_size, seq_len, d_model = decoder_input.shape
        decoder_input[:, :seq_len//2, :] = 1.0  # First half
        decoder_input[:, seq_len//2:, :] = -1.0  # Second half
        
        means1, _, _ = decoder(encoder_output, decoder_input)
        
        # Modify only the last timestep
        decoder_input_modified = decoder_input.clone()
        decoder_input_modified[:, -1, :] = 10.0  # Large change in last timestep
        
        means2, _, _ = decoder(encoder_output, decoder_input_modified)
        
        # Due to causal masking, early timesteps should be identical
        # (they shouldn't see the future modification)
        early_timesteps = means1[:, :seq_len//2, :, :]
        early_timesteps_modified = means2[:, :seq_len//2, :, :]
        
        # Early timesteps should be very similar (allowing for small numerical differences)
        assert torch.allclose(early_timesteps, early_timesteps_modified, atol=1e-3), \
            "Early timesteps should not be affected by future changes (causal masking)"
    
    def test_sampling(self, decoder, sample_data):
        """Test sampling from the mixture distribution."""
        encoder_output, decoder_input = sample_data
        
        mixture_params = decoder(encoder_output, decoder_input)
        
        # Test sampling
        num_samples = 5
        samples = decoder.sample(mixture_params, num_samples)
        
        batch_size = encoder_output.size(0)
        pred_len = decoder.pred_len
        num_targets = decoder.num_targets
        
        expected_shape = (num_samples, batch_size, pred_len, num_targets)
        assert samples.shape == expected_shape, f"Expected {expected_shape}, got {samples.shape}"
        
        # Check that samples are finite
        assert torch.all(torch.isfinite(samples)), "Samples should be finite"
        
        # Check that samples have reasonable variance (not all identical)
        sample_std = samples.std(dim=0)
        assert torch.any(sample_std > 1e-6), "Samples should have some variance"
    
    def test_point_prediction(self, decoder, sample_data):
        """Test point prediction extraction."""
        encoder_output, decoder_input = sample_data
        
        mixture_params = decoder(encoder_output, decoder_input)
        point_pred = decoder.get_point_prediction(mixture_params)
        
        batch_size = encoder_output.size(0)
        pred_len = decoder.pred_len
        num_targets = decoder.num_targets
        
        expected_shape = (batch_size, pred_len, num_targets)
        assert point_pred.shape == expected_shape, f"Expected {expected_shape}, got {point_pred.shape}"
        
        # Check that predictions are finite
        assert torch.all(torch.isfinite(point_pred)), "Point predictions should be finite"
    
    def test_uncertainty_estimates(self, decoder, sample_data):
        """Test uncertainty estimation."""
        encoder_output, decoder_input = sample_data
        
        mixture_params = decoder(encoder_output, decoder_input)
        uncertainty = decoder.get_uncertainty_estimates(mixture_params)
        
        # Check that all uncertainty measures are present
        expected_keys = ['total_std', 'epistemic_std', 'aleatoric_std', 'mixture_weights']
        for key in expected_keys:
            assert key in uncertainty, f"Missing uncertainty measure: {key}"
        
        batch_size = encoder_output.size(0)
        pred_len = decoder.pred_len
        num_targets = decoder.num_targets
        num_components = decoder.num_components
        
        # Check shapes
        expected_std_shape = (batch_size, pred_len, num_targets)
        expected_weights_shape = (batch_size, pred_len, num_components)
        
        assert uncertainty['total_std'].shape == expected_std_shape
        assert uncertainty['epistemic_std'].shape == expected_std_shape
        assert uncertainty['aleatoric_std'].shape == expected_std_shape
        assert uncertainty['mixture_weights'].shape == expected_weights_shape
        
        # Check that standard deviations are positive
        assert torch.all(uncertainty['total_std'] >= 0), "Total std should be non-negative"
        assert torch.all(uncertainty['epistemic_std'] >= 0), "Epistemic std should be non-negative"
        assert torch.all(uncertainty['aleatoric_std'] >= 0), "Aleatoric std should be non-negative"
        
        # Check that mixture weights sum to 1
        weight_sums = uncertainty['mixture_weights'].sum(dim=-1)
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-6), \
            "Mixture weights should sum to 1"
    
    def test_different_sequence_lengths(self, decoder_params):
        """Test decoder with different sequence lengths."""
        decoder = SequentialMixtureDensityDecoder(**decoder_params)
        
        batch_size = 4
        d_model = decoder_params['d_model']
        
        # Test with different encoder sequence lengths
        for enc_len in [48, 96, 144]:
            encoder_output = torch.randn(batch_size, enc_len, d_model)
            decoder_input = torch.randn(batch_size, decoder_params['pred_len'], d_model)
            
            means, log_stds, log_weights = decoder(encoder_output, decoder_input)
            
            # Output shape should be consistent regardless of encoder length
            assert means.shape[1] == decoder_params['pred_len']
            assert log_stds.shape[1] == decoder_params['pred_len']
            assert log_weights.shape[1] == decoder_params['pred_len']


class TestSequentialMixtureNLLLoss:
    """Test the sequential mixture NLL loss function."""
    
    @pytest.fixture
    def loss_function(self):
        """Create a sequential mixture NLL loss."""
        return SequentialMixtureNLLLoss(reduction='mean')
    
    @pytest.fixture
    def sample_mixture_params(self):
        """Create sample mixture parameters."""
        batch_size, seq_len, num_targets, num_components = 4, 24, 4, 3
        
        means = torch.randn(batch_size, seq_len, num_targets, num_components)
        log_stds = torch.randn(batch_size, seq_len, num_targets, num_components) * 0.5 - 1.0  # Around -1
        log_weights = torch.randn(batch_size, seq_len, num_components)
        
        return means, log_stds, log_weights
    
    @pytest.fixture
    def sample_targets(self, sample_mixture_params):
        """Create sample targets."""
        means, _, _ = sample_mixture_params
        batch_size, seq_len, num_targets = means.shape[:3]
        
        # Create targets that are somewhat consistent with the means
        targets = torch.randn(batch_size, seq_len, num_targets)
        return targets
    
    def test_loss_computation(self, loss_function, sample_mixture_params, sample_targets):
        """Test basic loss computation."""
        loss = loss_function(sample_mixture_params, sample_targets)
        
        # Check that loss is a scalar
        assert loss.dim() == 0, "Loss should be a scalar"
        
        # Check that loss is positive (NLL should be positive)
        assert loss.item() >= 0, "NLL loss should be non-negative"
        
        # Check that loss is finite
        assert torch.isfinite(loss), "Loss should be finite"
    
    def test_loss_reduction(self, sample_mixture_params, sample_targets):
        """Test different reduction modes."""
        # Test mean reduction
        loss_mean = SequentialMixtureNLLLoss(reduction='mean')
        loss_val_mean = loss_mean(sample_mixture_params, sample_targets)
        
        # Test sum reduction
        loss_sum = SequentialMixtureNLLLoss(reduction='sum')
        loss_val_sum = loss_sum(sample_mixture_params, sample_targets)
        
        # Test no reduction
        loss_none = SequentialMixtureNLLLoss(reduction='none')
        loss_val_none = loss_none(sample_mixture_params, sample_targets)
        
        # Check shapes
        assert loss_val_mean.dim() == 0, "Mean reduction should give scalar"
        assert loss_val_sum.dim() == 0, "Sum reduction should give scalar"
        assert loss_val_none.shape == sample_targets.shape[:2], "No reduction should preserve batch and time dims"
        
        # Check relationship between reductions
        expected_mean = loss_val_none.mean()
        expected_sum = loss_val_none.sum()
        
        assert torch.allclose(loss_val_mean, expected_mean, atol=1e-6), "Mean reduction incorrect"
        assert torch.allclose(loss_val_sum, expected_sum, atol=1e-6), "Sum reduction incorrect"
    
    def test_gradient_flow(self, loss_function, sample_mixture_params, sample_targets):
        """Test that gradients flow properly through the loss."""
        means, log_stds, log_weights = sample_mixture_params
        
        # Make parameters require gradients
        means.requires_grad_(True)
        log_stds.requires_grad_(True)
        log_weights.requires_grad_(True)
        
        loss = loss_function((means, log_stds, log_weights), sample_targets)
        loss.backward()
        
        # Check that gradients are computed
        assert means.grad is not None, "Means should have gradients"
        assert log_stds.grad is not None, "Log stds should have gradients"
        assert log_weights.grad is not None, "Log weights should have gradients"
        
        # Check that gradients are finite
        assert torch.all(torch.isfinite(means.grad)), "Means gradients should be finite"
        assert torch.all(torch.isfinite(log_stds.grad)), "Log stds gradients should be finite"
        assert torch.all(torch.isfinite(log_weights.grad)), "Log weights gradients should be finite"
    
    def test_perfect_prediction(self, loss_function):
        """Test loss when predictions are perfect."""
        batch_size, seq_len, num_targets, num_components = 2, 12, 4, 3
        
        # Create targets
        targets = torch.randn(batch_size, seq_len, num_targets)
        
        # Create means that exactly match targets (for first component)
        means = torch.randn(batch_size, seq_len, num_targets, num_components)
        means[:, :, :, 0] = targets  # First component matches targets exactly
        
        # Very small standard deviations (high confidence)
        log_stds = torch.full((batch_size, seq_len, num_targets, num_components), -5.0)
        
        # High weight for first component
        log_weights = torch.full((batch_size, seq_len, num_components), -10.0)
        log_weights[:, :, 0] = 0.0  # First component has highest weight
        
        loss = loss_function((means, log_stds, log_weights), targets)
        
        # Loss should be very small for perfect predictions
        assert loss.item() < 10.0, f"Loss should be small for perfect predictions, got {loss.item()}"


class TestPositionalEncoding:
    """Test the positional encoding component."""
    
    def test_positional_encoding(self):
        """Test positional encoding generation."""
        d_model = 64
        max_len = 100
        
        pos_encoding = PositionalEncoding(d_model, max_len)
        
        # Test with sample input
        batch_size, seq_len = 4, 24
        x = torch.randn(batch_size, seq_len, d_model)
        
        encoded = pos_encoding(x)
        
        # Check that shape is preserved
        assert encoded.shape == x.shape, "Positional encoding should preserve shape"
        
        # Check that encoding is different from input
        assert not torch.allclose(encoded, x), "Positional encoding should modify input"
        
        # Check that encoding is deterministic
        encoded2 = pos_encoding(x)
        assert torch.allclose(encoded, encoded2), "Positional encoding should be deterministic"
    
    def test_different_sequence_lengths(self):
        """Test positional encoding with different sequence lengths."""
        d_model = 32
        max_len = 200
        
        pos_encoding = PositionalEncoding(d_model, max_len)
        
        batch_size = 2
        
        for seq_len in [10, 50, 100, 150]:
            x = torch.randn(batch_size, seq_len, d_model)
            encoded = pos_encoding(x)
            
            assert encoded.shape == x.shape, f"Shape mismatch for seq_len={seq_len}"
            assert torch.all(torch.isfinite(encoded)), f"Non-finite values for seq_len={seq_len}"


if __name__ == "__main__":
    # Run basic tests
    print("🧪 Testing Sequential Mixture Decoder...")
    
    # Test decoder
    decoder = SequentialMixtureDensityDecoder(
        d_model=64,
        pred_len=24,
        num_components=3,
        num_targets=4,
        num_decoder_layers=2,
        num_heads=4,
        dropout=0.1
    )
    
    # Create sample data
    batch_size = 4
    enc_seq_len = 96
    dec_seq_len = 24
    d_model = 64
    
    encoder_output = torch.randn(batch_size, enc_seq_len, d_model)
    decoder_input = torch.randn(batch_size, dec_seq_len, d_model)
    
    try:
        # Test forward pass
        means, log_stds, log_weights = decoder(encoder_output, decoder_input)
        print(f"✅ Forward pass successful!")
        print(f"   - Means shape: {means.shape}")
        print(f"   - Log stds shape: {log_stds.shape}")
        print(f"   - Log weights shape: {log_weights.shape}")
        
        # Test point prediction
        point_pred = decoder.get_point_prediction((means, log_stds, log_weights))
        print(f"   - Point prediction shape: {point_pred.shape}")
        
        # Test uncertainty estimates
        uncertainty = decoder.get_uncertainty_estimates((means, log_stds, log_weights))
        print(f"   - Uncertainty keys: {list(uncertainty.keys())}")
        
        # Test sampling
        samples = decoder.sample((means, log_stds, log_weights), num_samples=3)
        print(f"   - Samples shape: {samples.shape}")
        
        # Test loss function
        loss_fn = SequentialMixtureNLLLoss(reduction='mean')
        targets = torch.randn(batch_size, dec_seq_len, 4)
        loss = loss_fn((means, log_stds, log_weights), targets)
        print(f"   - Loss value: {loss.item():.4f}")
        
        print("✅ All Sequential Mixture Decoder tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("🧪 Sequential Mixture Decoder tests completed!")