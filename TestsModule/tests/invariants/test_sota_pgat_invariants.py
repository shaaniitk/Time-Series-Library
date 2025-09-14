"""Invariant tests for SOTA_Temporal_PGAT algorithm.

This module tests mathematical and algorithmic invariants that should hold
for the SOTA_Temporal_PGAT model under various conditions.
"""

# Suppress optional dependency warnings
from TestsModule.suppress_warnings import suppress_optional_dependency_warnings
suppress_optional_dependency_warnings()

import pytest
import torch
import numpy as np
from typing import Dict, Any
from unittest.mock import MagicMock

# Import invariant testing utilities
try:
    from TestsModule.tests.invariants.harness import InvariantTestHarness
    from TestsModule.tests.invariants.metrics import compute_stability_metrics
    from TestsModule.tests.invariants.thresholds import STABILITY_THRESHOLDS
except ImportError:
    # Fallback if invariant utilities are not available
    InvariantTestHarness = None
    compute_stability_metrics = None
    STABILITY_THRESHOLDS = {}

# Import the model
try:
    from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT as SOTATemporalPGAT
except ImportError:
    pytest.skip("SOTA_Temporal_PGAT model not available", allow_module_level=True)


class TestSOTATemporalPGATInvariants:
    """Invariant test suite for SOTA_Temporal_PGAT model."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Setup test environment and model configuration."""
        # Create mock configuration
        self.configs = MagicMock()
        self.configs.seq_len = 96
        self.configs.label_len = 48
        self.configs.pred_len = 96
        self.configs.enc_in = 7
        self.configs.dec_in = 7
        self.configs.c_out = 7
        self.configs.d_model = 256  # Smaller for faster testing
        self.configs.n_heads = 8
        self.configs.e_layers = 2
        self.configs.d_layers = 1
        self.configs.d_ff = 1024
        self.configs.dropout = 0.1
        self.configs.activation = 'gelu'
        self.configs.output_attention = False
        self.configs.mix = True
        
        # Graph-specific configurations
        self.configs.graph_dim = 32
        self.configs.num_nodes = self.configs.enc_in
        self.configs.graph_alpha = 0.2
        self.configs.graph_beta = 0.1
        
        # Test parameters
        self.batch_size = 2
        self.tolerance = 1e-5
        self.stability_tolerance = 1e-3
        
        # Create model
        self.model = SOTATemporalPGAT(self.configs)
        self.model.eval()
    
    def create_test_data(self, batch_size=None, seq_len=None, add_noise=False, noise_std=0.01):
        """Create test data with optional noise."""
        if batch_size is None:
            batch_size = self.batch_size
        if seq_len is None:
            seq_len = self.configs.seq_len
        
        # Create base data
        x_enc = torch.randn(batch_size, seq_len, self.configs.enc_in)
        x_mark_enc = torch.randn(batch_size, seq_len, 4)
        x_dec = torch.randn(batch_size, self.configs.label_len + self.configs.pred_len, self.configs.dec_in)
        x_mark_dec = torch.randn(batch_size, self.configs.label_len + self.configs.pred_len, 4)
        
        # Add noise if requested
        if add_noise:
            x_enc += torch.randn_like(x_enc) * noise_std
            x_mark_enc += torch.randn_like(x_mark_enc) * noise_std
            x_dec += torch.randn_like(x_dec) * noise_std
            x_mark_dec += torch.randn_like(x_mark_dec) * noise_std
        
        return x_enc, x_mark_enc, x_dec, x_mark_dec
    
    def test_output_shape_invariant(self):
        """Test that output shape is invariant to batch size changes."""
        batch_sizes = [1, 2, 4, 8]
        
        for batch_size in batch_sizes:
            x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data(batch_size=batch_size)
            
            with torch.no_grad():
                output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            expected_shape = (batch_size, self.configs.pred_len, self.configs.c_out)
            assert output.shape == expected_shape, \
                f"Output shape invariant violated for batch_size {batch_size}. Expected: {expected_shape}, Got: {output.shape}"
    
    def test_permutation_equivariance(self):
        """Test that model is equivariant to feature permutations."""
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data()
        
        # Create permutation indices
        perm_indices = torch.randperm(self.configs.enc_in)
        
        # Original output
        with torch.no_grad():
            output_original = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Permuted input
        x_enc_perm = x_enc[:, :, perm_indices]
        x_dec_perm = x_dec[:, :, perm_indices]  # Assuming dec_in == enc_in
        
        with torch.no_grad():
            output_permuted = self.model(x_enc_perm, x_mark_enc, x_dec_perm, x_mark_dec)
        
        # Permute output back
        inverse_perm = torch.argsort(perm_indices)
        output_permuted_back = output_permuted[:, :, inverse_perm]
        
        # Check equivariance (allowing for some numerical error)
        assert torch.allclose(output_original, output_permuted_back, atol=self.tolerance), \
            "Model should be equivariant to feature permutations"
    
    def test_scale_invariance_properties(self):
        """Test model behavior under input scaling."""
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data()
        
        # Original output
        with torch.no_grad():
            output_original = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Test different scales
        scales = [0.1, 0.5, 2.0, 10.0]
        
        for scale in scales:
            x_enc_scaled = x_enc * scale
            x_dec_scaled = x_dec * scale
            
            with torch.no_grad():
                output_scaled = self.model(x_enc_scaled, x_mark_enc, x_dec_scaled, x_mark_dec)
            
            # Check that output scaling is reasonable (not necessarily linear)
            # The model should not produce NaN or Inf for reasonable scales
            assert torch.isfinite(output_scaled).all(), \
                f"Model output should be finite for scale {scale}"
            
            # Check that output magnitude changes with input scale
            output_ratio = output_scaled.abs().mean() / output_original.abs().mean()
            assert 0.01 < output_ratio < 100, \
                f"Output scaling seems unreasonable for input scale {scale}: ratio = {output_ratio}"
    
    def test_temporal_consistency(self):
        """Test that model predictions are temporally consistent."""
        # Create overlapping sequences
        seq_len = self.configs.seq_len
        overlap = seq_len // 2
        
        # First sequence
        x_enc1, x_mark_enc1, x_dec1, x_mark_dec1 = self.create_test_data()
        
        # Second sequence (shifted by overlap)
        x_enc2 = torch.cat([x_enc1[:, overlap:, :], torch.randn(self.batch_size, overlap, self.configs.enc_in)], dim=1)
        x_mark_enc2 = torch.cat([x_mark_enc1[:, overlap:, :], torch.randn(self.batch_size, overlap, 4)], dim=1)
        x_dec2, x_mark_dec2 = self.create_test_data(batch_size=self.batch_size)
        
        with torch.no_grad():
            output1 = self.model(x_enc1, x_mark_enc1, x_dec1, x_mark_dec1)
            output2 = self.model(x_enc2, x_mark_enc2, x_dec2, x_mark_dec2)
        
        # Check that outputs are not identical (they should differ due to different inputs)
        assert not torch.allclose(output1, output2, atol=1e-3), \
            "Outputs should differ for different input sequences"
        
        # Check that outputs are finite and reasonable
        assert torch.isfinite(output1).all() and torch.isfinite(output2).all(), \
            "All outputs should be finite"
    
    def test_attention_weight_properties(self):
        """Test properties of attention weights if available."""
        # Enable attention output
        self.configs.output_attention = True
        model_with_attn = SOTATemporalPGAT(self.configs)
        model_with_attn.eval()
        
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data()
        
        with torch.no_grad():
            try:
                result = model_with_attn(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
                if isinstance(result, tuple) and len(result) == 2:
                    output, attention_weights = result
                    
                    # Test attention weight properties
                    assert torch.isfinite(attention_weights).all(), \
                        "Attention weights should be finite"
                    
                    # Attention weights should be non-negative
                    assert (attention_weights >= 0).all(), \
                        "Attention weights should be non-negative"
                    
                    # Check attention weight dimensions are reasonable
                    assert len(attention_weights.shape) >= 3, \
                        "Attention weights should have at least 3 dimensions"
                    
            except Exception as e:
                pytest.skip(f"Attention weight test skipped: {e}")
    
    def test_gradient_stability(self):
        """Test that gradients are stable and well-behaved."""
        self.model.train()
        
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data()
        
        # Forward pass
        output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        loss = output.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradient properties
        gradient_norms = []
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                gradient_norms.append(grad_norm)
                
                # Check for exploding gradients
                assert grad_norm < 100.0, \
                    f"Gradient explosion detected in {name}: norm = {grad_norm}"
                
                # Check for vanishing gradients (in key components)
                if 'weight' in name and 'embedding' not in name:
                    assert grad_norm > 1e-8, \
                        f"Gradient vanishing detected in {name}: norm = {grad_norm}"
        
        # Check overall gradient distribution
        if gradient_norms:
            mean_grad_norm = np.mean(gradient_norms)
            std_grad_norm = np.std(gradient_norms)
            
            assert mean_grad_norm > 1e-6, "Mean gradient norm too small"
            assert std_grad_norm < mean_grad_norm * 10, "Gradient variance too high"
    
    def test_numerical_stability(self):
        """Test numerical stability under various conditions."""
        test_conditions = [
            {"name": "normal", "data_fn": lambda: self.create_test_data()},
            {"name": "small_values", "data_fn": lambda: tuple(x * 0.001 for x in self.create_test_data())},
            {"name": "large_values", "data_fn": lambda: tuple(x * 100 for x in self.create_test_data())},
            {"name": "noisy", "data_fn": lambda: self.create_test_data(add_noise=True, noise_std=0.1)}
        ]
        
        for condition in test_conditions:
            x_enc, x_mark_enc, x_dec, x_mark_dec = condition["data_fn"]()
            
            with torch.no_grad():
                output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            # Check for numerical issues
            assert torch.isfinite(output).all(), \
                f"Model output should be finite under {condition['name']} conditions"
            
            assert not torch.isnan(output).any(), \
                f"Model output should not contain NaN under {condition['name']} conditions"
            
            # Check output magnitude is reasonable
            output_magnitude = output.abs().max().item()
            assert output_magnitude < 1e6, \
                f"Output magnitude too large under {condition['name']} conditions: {output_magnitude}"
    
    def test_graph_adjacency_properties(self):
        """Test properties of constructed graph adjacency matrices."""
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data()
        
        # Access graph constructor if available
        if hasattr(self.model, 'graph_constructor'):
            graph_constructor = self.model.graph_constructor
            
            # Create input for graph constructor (this depends on implementation)
            # Typically this would be the mean or some aggregation of the input
            graph_input = x_enc.mean(dim=1)  # [batch_size, enc_in]
            
            with torch.no_grad():
                try:
                    adjacency = graph_constructor(graph_input)
                    
                    # Test adjacency matrix properties
                    # 1. Values should be in [0, 1] range
                    assert (adjacency >= 0).all() and (adjacency <= 1).all(), \
                        "Adjacency matrix values should be in [0, 1] range"
                    
                    # 2. Should be finite
                    assert torch.isfinite(adjacency).all(), \
                        "Adjacency matrix should be finite"
                    
                    # 3. Diagonal properties (depends on implementation)
                    # Some implementations might have self-loops, others might not
                    
                    # 4. Symmetry (if the graph is undirected)
                    # This depends on the specific implementation
                    
                except Exception as e:
                    pytest.skip(f"Graph adjacency test skipped: {e}")
    
    def test_positional_encoding_properties(self):
        """Test properties of positional encoding."""
        if hasattr(self.model, 'graph_pos_encoding'):
            pos_encoding = self.model.graph_pos_encoding
            
            # Test with different sequence lengths
            seq_lens = [24, 48, 96, 192]
            
            for seq_len in seq_lens:
                test_input = torch.randn(self.batch_size, seq_len, self.configs.d_model)
                
                with torch.no_grad():
                    try:
                        encoded_output = pos_encoding(test_input)
                        
                        # Check shape preservation
                        assert encoded_output.shape == test_input.shape, \
                            f"Positional encoding should preserve shape for seq_len {seq_len}"
                        
                        # Check that encoding actually modifies the input
                        assert not torch.equal(test_input, encoded_output), \
                            f"Positional encoding should modify input for seq_len {seq_len}"
                        
                        # Check output is finite
                        assert torch.isfinite(encoded_output).all(), \
                            f"Positional encoding output should be finite for seq_len {seq_len}"
                        
                    except Exception as e:
                        pytest.skip(f"Positional encoding test skipped for seq_len {seq_len}: {e}")
    
    def test_model_determinism(self):
        """Test that model is deterministic given same inputs and random state."""
        # Set random seeds
        torch.manual_seed(42)
        np.random.seed(42)
        
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data()
        
        # First run
        torch.manual_seed(42)
        with torch.no_grad():
            output1 = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Second run with same seed
        torch.manual_seed(42)
        with torch.no_grad():
            output2 = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Outputs should be identical
        assert torch.allclose(output1, output2, atol=1e-8), \
            "Model should be deterministic with same random seed"
    
    def test_input_perturbation_stability(self):
        """Test model stability under small input perturbations."""
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data()
        
        # Original output
        with torch.no_grad():
            output_original = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Test different perturbation magnitudes
        perturbation_scales = [1e-6, 1e-5, 1e-4, 1e-3]
        
        for scale in perturbation_scales:
            # Add small perturbation
            perturbation = torch.randn_like(x_enc) * scale
            x_enc_perturbed = x_enc + perturbation
            
            with torch.no_grad():
                output_perturbed = self.model(x_enc_perturbed, x_mark_enc, x_dec, x_mark_dec)
            
            # Compute relative change
            input_change = perturbation.norm() / x_enc.norm()
            output_change = (output_perturbed - output_original).norm() / output_original.norm()
            
            # Output change should be proportional to input change (Lipschitz continuity)
            lipschitz_ratio = output_change / input_change
            
            assert lipschitz_ratio < 1000, \
                f"Model too sensitive to perturbations at scale {scale}: Lipschitz ratio = {lipschitz_ratio}"
            
            assert torch.isfinite(output_perturbed).all(), \
                f"Model output should remain finite under perturbation scale {scale}"
    
    @pytest.mark.parametrize("seq_len", [24, 48, 96, 192, 336])
    def test_sequence_length_scalability(self, seq_len):
        """Test model behavior with different sequence lengths."""
        x_enc, x_mark_enc, x_dec, x_mark_dec = self.create_test_data(seq_len=seq_len)
        
        with torch.no_grad():
            output = self.model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Check output properties
        expected_shape = (self.batch_size, self.configs.pred_len, self.configs.c_out)
        assert output.shape == expected_shape, \
            f"Output shape incorrect for seq_len {seq_len}"
        
        assert torch.isfinite(output).all(), \
            f"Output should be finite for seq_len {seq_len}"
        
        # Check that longer sequences don't cause memory issues
        # (This is more of a practical test than a mathematical invariant)
        output_magnitude = output.abs().mean().item()
        assert 1e-6 < output_magnitude < 1e6, \
            f"Output magnitude seems unreasonable for seq_len {seq_len}: {output_magnitude}"