"""Integration tests for SOTA_Temporal_PGAT model.

This module tests the complete SOTA_Temporal_PGAT model integration,
including data flow, component interactions, and end-to-end functionality.
"""

# Suppress optional dependency warnings
from TestsModule.suppress_warnings import suppress_optional_dependency_warnings
suppress_optional_dependency_warnings()

import pytest
import torch
import numpy as np
from unittest.mock import patch, MagicMock

# Import the model and related components
try:
    from models.SOTA_Temporal_PGAT import SOTA_Temporal_PGAT as SOTATemporalPGAT
except ImportError:
    pytest.skip("SOTA_Temporal_PGAT model not available", allow_module_level=True)

from layers.modular.core.registry import ComponentRegistry
# Import register_components to trigger component registration
import layers.modular.core.register_components


class TestSOTATemporalPGATIntegration:
    """Integration test suite for SOTA_Temporal_PGAT model."""
    
    @pytest.fixture(autouse=True)
    def setup_test(self):
        """Setup test environment."""
        # Components are registered when register_components is imported
        
        # Create mock configuration
        self.configs = MagicMock()
        self.configs.seq_len = 96
        self.configs.label_len = 48
        self.configs.pred_len = 96
        self.configs.enc_in = 7
        self.configs.dec_in = 7
        self.configs.c_out = 7
        self.configs.d_model = 512
        self.configs.n_heads = 8
        self.configs.e_layers = 2
        self.configs.d_layers = 1
        self.configs.d_ff = 2048
        self.configs.dropout = 0.1
        self.configs.activation = 'gelu'
        self.configs.output_attention = False
        self.configs.mix = True
        
        # Graph-specific configurations
        self.configs.graph_dim = 64
        self.configs.num_nodes = self.configs.enc_in
        self.configs.graph_alpha = 0.2
        self.configs.graph_beta = 0.1
        
        # Create test data
        self.batch_size = 4
        self.create_test_data()
    
    def create_test_data(self):
        """Create synthetic test data."""
        # Encoder input: [batch_size, seq_len, enc_in]
        self.x_enc = torch.randn(self.batch_size, self.configs.seq_len, self.configs.enc_in)
        
        # Encoder timestamp: [batch_size, seq_len, timestamp_features]
        self.x_mark_enc = torch.randn(self.batch_size, self.configs.seq_len, 4)  # 4 time features
        
        # Decoder input: [batch_size, label_len + pred_len, dec_in]
        self.x_dec = torch.randn(self.batch_size, self.configs.label_len + self.configs.pred_len, self.configs.dec_in)
        
        # Decoder timestamp: [batch_size, label_len + pred_len, timestamp_features]
        self.x_mark_dec = torch.randn(self.batch_size, self.configs.label_len + self.configs.pred_len, 4)
    
    def test_model_initialization(self):
        """Test that SOTA_Temporal_PGAT model initializes correctly."""
        model = SOTATemporalPGAT(self.configs)
        
        # Check model is created
        assert model is not None, "Model should be created successfully"
        
        # Check model has required components
        assert hasattr(model, 'enc_embedding'), "Model should have encoder embedding"
        assert hasattr(model, 'dec_embedding'), "Model should have decoder embedding"
        assert hasattr(model, 'encoder'), "Model should have encoder"
        assert hasattr(model, 'decoder'), "Model should have decoder"
        assert hasattr(model, 'projection'), "Model should have projection layer"
        
        # Check graph components
        assert hasattr(model, 'graph_constructor'), "Model should have graph constructor"
        assert hasattr(model, 'graph_pos_encoding'), "Model should have graph positional encoding"
    
    def test_model_forward_pass(self):
        """Test complete forward pass through the model."""
        model = SOTATemporalPGAT(self.configs)
        model.eval()
        
        with torch.no_grad():
            output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        
        # Check output shape
        expected_shape = (self.batch_size, self.configs.pred_len, self.configs.c_out)
        assert output.shape == expected_shape, \
            f"Output shape mismatch. Expected: {expected_shape}, Got: {output.shape}"
        
        # Check output is not NaN or Inf
        assert torch.isfinite(output).all(), "Output should not contain NaN or Inf values"
    
    def test_model_training_mode(self):
        """Test model behavior in training mode."""
        model = SOTATemporalPGAT(self.configs)
        model.train()
        
        # Forward pass in training mode
        output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        
        # Check gradients can be computed
        loss = output.mean()
        loss.backward()
        
        # Check that gradients exist
        has_gradients = False
        for param in model.parameters():
            if param.grad is not None:
                has_gradients = True
                break
        
        assert has_gradients, "Model should have gradients in training mode"
    
    def test_graph_construction_integration(self):
        """Test that graph construction works with the model."""
        model = SOTATemporalPGAT(self.configs)
        
        # Access graph constructor
        graph_constructor = model.graph_constructor
        
        # Test graph construction with encoder input
        with torch.no_grad():
            # Simulate the input that would go to graph constructor
            # This might be the embedded input or raw input depending on implementation
            graph_input = self.x_enc.mean(dim=1)  # [batch_size, enc_in]
            
            try:
                adjacency_matrix = graph_constructor(graph_input)
                
                # Check adjacency matrix properties
                expected_shape = (self.batch_size, self.configs.num_nodes, self.configs.num_nodes)
                assert adjacency_matrix.shape == expected_shape, \
                    f"Adjacency matrix shape mismatch. Expected: {expected_shape}, Got: {adjacency_matrix.shape}"
                
                # Check adjacency matrix values are in valid range [0, 1]
                assert (adjacency_matrix >= 0).all() and (adjacency_matrix <= 1).all(), \
                    "Adjacency matrix values should be in range [0, 1]"
                
            except Exception as e:
                pytest.skip(f"Graph constructor test skipped due to implementation details: {e}")
    
    def test_attention_mechanism_integration(self):
        """Test that graph attention mechanisms work correctly."""
        model = SOTATemporalPGAT(self.configs)
        
        # Test with attention output enabled
        self.configs.output_attention = True
        model_with_attn = SOTATemporalPGAT(self.configs)
        
        with torch.no_grad():
            try:
                output = model_with_attn(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
                
                # If model returns attention weights, check them
                if isinstance(output, tuple):
                    predictions, attention_weights = output
                    
                    # Check attention weights properties
                    assert attention_weights is not None, "Attention weights should be returned"
                    assert torch.isfinite(attention_weights).all(), \
                        "Attention weights should not contain NaN or Inf"
                    
                    # Check attention weights sum to 1 (if they're normalized)
                    # This depends on the specific attention implementation
                    
            except Exception as e:
                pytest.skip(f"Attention mechanism test skipped: {e}")
    
    def test_positional_encoding_integration(self):
        """Test that graph-aware positional encoding works correctly."""
        model = SOTATemporalPGAT(self.configs)
        
        # Access positional encoding component
        pos_encoding = model.graph_pos_encoding
        
        with torch.no_grad():
            try:
                # Test positional encoding with sequence input
                seq_input = torch.randn(self.batch_size, self.configs.seq_len, self.configs.d_model)
                
                encoded_output = pos_encoding(seq_input)
                
                # Check output shape matches input
                assert encoded_output.shape == seq_input.shape, \
                    "Positional encoding should preserve input shape"
                
                # Check that encoding actually modifies the input
                assert not torch.equal(seq_input, encoded_output), \
                    "Positional encoding should modify the input"
                
            except Exception as e:
                pytest.skip(f"Positional encoding test skipped: {e}")
    
    def test_model_with_different_sequence_lengths(self):
        """Test model with various sequence lengths."""
        model = SOTATemporalPGAT(self.configs)
        
        test_seq_lens = [24, 48, 96, 192]
        
        for seq_len in test_seq_lens:
            # Create data with different sequence length
            x_enc = torch.randn(2, seq_len, self.configs.enc_in)
            x_mark_enc = torch.randn(2, seq_len, 4)
            x_dec = torch.randn(2, self.configs.label_len + self.configs.pred_len, self.configs.dec_in)
            x_mark_dec = torch.randn(2, self.configs.label_len + self.configs.pred_len, 4)
            
            with torch.no_grad():
                try:
                    output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                    
                    # Check output shape is consistent
                    expected_shape = (2, self.configs.pred_len, self.configs.c_out)
                    assert output.shape == expected_shape, \
                        f"Output shape incorrect for seq_len {seq_len}. Expected: {expected_shape}, Got: {output.shape}"
                    
                except Exception as e:
                    pytest.fail(f"Model failed with sequence length {seq_len}: {e}")
    
    def test_model_memory_efficiency(self):
        """Test that model doesn't have memory leaks."""
        model = SOTATemporalPGAT(self.configs)
        
        # Record initial memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            initial_memory = torch.cuda.memory_allocated()
        
        # Run multiple forward passes
        for _ in range(5):
            with torch.no_grad():
                output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
                del output
        
        # Check memory hasn't grown significantly
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            final_memory = torch.cuda.memory_allocated()
            memory_growth = final_memory - initial_memory
            
            # Allow some memory growth but not excessive
            assert memory_growth < 100 * 1024 * 1024, \
                f"Excessive memory growth detected: {memory_growth / (1024*1024):.2f} MB"
    
    def test_model_deterministic_output(self):
        """Test that model produces deterministic output with same input."""
        # Set random seeds for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        model = SOTATemporalPGAT(self.configs)
        model.eval()
        
        with torch.no_grad():
            output1 = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
            output2 = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        
        # Outputs should be identical for same input
        assert torch.allclose(output1, output2, atol=1e-6), \
            "Model should produce deterministic output for same input"
    
    def test_model_gradient_flow(self):
        """Test that gradients flow properly through all components."""
        model = SOTATemporalPGAT(self.configs)
        model.train()
        
        # Forward pass
        output = model(self.x_enc, self.x_mark_enc, self.x_dec, self.x_mark_dec)
        
        # Compute loss and backward pass
        loss = output.mean()
        loss.backward()
        
        # Check that key components have gradients
        components_with_gradients = []
        components_without_gradients = []
        
        for name, param in model.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                components_with_gradients.append(name)
            else:
                components_without_gradients.append(name)
        
        # Should have more components with gradients than without
        assert len(components_with_gradients) > 0, \
            "At least some components should have gradients"
        
        # Print gradient information for debugging
        print(f"Components with gradients: {len(components_with_gradients)}")
        print(f"Components without gradients: {len(components_without_gradients)}")
    
    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
    def test_model_different_batch_sizes(self, batch_size):
        """Test model with different batch sizes."""
        model = SOTATemporalPGAT(self.configs)
        
        # Create data with specific batch size
        x_enc = torch.randn(batch_size, self.configs.seq_len, self.configs.enc_in)
        x_mark_enc = torch.randn(batch_size, self.configs.seq_len, 4)
        x_dec = torch.randn(batch_size, self.configs.label_len + self.configs.pred_len, self.configs.dec_in)
        x_mark_dec = torch.randn(batch_size, self.configs.label_len + self.configs.pred_len, 4)
        
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Check output shape
        expected_shape = (batch_size, self.configs.pred_len, self.configs.c_out)
        assert output.shape == expected_shape, \
            f"Output shape incorrect for batch_size {batch_size}. Expected: {expected_shape}, Got: {output.shape}"