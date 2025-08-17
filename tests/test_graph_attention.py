"""
Test suite for Graph Attention Network components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from layers.modular.attention.graph_attention import (
    GraphAttentionLayer, MultiGraphAttention, 
    construct_correlation_graph, construct_knn_graph
)


class TestGraphAttentionLayer:
    """Test cases for GraphAttentionLayer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, seq_len, d_model = 2, 10, 64
        x = torch.randn(batch_size, seq_len, d_model)
        adj_matrix = torch.randint(0, 2, (batch_size, seq_len, seq_len)).float()
        return x, adj_matrix
    
    @pytest.mark.smoke
    def test_init(self):
        """Test GraphAttentionLayer initialization."""
        layer = GraphAttentionLayer(d_model=64, n_heads=8)
        assert layer.d_model == 64
        assert layer.n_heads == 8
        assert layer.d_k == 8  # 64 // 8
        
    @pytest.mark.smoke
    def test_forward_shape(self, sample_data):
        """Test forward pass output shapes."""
        x, adj_matrix = sample_data
        layer = GraphAttentionLayer(d_model=64, n_heads=8)
        
        output, attention = layer(x, adj_matrix)
        
        assert output.shape == x.shape
        assert attention.shape == (x.size(0), x.size(1), x.size(1))
        
    @pytest.mark.extended
    def test_forward_without_adjacency(self, sample_data):
        """Test forward pass without adjacency matrix."""
        x, _ = sample_data
        layer = GraphAttentionLayer(d_model=64, n_heads=8)
        
        output, attention = layer(x)
        
        assert output.shape == x.shape
        assert attention.shape == (x.size(0), x.size(1), x.size(1))
        
    @pytest.mark.extended
    def test_concat_vs_average(self, sample_data):
        """Test concatenation vs averaging of multi-head outputs."""
        x, adj_matrix = sample_data
        
        # Concatenation mode
        layer_concat = GraphAttentionLayer(d_model=64, n_heads=8, concat=True)
        output_concat, _ = layer_concat(x, adj_matrix)
        assert output_concat.shape[-1] == 64
        
        # Averaging mode
        layer_avg = GraphAttentionLayer(d_model=64, n_heads=8, concat=False)
        output_avg, _ = layer_avg(x, adj_matrix)
        assert output_avg.shape[-1] == 64
        
    @pytest.mark.extended
    def test_attention_mask(self, sample_data):
        """Test attention masking functionality."""
        x, adj_matrix = sample_data
        layer = GraphAttentionLayer(d_model=64, n_heads=8)
        
        # Create attention mask
        attn_mask = torch.ones(x.size(0), x.size(1), x.size(1))
        attn_mask[:, :, -1] = 0  # Mask last position
        
        output, attention = layer(x, adj_matrix, attn_mask)
        
        # Check that masked positions have near-zero attention
        assert torch.allclose(attention[:, :, -1], torch.zeros_like(attention[:, :, -1]), atol=1e-6)
        
    @pytest.mark.extended
    def test_gradient_flow(self, sample_data):
        """Test gradient flow through the layer."""
        x, adj_matrix = sample_data
        x.requires_grad_(True)
        
        layer = GraphAttentionLayer(d_model=64, n_heads=8)
        output, _ = layer(x, adj_matrix)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestMultiGraphAttention:
    """Test cases for MultiGraphAttention."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, seq_len, d_model = 2, 10, 64
        x = torch.randn(batch_size, seq_len, d_model)
        adj_matrix = torch.randint(0, 2, (batch_size, seq_len, seq_len)).float()
        return x, adj_matrix
    
    @pytest.mark.smoke
    def test_init(self):
        """Test MultiGraphAttention initialization."""
        layer = MultiGraphAttention(d_model=64, n_heads=8)
        assert hasattr(layer, 'gat')
        assert hasattr(layer, 'norm')
        assert hasattr(layer, 'dropout')
        
    @pytest.mark.smoke
    def test_forward_shape(self, sample_data):
        """Test forward pass output shapes."""
        x, adj_matrix = sample_data
        layer = MultiGraphAttention(d_model=64, n_heads=8)
        
        output, attention = layer(x, adj_matrix)
        
        assert output.shape == x.shape
        assert attention.shape == (x.size(0), x.size(1), x.size(1))
        
    @pytest.mark.extended
    def test_residual_connection(self, sample_data):
        """Test residual connection functionality."""
        x, adj_matrix = sample_data
        layer = MultiGraphAttention(d_model=64, n_heads=8)
        
        # Set dropout to 0 for deterministic testing
        layer.dropout.p = 0.0
        
        output, _ = layer(x, adj_matrix)
        
        # Output should be different from input due to attention
        assert not torch.allclose(output, x, atol=1e-6)


class TestGraphConstruction:
    """Test cases for graph construction utilities."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, seq_len, num_features = 2, 20, 10
        x = torch.randn(batch_size, seq_len, num_features)
        return x
    
    @pytest.mark.smoke
    def test_correlation_graph_shape(self, sample_data):
        """Test correlation graph construction output shape."""
        x = sample_data
        adj_matrix = construct_correlation_graph(x, threshold=0.5)
        
        expected_shape = (x.size(0), x.size(2), x.size(2))
        assert adj_matrix.shape == expected_shape
        
    @pytest.mark.smoke
    def test_knn_graph_shape(self, sample_data):
        """Test k-NN graph construction output shape."""
        x = sample_data
        k = 3
        adj_matrix = construct_knn_graph(x, k=k)
        
        expected_shape = (x.size(0), x.size(1), x.size(1))
        assert adj_matrix.shape == expected_shape
        
    @pytest.mark.extended
    def test_correlation_graph_properties(self, sample_data):
        """Test correlation graph properties."""
        x = sample_data
        adj_matrix = construct_correlation_graph(x, threshold=0.5)
        
        # Should be symmetric
        assert torch.allclose(adj_matrix, adj_matrix.transpose(-1, -2))
        
        # Should be binary (0 or 1)
        unique_values = torch.unique(adj_matrix)
        assert len(unique_values) <= 2
        assert all(val in [0.0, 1.0] for val in unique_values.tolist())
        
    @pytest.mark.extended
    def test_knn_graph_properties(self, sample_data):
        """Test k-NN graph properties."""
        x = sample_data
        k = 3
        adj_matrix = construct_knn_graph(x, k=k)
        
        # Should be symmetric
        assert torch.allclose(adj_matrix, adj_matrix.transpose(-1, -2))
        
        # Should be binary (0 or 1)
        unique_values = torch.unique(adj_matrix)
        assert len(unique_values) <= 2
        assert all(val in [0.0, 1.0] for val in unique_values.tolist())
        
        # Each node should have at most k neighbors (excluding self)
        row_sums = adj_matrix.sum(dim=-1)
        assert torch.all(row_sums <= 2 * k)  # *2 because of symmetry
        
    @pytest.mark.extended
    def test_correlation_threshold_effect(self, sample_data):
        """Test effect of correlation threshold on graph density."""
        x = sample_data
        
        adj_low = construct_correlation_graph(x, threshold=0.1)
        adj_high = construct_correlation_graph(x, threshold=0.9)
        
        # Lower threshold should result in denser graph
        density_low = adj_low.sum() / adj_low.numel()
        density_high = adj_high.sum() / adj_high.numel()
        
        assert density_low >= density_high
        
    @pytest.mark.extended
    def test_knn_k_effect(self, sample_data):
        """Test effect of k parameter on graph density."""
        x = sample_data
        
        adj_k3 = construct_knn_graph(x, k=3)
        adj_k5 = construct_knn_graph(x, k=5)
        
        # Higher k should result in denser graph
        density_k3 = adj_k3.sum() / adj_k3.numel()
        density_k5 = adj_k5.sum() / adj_k5.numel()
        
        assert density_k5 >= density_k3


class TestGraphAttentionIntegration:
    """Integration tests for graph attention components."""
    
    @pytest.mark.extended
    def test_end_to_end_pipeline(self):
        """Test complete pipeline from data to output."""
        batch_size, seq_len, d_model = 2, 15, 64
        
        # Create sample time series data
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Construct graph
        adj_matrix = construct_correlation_graph(x, threshold=0.3)
        
        # Apply graph attention
        layer = MultiGraphAttention(d_model=d_model, n_heads=8)
        output, attention = layer(x, adj_matrix)
        
        # Verify output properties
        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert attention.shape == (batch_size, seq_len, seq_len)
        
    @pytest.mark.extended
    def test_different_graph_types(self):
        """Test with different graph construction methods."""
        batch_size, seq_len, d_model = 2, 12, 64
        x = torch.randn(batch_size, seq_len, d_model)
        
        layer = MultiGraphAttention(d_model=d_model, n_heads=8)
        
        # Test with correlation graph
        adj_corr = construct_correlation_graph(x, threshold=0.4)
        output_corr, _ = layer(x, adj_corr)
        
        # Test with k-NN graph
        adj_knn = construct_knn_graph(x, k=4)
        output_knn, _ = layer(x, adj_knn)
        
        # Test without graph
        output_none, _ = layer(x)
        
        # All should produce valid outputs
        for output in [output_corr, output_knn, output_none]:
            assert output.shape == x.shape
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            
    @pytest.mark.extended
    def test_batch_consistency(self):
        """Test consistency across different batch sizes."""
        seq_len, d_model = 10, 64
        
        layer = MultiGraphAttention(d_model=d_model, n_heads=8)
        
        # Test with different batch sizes
        for batch_size in [1, 2, 4]:
            x = torch.randn(batch_size, seq_len, d_model)
            adj_matrix = construct_correlation_graph(x, threshold=0.5)
            
            output, attention = layer(x, adj_matrix)
            
            assert output.shape == (batch_size, seq_len, d_model)
            assert attention.shape == (batch_size, seq_len, seq_len)


if __name__ == "__main__":
    pytest.main([__file__])