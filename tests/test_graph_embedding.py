"""
Test suite for Graph Embedding components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from layers.GraphEmbed import (
    GraphPositionalEncoding, SpatialEmbedding, GraphTimeSeriesEmbedding,
    AdaptiveGraphEmbedding, GraphFeatureEmbedding
)


class TestGraphPositionalEncoding:
    """Test cases for GraphPositionalEncoding."""
    
    @pytest.mark.smoke
    def test_init(self):
        """Test GraphPositionalEncoding initialization."""
        pe = GraphPositionalEncoding(d_model=64, max_len=100)
        assert pe.pe.shape == (100, 1, 64)
        
    @pytest.mark.smoke
    def test_forward_shape(self):
        """Test forward pass output shape."""
        pe = GraphPositionalEncoding(d_model=64, max_len=100)
        x = torch.randn(20, 2, 64)  # (seq_len, batch_size, d_model)
        
        output = pe(x)
        assert output.shape == x.shape
        
    @pytest.mark.extended
    def test_positional_encoding_values(self):
        """Test that positional encoding values are reasonable."""
        pe = GraphPositionalEncoding(d_model=64, max_len=100, dropout=0.0)
        x = torch.zeros(20, 2, 64)
        
        output = pe(x)
        
        # Output should not be all zeros (due to positional encoding)
        assert not torch.allclose(output, torch.zeros_like(output))
        
        # Should be bounded (due to sin/cos)
        assert torch.all(torch.abs(output) <= 10)  # Reasonable bound


class TestSpatialEmbedding:
    """Test cases for SpatialEmbedding."""
    
    @pytest.mark.smoke
    def test_init(self):
        """Test SpatialEmbedding initialization."""
        se = SpatialEmbedding(num_nodes=10, d_model=64)
        assert se.spatial_emb.shape == (10, 64)
        
    @pytest.mark.smoke
    def test_forward_shape(self):
        """Test forward pass output shape."""
        se = SpatialEmbedding(num_nodes=10, d_model=64)
        batch_size = 3
        
        output = se(batch_size)
        assert output.shape == (batch_size, 10, 64)
        
    @pytest.mark.extended
    def test_batch_consistency(self):
        """Test consistency across different batch sizes."""
        se = SpatialEmbedding(num_nodes=10, d_model=64, dropout=0.0)
        
        output1 = se(1)
        output2 = se(2)
        
        # First batch should be identical
        assert torch.allclose(output1[0], output2[0])


class TestGraphTimeSeriesEmbedding:
    """Test cases for GraphTimeSeriesEmbedding."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # 3D data: [B, L, C]
        data_3d = torch.randn(2, 10, 5)
        # 4D data: [B, L, N, C]
        data_4d = torch.randn(2, 10, 8, 5)
        return data_3d, data_4d
    
    @pytest.mark.smoke
    def test_init_without_spatial(self):
        """Test initialization without spatial embedding."""
        embed = GraphTimeSeriesEmbedding(c_in=5, d_model=64)
        assert embed.spatial_embed is None
        
    @pytest.mark.smoke
    def test_init_with_spatial(self):
        """Test initialization with spatial embedding."""
        embed = GraphTimeSeriesEmbedding(c_in=5, d_model=64, num_nodes=10)
        assert embed.spatial_embed is not None
        
    @pytest.mark.smoke
    def test_forward_3d_shape(self, sample_data):
        """Test forward pass with 3D input."""
        data_3d, _ = sample_data
        embed = GraphTimeSeriesEmbedding(c_in=5, d_model=64)
        
        output = embed(data_3d)
        assert output.shape == (2, 10, 64)
        
    @pytest.mark.smoke
    def test_forward_4d_shape(self, sample_data):
        """Test forward pass with 4D input."""
        _, data_4d = sample_data
        embed = GraphTimeSeriesEmbedding(c_in=5, d_model=64, num_nodes=8)
        
        output = embed(data_4d)
        assert output.shape == (2, 10, 8, 64)
        
    @pytest.mark.extended
    def test_fixed_vs_learned_embedding(self, sample_data):
        """Test fixed vs learned temporal embedding."""
        data_3d, _ = sample_data
        
        # Fixed embedding
        embed_fixed = GraphTimeSeriesEmbedding(c_in=5, d_model=64, embed_type='fixed')
        output_fixed = embed_fixed(data_3d)
        
        # Learned embedding
        embed_learned = GraphTimeSeriesEmbedding(c_in=5, d_model=64, embed_type='learned')
        temporal_idx = torch.arange(10).unsqueeze(0).expand(2, -1)
        output_learned = embed_learned(data_3d, temporal_idx)
        
        assert output_fixed.shape == output_learned.shape
        # Should produce different outputs
        assert not torch.allclose(output_fixed, output_learned, atol=1e-6)
        
    @pytest.mark.extended
    def test_spatial_embedding_effect(self, sample_data):
        """Test effect of spatial embedding."""
        _, data_4d = sample_data
        
        # Without spatial embedding
        embed_no_spatial = GraphTimeSeriesEmbedding(c_in=5, d_model=64)
        output_no_spatial = embed_no_spatial(data_4d)
        
        # With spatial embedding
        embed_with_spatial = GraphTimeSeriesEmbedding(c_in=5, d_model=64, num_nodes=8)
        output_with_spatial = embed_with_spatial(data_4d)
        
        assert output_no_spatial.shape == output_with_spatial.shape
        # Should produce different outputs
        assert not torch.allclose(output_no_spatial, output_with_spatial, atol=1e-6)


class TestAdaptiveGraphEmbedding:
    """Test cases for AdaptiveGraphEmbedding."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return torch.randn(2, 10, 8, 5)  # [B, L, N, C]
    
    @pytest.mark.smoke
    def test_init(self):
        """Test AdaptiveGraphEmbedding initialization."""
        embed = AdaptiveGraphEmbedding(c_in=5, d_model=64, num_nodes=8)
        assert hasattr(embed, 'temporal_weight')
        assert hasattr(embed, 'spatial_weight')
        assert hasattr(embed, 'gate')
        
    @pytest.mark.smoke
    def test_forward_shape(self, sample_data):
        """Test forward pass output shape."""
        embed = AdaptiveGraphEmbedding(c_in=5, d_model=64, num_nodes=8)
        
        output = embed(sample_data)
        assert output.shape == (2, 10, 8, 64)
        
    @pytest.mark.extended
    def test_adaptive_weighting(self, sample_data):
        """Test that adaptive weighting produces different outputs."""
        embed = AdaptiveGraphEmbedding(c_in=5, d_model=64, num_nodes=8)
        
        # Two different inputs should produce different gate weights
        data1 = sample_data
        data2 = torch.randn_like(sample_data) * 2  # Different scale
        
        output1 = embed(data1)
        output2 = embed(data2)
        
        # Should produce different outputs due to adaptive weighting
        assert not torch.allclose(output1, output2, atol=1e-6)


class TestGraphFeatureEmbedding:
    """Test cases for GraphFeatureEmbedding."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # 3D data: [B, L, C]
        data_3d = torch.randn(2, 10, 5)
        # 4D data: [B, L, N, C]
        data_4d = torch.randn(2, 10, 8, 5)
        # Adjacency matrix
        adj_matrix = torch.randint(0, 2, (2, 8, 8)).float()
        return data_3d, data_4d, adj_matrix
    
    @pytest.mark.smoke
    def test_init_with_graph_conv(self):
        """Test initialization with graph convolution."""
        embed = GraphFeatureEmbedding(c_in=5, d_model=64, use_graph_conv=True)
        assert hasattr(embed, 'graph_conv')
        assert hasattr(embed, 'norm')
        
    @pytest.mark.smoke
    def test_init_without_graph_conv(self):
        """Test initialization without graph convolution."""
        embed = GraphFeatureEmbedding(c_in=5, d_model=64, use_graph_conv=False)
        assert hasattr(embed, 'linear')
        
    @pytest.mark.smoke
    def test_forward_3d_shape(self, sample_data):
        """Test forward pass with 3D input."""
        data_3d, _, _ = sample_data
        embed = GraphFeatureEmbedding(c_in=5, d_model=64, use_graph_conv=False)
        
        output = embed(data_3d)
        assert output.shape == (2, 10, 64)
        
    @pytest.mark.smoke
    def test_forward_4d_shape(self, sample_data):
        """Test forward pass with 4D input and adjacency matrix."""
        _, data_4d, adj_matrix = sample_data
        embed = GraphFeatureEmbedding(c_in=5, d_model=64, use_graph_conv=True)
        
        output = embed(data_4d, adj_matrix)
        assert output.shape == (2, 10, 8, 64)
        
    @pytest.mark.extended
    def test_graph_conv_effect(self, sample_data):
        """Test effect of graph convolution."""
        _, data_4d, adj_matrix = sample_data
        
        # Without graph convolution
        embed_no_conv = GraphFeatureEmbedding(c_in=5, d_model=64, use_graph_conv=False)
        output_no_conv = embed_no_conv(data_4d)
        
        # With graph convolution
        embed_with_conv = GraphFeatureEmbedding(c_in=5, d_model=64, use_graph_conv=True)
        output_with_conv = embed_with_conv(data_4d, adj_matrix)
        
        assert output_no_conv.shape == output_with_conv.shape
        # Should produce different outputs
        assert not torch.allclose(output_no_conv, output_with_conv, atol=1e-6)
        
    @pytest.mark.extended
    def test_adjacency_matrix_formats(self, sample_data):
        """Test different adjacency matrix formats."""
        _, data_4d, _ = sample_data
        embed = GraphFeatureEmbedding(c_in=5, d_model=64, use_graph_conv=True)
        
        # 2D adjacency matrix [N, N]
        adj_2d = torch.randint(0, 2, (8, 8)).float()
        output_2d = embed(data_4d, adj_2d)
        
        # 3D adjacency matrix [B, N, N]
        adj_3d = torch.randint(0, 2, (2, 8, 8)).float()
        output_3d = embed(data_4d, adj_3d)
        
        assert output_2d.shape == output_3d.shape == (2, 10, 8, 64)


class TestGraphEmbeddingIntegration:
    """Integration tests for graph embedding components."""
    
    @pytest.mark.extended
    def test_end_to_end_pipeline(self):
        """Test complete embedding pipeline."""
        batch_size, seq_len, num_nodes, c_in = 2, 15, 10, 5
        d_model = 64
        
        # Create sample data
        x = torch.randn(batch_size, seq_len, num_nodes, c_in)
        adj_matrix = torch.randint(0, 2, (batch_size, num_nodes, num_nodes)).float()
        
        # Feature embedding
        feature_embed = GraphFeatureEmbedding(c_in=c_in, d_model=d_model, use_graph_conv=True)
        x_feat = feature_embed(x, adj_matrix)
        
        # Time series embedding
        ts_embed = GraphTimeSeriesEmbedding(c_in=d_model, d_model=d_model, num_nodes=num_nodes)
        x_final = ts_embed(x_feat)
        
        # Verify final output
        assert x_final.shape == (batch_size, seq_len, num_nodes, d_model)
        assert not torch.isnan(x_final).any()
        assert not torch.isinf(x_final).any()
        
    @pytest.mark.extended
    def test_gradient_flow(self):
        """Test gradient flow through embedding components."""
        x = torch.randn(2, 10, 8, 5, requires_grad=True)
        
        embed = GraphTimeSeriesEmbedding(c_in=5, d_model=64, num_nodes=8)
        output = embed(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        
    @pytest.mark.extended
    def test_different_input_sizes(self):
        """Test with different input sizes."""
        embed = GraphTimeSeriesEmbedding(c_in=5, d_model=64)
        
        # Test different sequence lengths
        for seq_len in [5, 10, 20]:
            x = torch.randn(2, seq_len, 5)
            output = embed(x)
            assert output.shape == (2, seq_len, 64)
            
        # Test different batch sizes
        for batch_size in [1, 3, 5]:
            x = torch.randn(batch_size, 10, 5)
            output = embed(x)
            assert output.shape == (batch_size, 10, 64)


if __name__ == "__main__":
    pytest.main([__file__])