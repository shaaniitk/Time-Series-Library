"""
Test suite for Graph Encoder components.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from layers.modular.encoder.graph_encoder import (
    GraphEncoderLayer, GraphTimeSeriesEncoder, 
    HybridGraphEncoder, AdaptiveGraphEncoder
)


class TestGraphEncoderLayer:
    """Test cases for GraphEncoderLayer."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, seq_len, d_model = 2, 12, 64
        x = torch.randn(batch_size, seq_len, d_model)
        adj_matrix = torch.randint(0, 2, (batch_size, seq_len, seq_len)).float()
        return x, adj_matrix
    
    @pytest.mark.smoke
    def test_init_correlation_graph(self):
        """Test initialization with correlation graph type."""
        layer = GraphEncoderLayer(d_model=64, graph_type="correlation", threshold=0.5)
        assert layer.graph_type == "correlation"
        assert layer.graph_kwargs['threshold'] == 0.5
        
    @pytest.mark.smoke
    def test_init_knn_graph(self):
        """Test initialization with k-NN graph type."""
        layer = GraphEncoderLayer(d_model=64, graph_type="knn", k=5)
        assert layer.graph_type == "knn"
        assert layer.graph_kwargs['k'] == 5
        
    @pytest.mark.smoke
    def test_init_learned_graph(self):
        """Test initialization with learned graph type."""
        layer = GraphEncoderLayer(d_model=64, graph_type="learned", num_nodes=10)
        assert layer.graph_type == "learned"
        assert hasattr(layer, 'adj_matrix')
        assert layer.adj_matrix.shape == (10, 10)
        
    @pytest.mark.smoke
    def test_forward_shape(self, sample_data):
        """Test forward pass output shapes."""
        x, adj_matrix = sample_data
        layer = GraphEncoderLayer(d_model=64, graph_type="provided")
        
        output, attention = layer(x, adj_matrix)
        
        assert output.shape == x.shape
        assert attention.shape == (x.size(0), x.size(1), x.size(1))
        
    @pytest.mark.extended
    def test_graph_construction_correlation(self, sample_data):
        """Test correlation graph construction."""
        x, _ = sample_data
        layer = GraphEncoderLayer(d_model=64, graph_type="correlation", threshold=0.3)
        
        adj_matrix = layer.construct_graph(x)
        
        assert adj_matrix is not None
        assert adj_matrix.shape == (x.size(0), x.size(1), x.size(1))
        # Should be binary
        unique_values = torch.unique(adj_matrix)
        assert all(val in [0.0, 1.0] for val in unique_values.tolist())
        
    @pytest.mark.extended
    def test_graph_construction_knn(self, sample_data):
        """Test k-NN graph construction."""
        x, _ = sample_data
        layer = GraphEncoderLayer(d_model=64, graph_type="knn", k=4)
        
        adj_matrix = layer.construct_graph(x)
        
        assert adj_matrix is not None
        assert adj_matrix.shape == (x.size(0), x.size(1), x.size(1))
        # Should be binary
        unique_values = torch.unique(adj_matrix)
        assert all(val in [0.0, 1.0] for val in unique_values.tolist())
        
    @pytest.mark.extended
    def test_graph_construction_learned(self, sample_data):
        """Test learned graph construction."""
        x, _ = sample_data
        layer = GraphEncoderLayer(d_model=64, graph_type="learned", num_nodes=x.size(1))
        
        adj_matrix = layer.construct_graph(x)
        
        assert adj_matrix is not None
        assert adj_matrix.shape == (x.size(0), x.size(1), x.size(1))
        # Should be in [0, 1] due to sigmoid
        assert torch.all(adj_matrix >= 0) and torch.all(adj_matrix <= 1)
        
    @pytest.mark.extended
    def test_forward_without_adjacency(self, sample_data):
        """Test forward pass without providing adjacency matrix."""
        x, _ = sample_data
        layer = GraphEncoderLayer(d_model=64, graph_type="correlation", threshold=0.4)
        
        output, attention = layer(x)  # No adjacency matrix provided
        
        assert output.shape == x.shape
        assert attention.shape == (x.size(0), x.size(1), x.size(1))
        
    @pytest.mark.extended
    def test_gradient_flow(self, sample_data):
        """Test gradient flow through the layer."""
        x, adj_matrix = sample_data
        x.requires_grad_(True)
        
        layer = GraphEncoderLayer(d_model=64, graph_type="provided")
        output, _ = layer(x, adj_matrix)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))


class TestGraphTimeSeriesEncoder:
    """Test cases for GraphTimeSeriesEncoder."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, seq_len, d_model = 2, 15, 64
        x = torch.randn(batch_size, seq_len, d_model)
        adj_matrix = torch.randint(0, 2, (batch_size, seq_len, seq_len)).float()
        return x, adj_matrix
    
    @pytest.mark.smoke
    def test_init(self):
        """Test GraphTimeSeriesEncoder initialization."""
        encoder = GraphTimeSeriesEncoder(
            num_layers=3, d_model=64, n_heads=8, graph_type="correlation"
        )
        assert len(encoder.layers) == 3
        assert encoder.d_model == 64
        assert encoder.graph_type == "correlation"
        
    @pytest.mark.smoke
    def test_forward_shape(self, sample_data):
        """Test forward pass output shapes."""
        x, adj_matrix = sample_data
        encoder = GraphTimeSeriesEncoder(
            num_layers=2, d_model=64, n_heads=8, graph_type="provided"
        )
        
        output, attentions = encoder(x, adj_matrix)
        
        assert output.shape == x.shape
        assert len(attentions) == 2  # Number of layers
        for attention in attentions:
            assert attention.shape == (x.size(0), x.size(1), x.size(1))
            
    @pytest.mark.extended
    def test_different_graph_types(self, sample_data):
        """Test encoder with different graph types."""
        x, _ = sample_data
        
        graph_types = ["correlation", "knn", "learned"]
        graph_kwargs = [
            {"threshold": 0.5},
            {"k": 3},
            {"num_nodes": x.size(1)}
        ]
        
        for graph_type, kwargs in zip(graph_types, graph_kwargs):
            encoder = GraphTimeSeriesEncoder(
                num_layers=2, d_model=64, n_heads=8, 
                graph_type=graph_type, **kwargs
            )
            
            output, attentions = encoder(x)
            
            assert output.shape == x.shape
            assert len(attentions) == 2
            
    @pytest.mark.extended
    def test_with_normalization(self, sample_data):
        """Test encoder with normalization layer."""
        x, adj_matrix = sample_data
        norm_layer = nn.LayerNorm(64)
        
        encoder = GraphTimeSeriesEncoder(
            num_layers=2, d_model=64, n_heads=8, 
            graph_type="provided", norm_layer=norm_layer
        )
        
        output, _ = encoder(x, adj_matrix)
        
        assert output.shape == x.shape
        # Output should be normalized
        assert torch.allclose(output.mean(dim=-1), torch.zeros_like(output.mean(dim=-1)), atol=1e-6)


class TestHybridGraphEncoder:
    """Test cases for HybridGraphEncoder."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, seq_len, d_model = 2, 16, 64
        x = torch.randn(batch_size, seq_len, d_model)
        adj_matrix = torch.randint(0, 2, (batch_size, seq_len, seq_len)).float()
        return x, adj_matrix
    
    @pytest.mark.smoke
    def test_init_default_graph_layers(self):
        """Test initialization with default graph layers."""
        encoder = HybridGraphEncoder(
            num_layers=4, d_model=64, n_heads=8, graph_type="correlation"
        )
        assert len(encoder.layers) == 4
        # Default: every other layer should be graph layer
        assert encoder.graph_layers == [0, 2]
        
    @pytest.mark.smoke
    def test_init_custom_graph_layers(self):
        """Test initialization with custom graph layers."""
        graph_layers = [1, 3]
        encoder = HybridGraphEncoder(
            num_layers=4, d_model=64, n_heads=8, 
            graph_type="correlation", graph_layers=graph_layers
        )
        assert encoder.graph_layers == graph_layers
        
    @pytest.mark.smoke
    def test_forward_shape(self, sample_data):
        """Test forward pass output shapes."""
        x, adj_matrix = sample_data
        encoder = HybridGraphEncoder(
            num_layers=3, d_model=64, n_heads=8, 
            graph_type="provided", graph_layers=[0, 2]
        )
        
        output, attentions = encoder(x, adj_matrix)
        
        assert output.shape == x.shape
        assert len(attentions) == 3
        
    @pytest.mark.extended
    def test_mixed_layer_types(self, sample_data):
        """Test that hybrid encoder uses both graph and standard layers."""
        x, adj_matrix = sample_data
        
        # Create encoder with specific graph layers
        encoder = HybridGraphEncoder(
            num_layers=4, d_model=64, n_heads=8,
            graph_type="provided", graph_layers=[1, 3]
        )
        
        # Check that we have both types of layers
        graph_layer_count = 0
        standard_layer_count = 0
        
        for i, layer in enumerate(encoder.layers):
            if i in encoder.graph_layers:
                graph_layer_count += 1
                # Should be GraphEncoderLayer
                assert hasattr(layer, 'graph_attention')
            else:
                standard_layer_count += 1
                # Should be StandardEncoderLayer
                assert hasattr(layer, 'attention')
                
        assert graph_layer_count == 2
        assert standard_layer_count == 2


class TestAdaptiveGraphEncoder:
    """Test cases for AdaptiveGraphEncoder."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        batch_size, seq_len, d_model = 2, 14, 64
        x = torch.randn(batch_size, seq_len, d_model)
        adj_matrix = torch.randint(0, 2, (batch_size, seq_len, seq_len)).float()
        return x, adj_matrix
    
    @pytest.mark.smoke
    def test_init(self):
        """Test AdaptiveGraphEncoder initialization."""
        encoder = AdaptiveGraphEncoder(
            num_layers=3, d_model=64, n_heads=8, graph_type="correlation"
        )
        assert len(encoder.graph_layers) == 3
        assert len(encoder.temporal_layers) == 3
        assert hasattr(encoder, 'weight_predictor')
        
    @pytest.mark.smoke
    def test_forward_shape(self, sample_data):
        """Test forward pass output shapes."""
        x, adj_matrix = sample_data
        encoder = AdaptiveGraphEncoder(
            num_layers=2, d_model=64, n_heads=8, graph_type="provided"
        )
        
        output, attentions = encoder(x, adj_matrix)
        
        assert output.shape == x.shape
        assert len(attentions) == 2
        
    @pytest.mark.extended
    def test_adaptive_weighting(self, sample_data):
        """Test that adaptive weighting produces different outputs for different inputs."""
        x1, adj_matrix = sample_data
        x2 = torch.randn_like(x1) * 2  # Different scale
        
        encoder = AdaptiveGraphEncoder(
            num_layers=2, d_model=64, n_heads=8, graph_type="provided"
        )
        
        output1, _ = encoder(x1, adj_matrix)
        output2, _ = encoder(x2, adj_matrix)
        
        # Should produce different outputs due to adaptive weighting
        assert not torch.allclose(output1, output2, atol=1e-6)
        
    @pytest.mark.extended
    def test_weight_predictor_output(self, sample_data):
        """Test that weight predictor produces valid weights."""
        x, adj_matrix = sample_data
        encoder = AdaptiveGraphEncoder(
            num_layers=1, d_model=64, n_heads=8, graph_type="provided"
        )
        
        # Access weight predictor directly
        weights = encoder.weight_predictor(x.mean(dim=1))
        
        # Should be in [0, 1] due to sigmoid
        assert torch.all(weights >= 0) and torch.all(weights <= 1)
        assert weights.shape == (x.size(0), 1)


class TestGraphEncoderIntegration:
    """Integration tests for graph encoder components."""
    
    @pytest.mark.extended
    def test_end_to_end_pipeline(self):
        """Test complete encoder pipeline."""
        batch_size, seq_len, d_model = 2, 20, 64
        
        # Create sample data
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Test different encoder types
        encoders = [
            GraphTimeSeriesEncoder(num_layers=2, d_model=d_model, n_heads=8, graph_type="correlation"),
            HybridGraphEncoder(num_layers=2, d_model=d_model, n_heads=8, graph_type="knn", k=3),
            AdaptiveGraphEncoder(num_layers=2, d_model=d_model, n_heads=8, graph_type="learned", num_nodes=seq_len)
        ]
        
        for encoder in encoders:
            output, attentions = encoder(x)
            
            # Verify output properties
            assert output.shape == x.shape
            assert not torch.isnan(output).any()
            assert not torch.isinf(output).any()
            assert len(attentions) == 2
            
    @pytest.mark.extended
    def test_gradient_flow(self):
        """Test gradient flow through encoders."""
        x = torch.randn(2, 10, 64, requires_grad=True)
        
        encoder = GraphTimeSeriesEncoder(
            num_layers=2, d_model=64, n_heads=8, graph_type="correlation"
        )
        
        output, _ = encoder(x)
        loss = output.sum()
        loss.backward()
        
        assert x.grad is not None
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
        
    @pytest.mark.extended
    def test_different_sequence_lengths(self):
        """Test encoders with different sequence lengths."""
        d_model = 64
        encoder = GraphTimeSeriesEncoder(
            num_layers=2, d_model=d_model, n_heads=8, graph_type="correlation"
        )
        
        for seq_len in [8, 16, 32]:
            x = torch.randn(2, seq_len, d_model)
            output, attentions = encoder(x)
            
            assert output.shape == (2, seq_len, d_model)
            assert len(attentions) == 2
            for attention in attentions:
                assert attention.shape == (2, seq_len, seq_len)
                
    @pytest.mark.extended
    def test_encoder_comparison(self):
        """Compare outputs from different encoder types."""
        batch_size, seq_len, d_model = 2, 12, 64
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Create different encoders
        graph_encoder = GraphTimeSeriesEncoder(
            num_layers=2, d_model=d_model, n_heads=8, graph_type="correlation"
        )
        hybrid_encoder = HybridGraphEncoder(
            num_layers=2, d_model=d_model, n_heads=8, 
            graph_type="correlation", graph_layers=[0]
        )
        adaptive_encoder = AdaptiveGraphEncoder(
            num_layers=2, d_model=d_model, n_heads=8, graph_type="correlation"
        )
        
        # Get outputs
        output_graph, _ = graph_encoder(x)
        output_hybrid, _ = hybrid_encoder(x)
        output_adaptive, _ = adaptive_encoder(x)
        
        # All should have same shape
        assert output_graph.shape == output_hybrid.shape == output_adaptive.shape
        
        # Should produce different outputs
        assert not torch.allclose(output_graph, output_hybrid, atol=1e-6)
        assert not torch.allclose(output_graph, output_adaptive, atol=1e-6)
        assert not torch.allclose(output_hybrid, output_adaptive, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])