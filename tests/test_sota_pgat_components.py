import torch
import torch.nn as nn
import pytest
import numpy as np
from unittest.mock import Mock, patch

# Import components to test
from layers.modular.decoder.mixture_density_decoder import MixtureDensityDecoder, MixtureNLLLoss
from layers.modular.attention.autocorr_temporal_attention import AutoCorrTemporalAttention, AutoCorrelationAttention
from layers.modular.embedding.structural_positional_encoding import (
    StructuralPositionalEncoding, LaplacianEigenmaps, EnhancedInitialEmbedding
)
from layers.modular.embedding.enhanced_temporal_encoding import (
    EnhancedTemporalEncoding, SinusoidalTemporalEncoding, AdaptiveTemporalEncoding
)
from layers.modular.graph.enhanced_pgat_layer import (
    EnhancedPGAT_CrossAttn_Layer, DynamicEdgeWeightComputation, EnhancedCrossAttentionGNNConv
)

class TestMixtureDensityDecoder:
    """Test suite for Mixture Density Network decoder."""
    
    @pytest.fixture
    def decoder(self):
        return MixtureDensityDecoder(
            d_model=64,
            pred_len=24,
            num_components=3
        )
    
    @pytest.fixture
    def sample_input(self):
        return torch.randn(8, 32, 64)  # [batch, seq_len, d_model]
    
    def test_decoder_initialization(self, decoder):
        """Test proper initialization of MDN decoder."""
        assert decoder.d_model == 64
        assert decoder.pred_len == 24
        assert decoder.num_components == 3
        assert isinstance(decoder.mean_head, nn.Linear)
        assert isinstance(decoder.std_head, nn.Linear)
        assert isinstance(decoder.weight_head, nn.Linear)
    
    def test_forward_pass(self, decoder, sample_input):
        """Test forward pass produces correct output shapes."""
        means, log_stds, log_weights = decoder(sample_input)
        
        batch_size = sample_input.size(0)
        expected_shape = (batch_size, decoder.pred_len, decoder.num_components)
        
        assert means.shape == expected_shape
        assert log_stds.shape == expected_shape
        assert log_weights.shape == expected_shape
    
    def test_sampling(self, decoder, sample_input):
        """Test sampling from mixture distribution."""
        mixture_params = decoder(sample_input)
        samples = decoder.sample(mixture_params, num_samples=100)
        
        batch_size = sample_input.size(0)
        expected_shape = (100, batch_size, decoder.pred_len)
        
        assert samples.shape == expected_shape
        assert not torch.isnan(samples).any()
    
    def test_prediction_summary(self, decoder, sample_input):
        """Test prediction summary computation."""
        mixture_params = decoder(sample_input)
        pred_mean, pred_std = decoder.prediction_summary(mixture_params)
        
        batch_size = sample_input.size(0)
        expected_shape = (batch_size, decoder.pred_len)
        
        assert pred_mean.shape == expected_shape
        assert pred_std.shape == expected_shape
        assert (pred_std > 0).all()  # Standard deviation should be positive
    
    def test_mixture_nll_loss(self, decoder, sample_input):
        """Test mixture negative log-likelihood loss."""
        mixture_params = decoder(sample_input)
        target = torch.randn(sample_input.size(0), decoder.pred_len)
        
        loss_fn = MixtureNLLLoss()
        loss = loss_fn(mixture_params, target)
        
        assert loss.dim() == 0  # Scalar loss
        assert loss > 0  # Loss should be positive
        assert not torch.isnan(loss)

class TestAutoCorrelationAttention:
    """Test suite for AutoCorrelation attention mechanism."""
    
    @pytest.fixture
    def attention(self):
        return AutoCorrTemporalAttention(
            d_model=64,
            num_heads=8,
            factor=3,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_inputs(self):
        batch_size, seq_len, d_model = 4, 32, 64
        queries = torch.randn(batch_size, seq_len, d_model)
        keys = torch.randn(batch_size, seq_len, d_model)
        values = torch.randn(batch_size, seq_len, d_model)
        history = torch.randn(batch_size, seq_len, d_model)
        return queries, keys, values, history
    
    def test_attention_initialization(self, attention):
        """Test proper initialization of autocorrelation attention."""
        assert attention.d_model == 64
        assert attention.num_heads == 8
        assert attention.factor == 3
        assert isinstance(attention.autocorr_attention, AutoCorrelationAttention)
    
    def test_forward_pass(self, attention, sample_inputs):
        """Test forward pass produces correct output shapes."""
        queries, keys, values, history = sample_inputs
        output = attention(queries, keys, values, history)
        
        assert output.shape == queries.shape
        assert not torch.isnan(output).any()
    
    def test_autocorrelation_computation(self):
        """Test autocorrelation computation in frequency domain."""
        autocorr = AutoCorrelationAttention(factor=3)
        
        # Create simple periodic signal
        seq_len = 32
        x = torch.sin(torch.linspace(0, 4*np.pi, seq_len)).unsqueeze(0).unsqueeze(-1)
        
        # Compute autocorrelation
        result = autocorr.time_delay_agg_training(x, x)
        
        assert result.shape == x.shape
        assert not torch.isnan(result).any()
    
    def test_top_k_selection(self):
        """Test top-k autocorrelation selection."""
        autocorr = AutoCorrelationAttention(factor=3)
        
        # Create autocorrelation values
        batch_size, seq_len = 2, 16
        autocorr_values = torch.randn(batch_size, seq_len)
        
        top_k_values, top_k_indices = autocorr._get_top_k_autocorr(
            autocorr_values, k=3
        )
        
        assert top_k_values.shape == (batch_size, 3)
        assert top_k_indices.shape == (batch_size, 3)
        
        # Check that values are sorted in descending order
        for i in range(batch_size):
            assert torch.all(top_k_values[i, :-1] >= top_k_values[i, 1:])

class TestStructuralPositionalEncoding:
    """Test suite for structural positional encoding."""
    
    @pytest.fixture
    def encoding(self):
        return StructuralPositionalEncoding(
            d_model=64,
            num_eigenvectors=8,
            max_nodes=100
        )
    
    @pytest.fixture
    def sample_data(self):
        num_nodes = 20
        node_features = torch.randn(num_nodes, 64)
        
        # Create simple adjacency matrix (ring graph)
        adj_matrix = torch.zeros(num_nodes, num_nodes)
        for i in range(num_nodes):
            adj_matrix[i, (i + 1) % num_nodes] = 1
            adj_matrix[(i + 1) % num_nodes, i] = 1
        
        return node_features, adj_matrix
    
    def test_laplacian_eigenmaps(self):
        """Test Laplacian eigenmap computation."""
        laplacian_maps = LaplacianEigenmaps(num_eigenvectors=4)
        
        # Simple 4-node cycle graph
        adj_matrix = torch.tensor([
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0]
        ], dtype=torch.float32)
        
        eigenvectors = laplacian_maps(adj_matrix)
        
        assert eigenvectors.shape == (4, 4)
        assert not torch.isnan(eigenvectors).any()
    
    def test_structural_encoding_forward(self, encoding, sample_data):
        """Test forward pass of structural encoding."""
        node_features, adj_matrix = sample_data
        encoded_features = encoding(node_features, adj_matrix)
        
        assert encoded_features.shape == node_features.shape
        assert not torch.isnan(encoded_features).any()
    
    def test_enhanced_initial_embedding(self):
        """Test enhanced initial embedding with structural encoding."""
        embedding = EnhancedInitialEmbedding(
            c_in=10,
            d_model=64,
            use_structural_encoding=True,
            num_eigenvectors=8
        )
        
        # Sample input
        batch_size, seq_len, c_in = 4, 32, 10
        x = torch.randn(batch_size, seq_len, c_in)
        
        # Simple adjacency matrix
        num_nodes = seq_len
        adj_matrix = torch.eye(num_nodes) + torch.diag(torch.ones(num_nodes-1), 1) + torch.diag(torch.ones(num_nodes-1), -1)
        
        output = embedding(x, adj_matrix)
        
        assert output.shape == (batch_size, seq_len, 64)
        assert not torch.isnan(output).any()

class TestEnhancedTemporalEncoding:
    """Test suite for enhanced temporal encoding."""
    
    @pytest.fixture
    def encoding(self):
        return EnhancedTemporalEncoding(
            d_model=64,
            max_len=100,
            num_scales=4
        )
    
    @pytest.fixture
    def sample_input(self):
        return torch.randn(4, 32, 64)  # [batch, seq_len, d_model]
    
    def test_sinusoidal_encoding(self):
        """Test sinusoidal temporal encoding."""
        sin_encoding = SinusoidalTemporalEncoding(d_model=64, max_len=100)
        
        seq_len = 32
        x = torch.randn(4, seq_len, 64)
        encoded = sin_encoding(x)
        
        assert encoded.shape == x.shape
        assert not torch.isnan(encoded).any()
    
    def test_adaptive_encoding(self):
        """Test adaptive temporal encoding."""
        adaptive_encoding = AdaptiveTemporalEncoding(
            d_model=64,
            max_len=100,
            num_scales=4
        )
        
        seq_len = 32
        x = torch.randn(4, seq_len, 64)
        encoded = adaptive_encoding(x)
        
        assert encoded.shape == x.shape
        assert not torch.isnan(encoded).any()
    
    def test_enhanced_encoding_forward(self, encoding, sample_input):
        """Test forward pass of enhanced temporal encoding."""
        output = encoding(sample_input)
        
        assert output.shape == sample_input.shape
        assert not torch.isnan(output).any()
    
    def test_multi_scale_features(self, encoding):
        """Test multi-scale temporal feature extraction."""
        # Test with different sequence lengths
        for seq_len in [16, 32, 64]:
            x = torch.randn(2, seq_len, 64)
            output = encoding(x)
            
            assert output.shape == x.shape
            assert not torch.isnan(output).any()

class TestDynamicEdgeWeights:
    """Test suite for dynamic edge weight computation."""
    
    @pytest.fixture
    def edge_weight_computer(self):
        return DynamicEdgeWeightComputation(
            d_model=64,
            num_heads=8,
            dropout=0.1
        )
    
    @pytest.fixture
    def sample_graph_data(self):
        num_source_nodes, num_target_nodes = 10, 8
        d_model = 64
        
        source_features = torch.randn(num_source_nodes, d_model)
        target_features = torch.randn(num_target_nodes, d_model)
        
        # Create random edge connectivity
        num_edges = 20
        edge_index = torch.randint(0, min(num_source_nodes, num_target_nodes), (2, num_edges))
        edge_index[0] = torch.randint(0, num_source_nodes, (num_edges,))
        edge_index[1] = torch.randint(0, num_target_nodes, (num_edges,))
        
        return source_features, target_features, edge_index
    
    def test_edge_weight_computation(self, edge_weight_computer, sample_graph_data):
        """Test dynamic edge weight computation."""
        source_features, target_features, edge_index = sample_graph_data
        
        edge_weights = edge_weight_computer(
            source_features, target_features, edge_index
        )
        
        num_edges = edge_index.size(1)
        num_heads = edge_weight_computer.num_heads
        
        assert edge_weights.shape == (num_edges, num_heads)
        assert (edge_weights >= 0).all()  # Weights should be non-negative
        assert (edge_weights <= 1).all()  # Weights should be <= 1 (after sigmoid)
        assert not torch.isnan(edge_weights).any()
    
    def test_enhanced_pgat_layer(self):
        """Test enhanced PGAT layer with dynamic edge weights."""
        layer = EnhancedPGAT_CrossAttn_Layer(
            d_model=64,
            num_heads=8,
            use_dynamic_weights=True
        )
        
        # Create sample heterogeneous graph data
        batch_size, d_model = 4, 64
        num_wave, num_trans, num_target = 10, 8, 6
        
        x_dict = {
            'wave': torch.randn(batch_size, num_wave, d_model),
            'transition': torch.randn(batch_size, num_trans, d_model),
            'target': torch.randn(batch_size, num_target, d_model)
        }
        
        t_dict = {
            'wave': torch.randn(batch_size, num_wave, d_model),
            'transition': torch.randn(batch_size, num_trans, d_model),
            'target': torch.randn(batch_size, num_target, d_model)
        }
        
        # Create edge indices
        edge_index_dict = {
            ('wave', 'interacts_with', 'transition'): torch.randint(0, min(num_wave, num_trans), (2, 15)),
            ('transition', 'influences', 'target'): torch.randint(0, min(num_trans, num_target), (2, 12))
        }
        
        # Forward pass
        x_out, t_out = layer(x_dict, t_dict, edge_index_dict)
        
        # Check output shapes
        assert x_out['wave'].shape == x_dict['wave'].shape
        assert x_out['transition'].shape == x_dict['transition'].shape
        assert x_out['target'].shape == x_dict['target'].shape
        
        assert t_out['wave'].shape == t_dict['wave'].shape
        assert t_out['transition'].shape == t_dict['transition'].shape
        assert t_out['target'].shape == t_dict['target'].shape
        
        # Check for NaN values
        for node_type in ['wave', 'transition', 'target']:
            assert not torch.isnan(x_out[node_type]).any()
            assert not torch.isnan(t_out[node_type]).any()
    
    def test_edge_weight_analysis(self):
        """Test edge weight analysis functionality."""
        layer = EnhancedPGAT_CrossAttn_Layer(
            d_model=64,
            num_heads=8,
            use_dynamic_weights=True
        )
        
        # Create minimal sample data
        d_model = 64
        x_dict = {
            'wave': torch.randn(5, d_model),
            'transition': torch.randn(4, d_model),
            'target': torch.randn(3, d_model)
        }
        
        edge_index_dict = {
            ('wave', 'interacts_with', 'transition'): torch.tensor([[0, 1, 2], [0, 1, 2]]),
            ('transition', 'influences', 'target'): torch.tensor([[0, 1], [0, 1]])
        }
        
        # Get edge weights
        edge_weights = layer.get_edge_weights(x_dict, edge_index_dict)
        
        assert ('wave', 'interacts_with', 'transition') in edge_weights
        assert ('transition', 'influences', 'target') in edge_weights
        
        # Check shapes
        wave_trans_weights = edge_weights[('wave', 'interacts_with', 'transition')]
        trans_target_weights = edge_weights[('transition', 'influences', 'target')]
        
        assert wave_trans_weights.shape == (3, 8)  # 3 edges, 8 heads
        assert trans_target_weights.shape == (2, 8)  # 2 edges, 8 heads

class TestIntegration:
    """Integration tests for SOTA PGAT components."""
    
    def test_component_compatibility(self):
        """Test that all components can work together."""
        # Test dimensions compatibility
        d_model = 64
        seq_len = 32
        pred_len = 24
        batch_size = 4
        
        # Initialize components
        temporal_encoding = EnhancedTemporalEncoding(d_model=d_model, max_len=100)
        autocorr_attention = AutoCorrTemporalAttention(d_model=d_model, num_heads=8)
        mdn_decoder = MixtureDensityDecoder(d_model=d_model, pred_len=pred_len, num_components=3)
        
        # Create sample data
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass through components
        x_encoded = temporal_encoding(x)
        x_attended = autocorr_attention(x_encoded, x_encoded, x_encoded, x_encoded)
        mixture_params = mdn_decoder(x_attended)
        
        # Check final output
        means, log_stds, log_weights = mixture_params
        assert means.shape == (batch_size, pred_len, 3)
        assert log_stds.shape == (batch_size, pred_len, 3)
        assert log_weights.shape == (batch_size, pred_len, 3)
    
    def test_gradient_flow(self):
        """Test that gradients flow properly through all components."""
        d_model = 32  # Smaller for faster testing
        
        # Create a simple pipeline
        components = nn.Sequential(
            EnhancedTemporalEncoding(d_model=d_model, max_len=50),
            AutoCorrTemporalAttention(d_model=d_model, num_heads=4),
            MixtureDensityDecoder(d_model=d_model, pred_len=12, num_components=2)
        )
        
        # Sample input and target
        x = torch.randn(2, 16, d_model, requires_grad=True)
        target = torch.randn(2, 12)
        
        # Forward pass
        temporal_encoded = components[0](x)
        attended = components[1](temporal_encoded, temporal_encoded, temporal_encoded, temporal_encoded)
        mixture_params = components[2](attended)
        
        # Compute loss
        loss_fn = MixtureNLLLoss()
        loss = loss_fn(mixture_params, target)
        
        # Backward pass
        loss.backward()
        
        # Check that gradients exist
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()
        
        # Check component gradients
        for component in components:
            for param in component.parameters():
                if param.requires_grad:
                    assert param.grad is not None
                    assert not torch.isnan(param.grad).any()

if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])