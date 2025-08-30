import torch
import pytest
from ..base_test_component import BaseComponentTest

class BaseAttentionTest(BaseComponentTest):
    """
    An abstract test suite for all attention components. It inherits from
    BaseComponentTest and adds attention-specific tests.
    """

    # --- Fixtures to provide data for attention tests ---
    @pytest.fixture
    def batch_size(self):
        return 2

    @pytest.fixture
    def d_model(self):
        return 16

    @pytest.fixture
    def n_heads(self):
        return 4
    
    @pytest.fixture
    def seq_len_q(self):
        return 32

    @pytest.fixture
    def seq_len_kv(self):
        return 32 # Can be overridden in specific tests for cross-attention

    @pytest.fixture
    def queries(self, batch_size, seq_len_q, d_model):
        return torch.randn(batch_size, seq_len_q, d_model)

    @pytest.fixture
    def keys(self, batch_size, seq_len_kv, d_model):
        return torch.randn(batch_size, seq_len_kv, d_model)

    @pytest.fixture
    def values(self, batch_size, seq_len_kv, d_model):
        return torch.randn(batch_size, seq_len_kv, d_model)
        
    # --- Attention-Specific Tests ---
    
    def test_forward_pass_runs(self, component_instance, queries, keys, values):
        """Tests that the forward pass executes without errors."""
        try:
            component_instance(queries, keys, values)
        except Exception as e:
            pytest.fail(f"Forward pass failed with an exception: {e}")

    def test_output_shape_is_correct(self, component_instance, queries, keys, values, batch_size, seq_len_q):
        """Tests if the output tensor has the expected shape."""
        output, _ = component_instance(queries, keys, values)
        
        expected_dim = component_instance.returnOutputDimension()
        assert output.shape == (batch_size, seq_len_q, expected_dim), "Output shape is incorrect"

    def test_attention_weights_shape_is_valid(self, component_instance, queries, keys, values, batch_size, n_heads, seq_len_q, seq_len_kv):
        """Tests the shape of returned attention weights, if any."""
        _, attn_weights = component_instance(queries, keys, values)
        
        if attn_weights is not None:
            assert attn_weights.dim() in [3, 4], "Attention weights should be 3D or 4D"
            assert attn_weights.shape[0] == batch_size, "Batch dimension mismatch in weights"
            
            if attn_weights.dim() == 4:
                assert attn_weights.shape[1] == n_heads, "Head dimension mismatch in weights"
            
            assert attn_weights.shape[-2] == seq_len_q, "Query sequence length mismatch in weights"
            assert attn_weights.shape[-1] == seq_len_kv, "Key/Value sequence length mismatch in weights"