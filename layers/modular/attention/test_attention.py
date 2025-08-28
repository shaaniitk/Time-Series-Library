import unittest
import torch
from abc import ABC, abstractmethod
from typing import Dict, Any

from attention.base import BaseAttention

class BaseAttentionTest(ABC, unittest.TestCase):
    """
    Abstract Base Class for testing all attention implementations.
    
    Any new attention test class should inherit from this and implement
    the abstract methods to ensure consistent testing across all components.
    """
    
    def setUp(self):
        """Set up common parameters for tests."""
        self.batch_size = 4
        self.seq_len = 64
        self.d_model = 32
        self.n_heads = 4
        self.attention_instance = self._get_attention_instance()
        self.queries = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.keys = torch.randn(self.batch_size, self.seq_len, self.d_model)
        self.values = torch.randn(self.batch_size, self.seq_len, self.d_model)

    @abstractmethod
    def _get_attention_instance(self) -> BaseAttention:
        """
        Return an instance of the specific attention module to be tested.
        This must be implemented by each concrete test class.
        """
        raise NotImplementedError

    def test_forward_pass_shape(self):
        """Test if the forward pass runs and output shape is correct."""
        output, _ = self.attention_instance(self.queries, self.keys, self.values)
        
        expected_dim = self.attention_instance.returnOutputDimension()
        self.assertEqual(output.shape, (self.batch_size, self.seq_len, expected_dim))

    def test_attention_weights_shape(self):
        """Test the shape of returned attention weights, if any."""
        _, attn_weights = self.attention_instance(self.queries, self.keys, self.values)
        
        if attn_weights is not None:
            # Common shape: [B, H, L, S] or [B, L, S]
            self.assertIn(attn_weights.dim(), [3, 4])
            self.assertEqual(attn_weights.shape[0], self.batch_size)
            # Allow for different seq lengths if that's a feature
            self.assertEqual(attn_weights.shape[-2], self.seq_len)
            self.assertEqual(attn_weights.shape[-1], self.seq_len)
            
    def test_scriptable(self):
        """Test if the module can be JIT-scripted."""
        try:
            torch.jit.script(self.attention_instance)
        except Exception as e:
            self.fail(f"Module scripting failed with error: {e}")


# --- Example Concrete Test ---
from attention.core.multihead_attention import MultiHeadAttention

class TestMultiHeadAttention(BaseAttentionTest):
    
    def _get_attention_instance(self) -> BaseAttention:
        return MultiHeadAttention(d_model=self.d_model, n_heads=self.n_heads, dropout=0.1)

# You would then create a similar test class for every attention module.