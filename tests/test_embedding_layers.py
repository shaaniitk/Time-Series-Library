import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layers.Embed import (
    PositionalEmbedding, TokenEmbedding, FixedEmbedding, 
    TemporalEmbedding, TimeFeatureEmbedding, DataEmbedding,
    DataEmbedding_inverted, DataEmbedding_wo_pos, PatchEmbedding
)


class TestEmbeddingComponents:
    """Test individual embedding components"""
    
    def test_positional_embedding(self):
        """Test positional embedding generation"""
        d_model = 64
        max_len = 1000
        pos_emb = PositionalEmbedding(d_model, max_len)
        
        # Test different sequence lengths
        for seq_len in [24, 96, 500]:
            x = torch.randn(2, seq_len, d_model)
            pos_encoding = pos_emb(x)
            
            assert pos_encoding.shape == (1, seq_len, d_model)
            assert not torch.isnan(pos_encoding).any()
            
            # Check sinusoidal pattern properties
            assert pos_encoding.requires_grad == False  # Should be fixed
    
    def test_token_embedding(self):
        """Test token embedding with convolution"""
        c_in = 7
        d_model = 64
        token_emb = TokenEmbedding(c_in, d_model)
        
        batch_size = 2
        seq_len = 96
        x = torch.randn(batch_size, seq_len, c_in)
        
        embedded = token_emb(x)
        
        assert embedded.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(embedded).any()
        
        # Test gradient flow
        loss = embedded.mean()
        loss.backward()
        assert any(p.grad is not None for p in token_emb.parameters())
    
    def test_fixed_embedding(self):
        """Test fixed embedding (non-trainable)"""
        c_in = 10
        d_model = 32
        fixed_emb = FixedEmbedding(c_in, d_model)
        
        # Test with integer indices
        x = torch.randint(0, c_in, (2, 24))
        embedded = fixed_emb(x)
        
        assert embedded.shape == (2, 24, d_model)
        assert not torch.isnan(embedded).any()
        
        # Should not have gradients (fixed)
        assert embedded.requires_grad == False
    
    def test_temporal_embedding(self):
        """Test temporal embedding for time features"""
        d_model = 64
        
        # Test different frequencies
        for freq in ['h', 't', 'd', 'm']:
            temp_emb = TemporalEmbedding(d_model, embed_type='fixed', freq=freq)
            
            # Create time mark tensor (month, day, weekday, hour, minute)
            batch_size = 2
            seq_len = 48
            
            if freq == 't':
                # Create proper time marks: [month(0-12), day(1-31), weekday(0-6), hour(0-23), minute(0-3)]
                x_mark = torch.stack([
                    torch.randint(0, 13, (batch_size, seq_len)),  # month
                    torch.randint(1, 32, (batch_size, seq_len)),  # day
                    torch.randint(0, 7, (batch_size, seq_len)),   # weekday
                    torch.randint(0, 24, (batch_size, seq_len)),  # hour
                    torch.randint(0, 4, (batch_size, seq_len))    # minute
                ], dim=-1)
            else:
                # Create proper time marks: [month(0-12), day(1-31), weekday(0-6), hour(0-23)]
                x_mark = torch.stack([
                    torch.randint(0, 13, (batch_size, seq_len)),  # month
                    torch.randint(1, 32, (batch_size, seq_len)),  # day
                    torch.randint(0, 7, (batch_size, seq_len)),   # weekday
                    torch.randint(0, 24, (batch_size, seq_len))   # hour
                ], dim=-1)
            
            embedded = temp_emb(x_mark)
            
            assert embedded.shape == (batch_size, seq_len, d_model)
            assert not torch.isnan(embedded).any()
    
    def test_time_feature_embedding(self):
        """Test time feature embedding (continuous features)"""
        d_model = 64
        
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        
        for freq, d_inp in freq_map.items():
            time_emb = TimeFeatureEmbedding(d_model, embed_type='timeF', freq=freq)
            
            batch_size = 2
            seq_len = 48
            x_mark = torch.randn(batch_size, seq_len, d_inp)
            
            embedded = time_emb(x_mark)
            
            assert embedded.shape == (batch_size, seq_len, d_model)
            assert not torch.isnan(embedded).any()


class TestDataEmbeddings:
    """Test complete data embedding modules"""
    
    def test_data_embedding_with_time_marks(self):
        """Test standard DataEmbedding with time marks"""
        c_in = 7
        d_model = 64
        data_emb = DataEmbedding(c_in, d_model, embed_type='fixed', freq='h')
        
        batch_size = 2
        seq_len = 96
        x = torch.randn(batch_size, seq_len, c_in)
        # Create proper time marks: [month(0-12), day(1-31), weekday(0-6), hour(0-23)]
        x_mark = torch.stack([
            torch.randint(0, 13, (batch_size, seq_len)),  # month
            torch.randint(1, 32, (batch_size, seq_len)),  # day
            torch.randint(0, 7, (batch_size, seq_len)),   # weekday
            torch.randint(0, 24, (batch_size, seq_len))   # hour
        ], dim=-1)
        
        embedded = data_emb(x, x_mark)
        
        assert embedded.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(embedded).any()
    
    def test_data_embedding_without_time_marks(self):
        """Test DataEmbedding without time marks"""
        c_in = 7
        d_model = 64
        data_emb = DataEmbedding(c_in, d_model)
        
        batch_size = 2
        seq_len = 96
        x = torch.randn(batch_size, seq_len, c_in)
        
        embedded = data_emb(x, None)
        
        assert embedded.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(embedded).any()
    
    def test_data_embedding_wo_pos(self):
        """Test DataEmbedding without positional encoding"""
        c_in = 7
        d_model = 64
        data_emb = DataEmbedding_wo_pos(c_in, d_model, embed_type='timeF', freq='h')
        
        batch_size = 2
        seq_len = 96
        x = torch.randn(batch_size, seq_len, c_in)
        x_mark = torch.randn(batch_size, seq_len, 4)  # timeF uses continuous features
        
        embedded = data_emb(x, x_mark)
        
        assert embedded.shape == (batch_size, seq_len, d_model)
        assert not torch.isnan(embedded).any()
    
    def test_data_embedding_inverted(self):
        """Test inverted DataEmbedding for iTransformer"""
        c_in = 96  # sequence length becomes input dimension
        d_model = 64
        data_emb = DataEmbedding_inverted(c_in, d_model)
        
        batch_size = 2
        n_vars = 7
        seq_len = 96
        x = torch.randn(batch_size, seq_len, n_vars)
        
        embedded = data_emb(x, None)
        
        # Output should be [Batch, Variate, d_model]
        assert embedded.shape == (batch_size, n_vars, d_model)
        assert not torch.isnan(embedded).any()
    
    def test_patch_embedding(self):
        """Test patch embedding for PatchTST"""
        d_model = 64
        patch_len = 16
        stride = 8
        padding = 4
        dropout = 0.1
        
        patch_emb = PatchEmbedding(d_model, patch_len, stride, padding, dropout)
        
        batch_size = 2
        n_vars = 7
        seq_len = 96
        x = torch.randn(batch_size, n_vars, seq_len)
        
        embedded, n_vars_out = patch_emb(x)
        
        # Calculate expected patch number
        padded_len = seq_len + padding
        n_patches = (padded_len - patch_len) // stride + 1
        expected_shape = (batch_size * n_vars, n_patches, d_model)
        
        assert embedded.shape == expected_shape
        assert n_vars_out == n_vars
        assert not torch.isnan(embedded).any()


class TestEmbeddingIntegration:
    """Test embedding integration and edge cases"""
    
    def test_embedding_gradient_flow(self):
        """Test gradient flow through embeddings"""
        c_in = 7
        d_model = 64
        
        embeddings = [
            DataEmbedding(c_in, d_model),
            DataEmbedding_wo_pos(c_in, d_model),
            DataEmbedding_inverted(96, d_model)  # Different input for inverted
        ]
        
        for i, emb in enumerate(embeddings):
            if i == 2:  # Inverted embedding
                x = torch.randn(2, 96, 7, requires_grad=True)
                output = emb(x, None)
            else:
                x = torch.randn(2, 96, c_in, requires_grad=True)
                # Create proper integer time marks for temporal embedding
                x_mark = torch.stack([
                    torch.randint(0, 13, (2, 96)),  # month
                    torch.randint(1, 32, (2, 96)),  # day
                    torch.randint(0, 7, (2, 96)),   # weekday
                    torch.randint(0, 24, (2, 96))   # hour
                ], dim=-1)
                output = emb(x, x_mark)
            
            loss = output.mean()
            loss.backward()
            
            # Check gradients exist
            assert x.grad is not None
            assert not torch.allclose(x.grad, torch.zeros_like(x.grad))
    
    def test_embedding_different_sequence_lengths(self):
        """Test embeddings with different sequence lengths"""
        c_in = 7
        d_model = 64
        data_emb = DataEmbedding(c_in, d_model)
        
        for seq_len in [24, 48, 96, 192, 336]:
            x = torch.randn(2, seq_len, c_in)
            # Create proper time marks: [month(0-12), day(1-31), weekday(0-6), hour(0-23)]
            x_mark = torch.stack([
                torch.randint(0, 13, (2, seq_len)),  # month
                torch.randint(1, 32, (2, seq_len)),  # day
                torch.randint(0, 7, (2, seq_len)),   # weekday
                torch.randint(0, 24, (2, seq_len))   # hour
            ], dim=-1)
            
            embedded = data_emb(x, x_mark)
            
            assert embedded.shape == (2, seq_len, d_model)
            assert not torch.isnan(embedded).any()
    
    def test_embedding_numerical_stability(self):
        """Test embedding numerical stability with extreme values"""
        c_in = 7
        d_model = 64
        data_emb = DataEmbedding(c_in, d_model)
        
        # Test with extreme input values
        for scale in [1e-6, 1e6]:
            x = torch.randn(2, 48, c_in) * scale
            # Create proper time marks: [month(0-12), day(1-31), weekday(0-6), hour(0-23)]
            x_mark = torch.stack([
                torch.randint(0, 13, (2, 48)),  # month
                torch.randint(1, 32, (2, 48)),  # day
                torch.randint(0, 7, (2, 48)),   # weekday
                torch.randint(0, 24, (2, 48))   # hour
            ], dim=-1)
            
            embedded = data_emb(x, x_mark)
            
            assert not torch.isnan(embedded).any()
            assert not torch.isinf(embedded).any()
    
    def test_embedding_consistency(self):
        """Test embedding consistency across multiple calls"""
        c_in = 7
        d_model = 64
        data_emb = DataEmbedding(c_in, d_model)
        
        # Same input should produce same output
        x = torch.randn(2, 48, c_in)
        # Create proper time marks: [month(0-12), day(1-31), weekday(0-6), hour(0-23)]
        x_mark = torch.stack([
            torch.randint(0, 13, (2, 48)),  # month
            torch.randint(1, 32, (2, 48)),  # day
            torch.randint(0, 7, (2, 48)),   # weekday
            torch.randint(0, 24, (2, 48))   # hour
        ], dim=-1)
        
        # Set deterministic mode for consistency test
        data_emb.eval()
        torch.manual_seed(42)
        with torch.no_grad():
            output1 = data_emb(x, x_mark)
        
        torch.manual_seed(42)
        with torch.no_grad():
            output2 = data_emb(x, x_mark)
        
        assert torch.allclose(output1, output2, atol=1e-6)
    
    def test_embedding_parameter_initialization(self):
        """Test proper parameter initialization"""
        c_in = 7
        d_model = 64
        
        # Test TokenEmbedding initialization
        token_emb = TokenEmbedding(c_in, d_model)
        
        # Check conv layer is properly initialized
        conv_layer = token_emb.tokenConv
        assert hasattr(conv_layer, 'weight')
        
        # Weight should not be all zeros or all ones
        weight = conv_layer.weight.data
        assert not torch.allclose(weight, torch.zeros_like(weight))
        assert not torch.allclose(weight, torch.ones_like(weight))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])