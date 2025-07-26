"""
Custom Multi-Head Attention Layer

This module provides a standalone, robust implementation of a standard
multi-head attention mechanism, promoting modularity and reusability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CustomMultiHeadAttention(nn.Module):
    """
    A standard Multi-Head Attention layer.
    
    Args:
        d_model (int): Total dimension of the model.
        n_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(CustomMultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        self.query_projection = nn.Linear(d_model, d_model)
        self.key_projection = nn.Linear(d_model, d_model)
        self.value_projection = nn.Linear(d_model, d_model)
        
        self.final_projection = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.scale = np.sqrt(self.head_dim)

    def forward(self, query, key, value, attn_mask=None):
        """
        Forward pass for Multi-Head Attention.
        
        Args:
            query: [batch_size, query_len, d_model]
            key: [batch_size, key_len, d_model]
            value: [batch_size, value_len, d_model]
            attn_mask: Optional mask.
            
        Returns:
            Attention output and attention weights.
        """
        batch_size = query.size(0)
        
        # Project and reshape for multi-head computation
        Q = self.query_projection(query).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key_projection(key).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value_projection(value).view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        
        if attn_mask is not None:
            attention_scores = attention_scores.masked_fill(attn_mask == 0, -1e9)
            
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Combine heads
        attention_output = torch.matmul(attention_weights, V)
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(batch_size, -1, self.d_model)
        
        # Final projection
        output = self.final_projection(attention_output)
        
        return output, attention_weights
