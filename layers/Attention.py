"""Attention layer factory and utilities.

This module provides a factory function to get attention layers and related utilities.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any

try:
    from .SelfAttention_Family import FullAttention, ProbAttention, AttentionLayer
except ImportError:
    # Fallback implementations
    class FullAttention(nn.Module):
        def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
            super().__init__()
            self.scale = scale
            self.mask_flag = mask_flag
            self.output_attention = output_attention
            self.dropout = nn.Dropout(attention_dropout)
            
        def forward(self, queries, keys, values, attn_mask=None):
            B, L, H, E = queries.shape
            _, S, _, D = values.shape
            scale = self.scale or 1. / (E ** 0.5)
            
            scores = torch.einsum("blhe,bshe->bhls", queries, keys)
            if self.scale:
                scores = scores * scale
                
            if attn_mask is not None:
                scores.masked_fill_(attn_mask, -1e9)
                
            A = self.dropout(torch.softmax(scores, dim=-1))
            V = torch.einsum("bhls,bshd->blhd", A, values)
            
            if self.output_attention:
                return V.contiguous(), A
            else:
                return V.contiguous(), None
    
    class ProbAttention(FullAttention):
        """Simplified ProbAttention that falls back to FullAttention."""
        pass
    
    class AttentionLayer(nn.Module):
        def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
            super().__init__()
            d_keys = d_keys or (d_model // n_heads)
            d_values = d_values or (d_model // n_heads)
            
            self.inner_attention = attention
            self.query_projection = nn.Linear(d_model, d_keys * n_heads)
            self.key_projection = nn.Linear(d_model, d_keys * n_heads)
            self.value_projection = nn.Linear(d_model, d_values * n_heads)
            self.out_projection = nn.Linear(d_values * n_heads, d_model)
            self.n_heads = n_heads
            
        def forward(self, queries, keys, values, attn_mask=None):
            B, L, _ = queries.shape
            _, S, _ = keys.shape
            H = self.n_heads
            
            queries = self.query_projection(queries).view(B, L, H, -1)
            keys = self.key_projection(keys).view(B, S, H, -1)
            values = self.value_projection(values).view(B, S, H, -1)
            
            out, attn = self.inner_attention(queries, keys, values, attn_mask)
            out = out.view(B, L, -1)
            
            return self.out_projection(out), attn

try:
    from .CustomMultiHeadAttention import CustomMultiHeadAttention
except ImportError:
    CustomMultiHeadAttention = None


def get_attention_layer(attention_type = "full", **kwargs) -> nn.Module:
    """Factory function to get attention layers.

    Args:
        attention_type: Type of attention ('full', 'prob', 'custom', etc.) or config object
        **kwargs: Additional arguments for the attention layer

    Returns:
        Attention layer instance
    """
    # Handle case where attention_type is a config object
    if hasattr(attention_type, 'attention_type'):
        attention_type = attention_type.attention_type
    
    # Handle None case
    if attention_type is None:
        attention_type = "full"
    
    attention_type = str(attention_type).lower()
    
    if attention_type == "full":
        return FullAttention(**kwargs)
    elif attention_type == "prob":
        return ProbAttention(**kwargs)
    elif attention_type == "custom" and CustomMultiHeadAttention is not None:
        return CustomMultiHeadAttention(**kwargs)
    else:
        # Default fallback
        return FullAttention(**kwargs)


def create_attention_layer(attention_type: str, d_model: int, n_heads: int, **kwargs) -> AttentionLayer:
    """Create a complete attention layer with projections.
    
    Args:
        attention_type: Type of attention mechanism
        d_model: Model dimension
        n_heads: Number of attention heads
        **kwargs: Additional arguments
        
    Returns:
        Complete attention layer
    """
    attention = get_attention_layer(attention_type, **kwargs)
    return AttentionLayer(attention, d_model, n_heads)


# Backward compatibility aliases
get_attention = get_attention_layer
create_attention = create_attention_layer