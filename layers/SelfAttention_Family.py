"""Self-attention family implementations for Crossformer model.

This module provides attention mechanisms used by the Crossformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
import math
import numpy as np


class FullAttention(nn.Module):
    """Full attention mechanism.
    
    Standard multi-head attention with full attention matrix computation.
    """
    
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)
        
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)
            
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)
        
        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    """Attention layer wrapper.
    
    Wraps attention mechanism with input/output projections.
    """
    
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        
    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
        
        return self.out_projection(out), attn


class TriangularCausalMask():
    """Triangular causal mask for attention."""
    
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)
    
    @property
    def mask(self):
        return self._mask


class TwoStageAttentionLayer(nn.Module):
    """Two-stage attention layer for Crossformer model.
    
    This layer implements a two-stage attention mechanism that processes
    segments with both intra-segment and inter-segment attention.
    """
    
    def __init__(self, configs, seg_num, factor, d_model, n_heads, d_ff, dropout):
        super(TwoStageAttentionLayer, self).__init__()
        self.seg_num = seg_num
        self.factor = factor
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        
        # Multi-head attention for intra-segment processing
        self.intra_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Multi-head attention for inter-segment processing
        self.inter_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """Forward pass of two-stage attention.
        
        Args:
            x: Input tensor of shape (batch_size, ts_dim, seg_num, d_model)
            
        Returns:
            Output tensor of same shape as input
        """
        batch_size, ts_dim, seg_num, d_model = x.shape
        
        # Stage 1: Intra-segment attention
        # Reshape for processing segments independently
        x_intra = rearrange(x, 'b ts seg d -> (b ts) seg d')
        
        # Apply intra-segment attention
        attn_out, _ = self.intra_attention(x_intra, x_intra, x_intra)
        x_intra = self.norm1(x_intra + self.dropout(attn_out))
        
        # Reshape back
        x_intra = rearrange(x_intra, '(b ts) seg d -> b ts seg d', b=batch_size)
        
        # Stage 2: Inter-segment attention
        # Reshape for processing time series independently
        x_inter = rearrange(x_intra, 'b ts seg d -> (b seg) ts d')
        
        # Apply inter-segment attention
        attn_out, _ = self.inter_attention(x_inter, x_inter, x_inter)
        x_inter = self.norm2(x_inter + self.dropout(attn_out))
        
        # Reshape back
        x_inter = rearrange(x_inter, '(b seg) ts d -> b ts seg d', b=batch_size)
        
        # Feed-forward network
        ffn_out = self.ffn(x_inter)
        output = self.norm3(x_inter + ffn_out)
        
        return output