"""
Custom Multi-Head Attention Layer

This module provides a standalone, robust implementation of a standard
multi-head attention mechanism, promoting modularity and reusability.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class CustomMultiHeadAttention(nn.Module):
    """
    Optimized Multi-Head Attention for time series with memory efficiency.
    
    Args:
        d_model (int): Total dimension of the model.
        n_heads (int): Number of parallel attention heads.
        dropout (float): Dropout probability.
        use_flash_attention (bool): Use memory-efficient attention if available.
        local_window (int): Local attention window size (0 = global attention).
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1, 
                 use_flash_attention: bool = True, local_window: int = 0):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.local_window = local_window
        self.use_flash_attention = use_flash_attention and hasattr(F, 'scaled_dot_product_attention')
        
        # Fused QKV projection for efficiency
        self.qkv_projection = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_projection = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        nn.init.xavier_uniform_(self.qkv_projection.weight)
        nn.init.xavier_uniform_(self.out_projection.weight)
        nn.init.zeros_(self.out_projection.bias)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None, 
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Optimized forward pass for self-attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attn_mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            (output, attention_weights)
        """
        B, L, D = x.shape
        
        # Fused QKV projection
        qkv = self.qkv_projection(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # [3, B, n_heads, L, head_dim]
        
        if self.use_flash_attention:
            # Use PyTorch's optimized attention
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=self.dropout.p if self.training else 0.0
            )
            attn_weights = None if not return_attention else self._compute_attention_weights(q, k, attn_mask)
        else:
            attn_output, attn_weights = self._manual_attention(q, k, v, attn_mask, return_attention)
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).reshape(B, L, D)
        output = self.out_projection(attn_output)
        
        return output, attn_weights
    
    def _manual_attention(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         attn_mask: Optional[torch.Tensor], return_attention: bool) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Manual attention computation with local window support"""
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply local attention window if specified
        if self.local_window > 0:
            scores = self._apply_local_mask(scores)
        
        # Apply attention mask
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)
        
        return attn_output, attn_weights if return_attention else None
    
    def _apply_local_mask(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply local attention window mask"""
        L = scores.size(-1)
        mask = torch.ones_like(scores, dtype=torch.bool)
        
        for i in range(L):
            start = max(0, i - self.local_window // 2)
            end = min(L, i + self.local_window // 2 + 1)
            mask[:, :, i, start:end] = False
        
        return scores.masked_fill(mask, float('-inf'))
    
    def _compute_attention_weights(self, q: torch.Tensor, k: torch.Tensor, 
                                  attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Compute attention weights for visualization"""
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        return F.softmax(scores, dim=-1)
