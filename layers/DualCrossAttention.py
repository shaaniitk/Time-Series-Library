"""
DualCrossAttention - Cross attention mechanism between target and covariate context vectors
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from layers.Normalization import get_norm_layer
from utils.logger import logger


class DualCrossAttention(nn.Module):
    """
    Implements dual cross-attention between target and covariate context vectors.
    Allows both contexts to attend to each other bidirectionally.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_residual: bool = True,
        use_layer_norm: bool = True,
        attention_temperature: float = 1.0,
        norm_type: str = 'layernorm'
    ):
        super(DualCrossAttention, self).__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.use_residual = use_residual
        self.use_layer_norm = use_layer_norm
        self.temperature = attention_temperature
        
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        # Cross-attention: Target attends to Covariate
        self.target_to_covariate_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention: Covariate attends to Target
        self.covariate_to_target_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        if use_layer_norm:
            self.target_norm2 = nn.LayerNorm(d_model)
            self.covariate_norm1 = nn.LayerNorm(d_model)
            self.covariate_norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward networks for post-attention processing
        self.target_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        self.covariate_ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        logger.info(f"DualCrossAttention initialized: d_model={d_model}, num_heads={num_heads}")
    
    def forward(
        self,
        target_context: torch.Tensor,
        covariate_context: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        covariate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply dual cross-attention between target and covariate contexts.
        
        Args:
            target_context: Target context vector [batch_size, d_model] or [batch_size, seq_len, d_model]
            covariate_context: Covariate context vector [batch_size, d_model] or [batch_size, seq_len, d_model]
            target_mask: Optional mask for target context
            covariate_mask: Optional mask for covariate context
            
        Returns:
            Tuple of (attended_target_context, attended_covariate_context)
        """
        batch_size = target_context.size(0)
        
        # Ensure inputs are 3D for attention (add sequence dimension if needed)
        if target_context.dim() == 2:
            target_context = target_context.unsqueeze(1)  # [B, 1, D]
        if covariate_context.dim() == 2:
            covariate_context = covariate_context.unsqueeze(1)  # [B, 1, D]
        
        logger.debug(f"DualCrossAttention input shapes - target: {target_context.shape}, "
                    f"covariate: {covariate_context.shape}")
        
        # Store original contexts for residual connections
        target_residual = target_context
        covariate_residual = covariate_context
        
        # Cross-attention 1: Target attends to Covariate
        try:
            target_attended, target_attention_weights = self.target_to_covariate_attention(
                query=target_context,
                key=covariate_context,
                value=covariate_context,
                key_padding_mask=covariate_mask
            )
            
            # Residual connection and layer norm
            if self.use_residual:
                target_attended = target_attended + target_residual
            if self.use_layer_norm:
                target_attended = self.target_norm1(target_attended)
            
            # Feed-forward network
            target_ffn_out = self.target_ffn(target_attended)
            if self.use_residual:
                target_ffn_out = target_ffn_out + target_attended
            if self.use_layer_norm:
                target_attended_final = self.target_norm2(target_ffn_out)
            else:
                target_attended_final = target_ffn_out
                
        except Exception as e:
            logger.error(f"Target-to-covariate attention failed: {e}")
            target_attended_final = target_context
            target_attention_weights = None
        
        # Cross-attention 2: Covariate attends to Target
        try:
            covariate_attended, covariate_attention_weights = self.covariate_to_target_attention(
                query=covariate_context,
                key=target_context,
                value=target_context,
                key_padding_mask=target_mask
            )
            
            # Residual connection and layer norm
            if self.use_residual:
                covariate_attended = covariate_attended + covariate_residual
            if self.use_layer_norm:
                covariate_attended = self.covariate_norm1(covariate_attended)
            
            # Feed-forward network
            covariate_ffn_out = self.covariate_ffn(covariate_attended)
            if self.use_residual:
                covariate_ffn_out = covariate_ffn_out + covariate_attended
            if self.use_layer_norm:
                covariate_attended_final = self.covariate_norm2(covariate_ffn_out)
            else:
                covariate_attended_final = covariate_ffn_out
                
        except Exception as e:
            logger.error(f"Covariate-to-target attention failed: {e}")
            covariate_attended_final = covariate_context
            covariate_attention_weights = None
        
        # Return pooled versions for consistency
        target_output = target_attended_final.mean(dim=1) if target_attended_final.dim() == 3 else target_attended_final
        covariate_output = covariate_attended_final.mean(dim=1) if covariate_attended_final.dim() == 3 else covariate_attended_final
        
        return target_output, covariate_output
    
    def get_attention_weights(self) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Return the last computed attention weights for visualization.
        
        Returns:
            Tuple of (target_to_covariate_weights, covariate_to_target_weights)
        """
        target_weights = getattr(self, '_last_target_attention_weights', None)
        covariate_weights = getattr(self, '_last_covariate_attention_weights', None)
        return target_weights, covariate_weights


class SimpleDualCrossAttention(nn.Module):
    """
    Simplified version of dual cross-attention for cases where full complexity isn't needed.
    """
    
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1
    ):
        super(SimpleDualCrossAttention, self).__init__()
        
        self.d_model = d_model
        
        # Simple linear transformations for cross-attention
        self.target_query = nn.Linear(d_model, d_model)
        self.target_key = nn.Linear(d_model, d_model)
        self.target_value = nn.Linear(d_model, d_model)
        
        self.covariate_query = nn.Linear(d_model, d_model)
        self.covariate_key = nn.Linear(d_model, d_model)
        self.covariate_value = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(
        self,
        target_context: torch.Tensor,
        covariate_context: torch.Tensor,
        target_mask: Optional[torch.Tensor] = None,
        covariate_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Simple dual cross-attention forward pass.
        """
        # Ensure 2D inputs [B, D]
        if target_context.dim() == 3:
            target_context = target_context.mean(dim=1)
        if covariate_context.dim() == 3:
            covariate_context = covariate_context.mean(dim=1)
        
        # Cross-attention: Target attends to Covariate
        target_q = self.target_query(target_context)
        covariate_k = self.covariate_key(covariate_context)
        covariate_v = self.covariate_value(covariate_context)
        
        # For 2D tensors, use dot product attention
        target_attention_score = torch.sum(target_q * covariate_k, dim=-1, keepdim=True) / (self.d_model ** 0.5)
        target_attention = torch.softmax(target_attention_score, dim=-1)
        target_attended = target_attention * covariate_v
        
        # Cross-attention: Covariate attends to Target
        covariate_q = self.covariate_query(covariate_context)
        target_k = self.target_key(target_context)
        target_v = self.target_value(target_context)
        
        covariate_attention_score = torch.sum(covariate_q * target_k, dim=-1, keepdim=True) / (self.d_model ** 0.5)
        covariate_attention = torch.softmax(covariate_attention_score, dim=-1)
        covariate_attended = covariate_attention * target_v
        
        # Fusion
        fused = self.fusion(torch.cat([target_attended, covariate_attended], dim=-1))
        fused = self.dropout(fused)
        
        return fused, target_attended, covariate_attended