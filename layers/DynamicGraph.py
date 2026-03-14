from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from layers.SelfAttention_Family import FullAttention, AttentionLayer

class DynamicGraphLearner(nn.Module):
    """
    Dynamic Cross-Covariate Graph Learner.
    Applies message passing across the Features (C) dimension independently at every timestep.
    Replaces static/trivial mixing with a fully adaptive sequence-aware graph layer.
    """
    def __init__(self, d_model, n_heads, dropout=0.1, output_attention=False):
        super(DynamicGraphLearner, self).__init__()
        self.output_attention = output_attention
        self.n_heads = n_heads
        
        # We reuse the highly optimized TSL AttentionLayer
        self.attention = AttentionLayer(
            FullAttention(
                mask_flag=False, 
                factor=5, 
                scale=None, 
                attention_dropout=dropout, 
                output_attention=output_attention
            ),
            d_model=d_model,
            n_heads=n_heads
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def _validate_input(self, x: torch.Tensor):
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"DynamicGraphLearner expects torch.Tensor input, got {type(x)}.")
        if x.ndim not in (3, 4):
            raise ValueError(
                f"DynamicGraphLearner expects rank-3 or rank-4 input, got shape {tuple(x.shape)}."
            )
        if not torch.isfinite(x).all():
            raise ValueError("DynamicGraphLearner input contains NaN/Inf values.")

    def _validate_attention_output(self, attn_out: torch.Tensor, attn_weights: Optional[torch.Tensor], return_attention: bool):
        if not torch.isfinite(attn_out).all():
            raise ValueError("Dynamic graph attention output contains NaN/Inf values.")
        if return_attention:
            if attn_weights is None:
                raise RuntimeError(
                    "DynamicGraphLearner was asked to return attention weights, but the attention backend returned None. "
                    "Instantiate with output_attention=True."
                )
            if not torch.isfinite(attn_weights).all():
                raise ValueError("Dynamic graph attention weights contain NaN/Inf values.")

    def _reshape_temporal_attention(self, attn_weights: torch.Tensor, batch_size: int, seq_len: int, num_nodes: int) -> torch.Tensor:
        expected_shape = (batch_size * seq_len, self.n_heads, num_nodes, num_nodes)
        if attn_weights.shape != expected_shape:
            raise ValueError(
                f"Dynamic graph attention weights must have shape {expected_shape}, got {tuple(attn_weights.shape)}."
            )
        return attn_weights.reshape(batch_size, seq_len, self.n_heads, num_nodes, num_nodes)
        
    def forward(self, x: torch.Tensor, return_attention: Optional[bool] = None) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        x shapes:
        If temporal: [B, T, C, d_model]
        If static context: [B, C, d_model]
        """
        self._validate_input(x)
        return_attention = self.output_attention if return_attention is None else return_attention
        
        if x.ndim == 4:
            B, T, C, d = x.shape
            
            # Collapse Batch and Time to treat Features acting as the sequence length
            x_flat = x.reshape(B * T, C, d)
            
            # Message passing across the graph nodes (features)
            attn_out, attn_weights = self.attention(x_flat, x_flat, x_flat, attn_mask=None)
            self._validate_attention_output(attn_out, attn_weights, return_attention)
            
            # Residual connection and stabilization
            out_flat = self.layer_norm(x_flat + self.dropout(attn_out))
            if not torch.isfinite(out_flat).all():
                raise ValueError("Dynamic graph residual output contains NaN/Inf values.")
            out = out_flat.reshape(B, T, C, d)
            
            if return_attention:
                attn_weights = self._reshape_temporal_attention(attn_weights, B, T, C)
                return out, attn_weights
            return out
            
        elif x.ndim == 3:
            B, C, _ = x.shape
            attn_out, attn_weights = self.attention(x, x, x, attn_mask=None)
            self._validate_attention_output(attn_out, attn_weights, return_attention)
            out = self.layer_norm(x + self.dropout(attn_out))
            if not torch.isfinite(out).all():
                raise ValueError("Dynamic graph residual output contains NaN/Inf values.")
            
            if return_attention:
                expected_shape = (B, self.n_heads, C, C)
                if attn_weights.shape != expected_shape:
                    raise ValueError(
                        f"Dynamic graph attention weights must have shape {expected_shape}, got {tuple(attn_weights.shape)}."
                    )
                return out, attn_weights
            return out

        raise ValueError(f"Unsupported DynamicGraphLearner input rank {x.ndim}.")
