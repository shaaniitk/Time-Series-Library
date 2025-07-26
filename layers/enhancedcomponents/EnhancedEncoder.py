"""
Enhanced Hierarchical Encoder for HierarchicalEnhancedAutoformer

This module provides a modular, reusable implementation of the hierarchical encoder logic
with support for Gated Mixture of Experts (MoE) FFN layers.
"""
import torch
import torch.nn as nn
from layers.GatedMoEFFN import GatedMoEFFN
from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelationLayer, AdaptiveAutoCorrelation
from layers.Normalization import get_norm_layer
from models.EnhancedAutoformer import EnhancedEncoder, EnhancedEncoderLayer

class MoEEncoderLayer(nn.Module):
    """An encoder layer that uses a Gated Mixture of Experts FFN."""
    def __init__(self, attention, d_model, d_ff, num_experts=4, dropout=0.1, activation="relu"):
        super().__init__()
        self.attention = attention
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.moe_ffn = GatedMoEFFN(d_model, d_ff, num_experts, dropout)

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y, aux_loss = self.moe_ffn(y)
        return self.norm2(x + y), attn, aux_loss

class HierarchicalEncoder(nn.Module):
    """Hierarchical encoder using standard or MoE layers."""
    def __init__(self, configs, n_levels=3, share_weights=False):
        super().__init__()
        self.use_moe_ffn = getattr(configs, 'use_moe_ffn', False)
        def create_encoder():
            autocorr = AdaptiveAutoCorrelationLayer(
                AdaptiveAutoCorrelation(True, configs.factor, configs.dropout),
                configs.d_model, configs.n_heads)
            layer = MoEEncoderLayer(autocorr, configs.d_model, configs.d_ff, getattr(configs, 'num_experts', 4), configs.dropout) \
                if self.use_moe_ffn else EnhancedEncoderLayer(autocorr, configs.d_model, configs.d_ff, configs.dropout)
            return EnhancedEncoder([layer], norm_layer=get_norm_layer('LayerNorm', configs.d_model))
        self.resolution_encoders = nn.ModuleList(
            [create_encoder() for _ in range(n_levels)] if not share_weights else [create_encoder()] * n_levels)

    def forward(self, multi_res_features, attn_mask=None):
        encoded, aux_loss = [], 0.0
        for features, encoder in zip(multi_res_features, self.resolution_encoders):
            output = encoder(features, attn_mask=attn_mask)
            if self.use_moe_ffn and isinstance(output, tuple) and len(output) == 3:
                enc, _, loss = output
                aux_loss += loss
            else:
                enc = output[0] if isinstance(output, tuple) else output
            encoded.append(enc)
        return encoded, aux_loss
