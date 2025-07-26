"""
Enhanced Hierarchical Decoder for HierarchicalEnhancedAutoformer

This module provides a modular, reusable implementation of the hierarchical decoder logic
with support for Gated Mixture of Experts (MoE) FFN layers.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.GatedMoEFFN import GatedMoEFFN
from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelationLayer, AdaptiveAutoCorrelation
from layers.Normalization import get_norm_layer
from models.EnhancedAutoformer import EnhancedDecoder, EnhancedDecoderLayer

class MoEDecoderLayer(nn.Module):
    """A decoder layer that uses a Gated Mixture of Experts FFN."""
    def __init__(self, self_attention, cross_attention, d_model, d_ff, num_experts=4, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.moe_ffn = GatedMoEFFN(d_model, d_ff, num_experts, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        y = x = self.norm2(x)
        y, aux_loss = self.moe_ffn(y)
        return self.norm3(x + y), aux_loss

class HierarchicalDecoder(nn.Module):
    """Hierarchical decoder using standard or MoE layers."""
    def __init__(self, configs, trend_dim, n_levels=3, share_weights=False):
        super().__init__()
        self.use_moe_ffn = getattr(configs, 'use_moe_ffn', False)
        def create_decoder():
            self_attn = AdaptiveAutoCorrelationLayer(
                AdaptiveAutoCorrelation(True, configs.factor, configs.dropout),
                configs.d_model, configs.n_heads)
            cross_attn = AdaptiveAutoCorrelationLayer(
                AdaptiveAutoCorrelation(False, configs.factor, configs.dropout),
                configs.d_model, configs.n_heads)
            if self.use_moe_ffn:
                layer = MoEDecoderLayer(self_attn, cross_attn, configs.d_model, configs.d_ff, getattr(configs, 'num_experts', 4), configs.dropout)
                return EnhancedDecoder([layer], configs.c_out, norm_layer=get_norm_layer('LayerNorm', configs.d_model), projection=nn.Linear(configs.d_model, configs.c_out, bias=True))
            else:
                layer = EnhancedDecoderLayer(self_attn, cross_attn, configs.d_model, trend_dim, configs.d_ff, configs.dropout)
                return EnhancedDecoder([layer], configs.c_out, norm_layer=get_norm_layer('LayerNorm', configs.d_model), projection=nn.Linear(configs.d_model, configs.c_out, bias=True))
        self.resolution_decoders = nn.ModuleList(
            [create_decoder() for _ in range(n_levels)] if not share_weights else [create_decoder()] * n_levels)

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        seasonals, trends, aux_loss = [], [], 0.0
        for decoder, cross_feats in zip(self.resolution_decoders, cross):
            target_len = cross_feats.size(1)
            x_aligned = self._align(x, target_len)
            trend_aligned = self._align(trend, target_len)
            if self.use_moe_ffn:
                output, loss = decoder(x_aligned, cross_feats, x_mask, cross_mask)
                aux_loss += loss
                s, t = output, trend_aligned # Trend/seasonal separation after fusion
            else:
                s, t = decoder(x_aligned, cross_feats, x_mask, cross_mask, trend=trend_aligned)
            seasonals.append(s)
            trends.append(t)
        return seasonals, trends, aux_loss

    def _align(self, t, target_len):
        return F.interpolate(t.transpose(1, 2), size=target_len, mode='linear').transpose(1, 2)
