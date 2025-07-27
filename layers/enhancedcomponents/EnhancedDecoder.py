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
    """A decoder layer that uses a Gated Mixture of Experts FFN and performs decomposition."""
    def __init__(self, self_attention, cross_attention, d_model, d_ff, num_experts=4, dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.moe_ffn = GatedMoEFFN(d_model, d_ff, num_experts, dropout)
        # Use LearnableSeriesDecomp from EnhancedAutoformer
        from models.EnhancedAutoformer import LearnableSeriesDecomp
        self.decomp1 = LearnableSeriesDecomp(d_model)
        self.decomp2 = LearnableSeriesDecomp(d_model)
        self.decomp3 = LearnableSeriesDecomp(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Self-attention and first decomposition
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        x, trend1 = self.decomp1(x)

        # Cross-attention and second decomposition
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x = self.norm2(x)
        x, trend2 = self.decomp2(x)

        # MoE FFN and third decomposition
        y = x
        y, aux_loss = self.moe_ffn(y)
        x = self.norm3(x + y)
        x, trend3 = self.decomp3(x)
        
        # Aggregate trends
        total_trend = trend1 + trend2 + trend3
        
        # Return seasonal, trend, and aux_loss
        return x, total_trend, aux_loss

class HierarchicalDecoder(nn.Module):
    """Hierarchical decoder using standard or MoE layers."""
    def __init__(self, configs, trend_dim, n_levels=3, share_weights=False):
        super().__init__()
        print(f"Debug: HierarchicalDecoder trend_dim={trend_dim}, d_model={configs.d_model}, c_out={configs.c_out}")
        self.use_moe_ffn = getattr(configs, 'use_moe_ffn', False)
        def create_decoder():
            self_attn = AdaptiveAutoCorrelationLayer(
                AdaptiveAutoCorrelation(True, configs.factor, configs.dropout),
                configs.d_model, configs.n_heads)
            cross_attn = AdaptiveAutoCorrelationLayer(
                AdaptiveAutoCorrelation(False, configs.factor, configs.dropout),
                configs.d_model, configs.n_heads)
            if self.use_moe_ffn:
                # MoEDecoderLayer now handles its own decomposition
                layer = MoEDecoderLayer(self_attn, cross_attn, configs.d_model, configs.d_ff, getattr(configs, 'num_experts', 4), configs.dropout)
                # The wrapper decoder no longer needs projection, as fusion happens later
                return EnhancedDecoder([layer], configs.d_model, norm_layer=get_norm_layer('LayerNorm', configs.d_model))
            else:
                layer = EnhancedDecoderLayer(self_attn, cross_attn, configs.d_model, trend_dim, configs.d_ff, configs.dropout)
                return EnhancedDecoder([layer], configs.d_model, norm_layer=get_norm_layer('LayerNorm', configs.d_model))
        self.resolution_decoders = nn.ModuleList(
            [create_decoder() for _ in range(n_levels)] if not share_weights else [create_decoder()] * n_levels)

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        seasonals, trends, aux_loss = [], [], 0.0
        for decoder, cross_feats in zip(self.resolution_decoders, cross):
            target_len = cross_feats.size(1)
            x_aligned = self._align(x, target_len)
            trend_aligned = self._align(trend, target_len)
            
            # The decoder's forward call now has a consistent signature
            output = decoder(x_aligned, cross_feats, x_mask, cross_mask, trend=trend_aligned)

            if self.use_moe_ffn:
                # Unpack the 3 return values from the MoE path
                s, t, loss = output
                aux_loss += loss
            else:
                # Unpack the 2 return values from the standard path
                s, t = output

            seasonals.append(s)
            trends.append(t)
        return seasonals, trends, aux_loss

    def _align(self, t, target_len):
        if t is None:
            return None
        return F.interpolate(t.transpose(1, 2), size=target_len, mode='linear').transpose(1, 2)
