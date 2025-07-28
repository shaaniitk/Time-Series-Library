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
        
        self.dropout = nn.Dropout(dropout)
        
        # Attention scaling parameters (like EnhancedDecoderLayer)
        self.self_attn_scale = nn.Parameter(torch.ones(1) * 0.1)
        self.cross_attn_scale = nn.Parameter(torch.ones(1) * 0.1)

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        # Enhanced self-attention with residual connection
        residual = x
        new_x = self.self_attention(x, x, x, attn_mask=x_mask)[0]
        x = residual + self.dropout(self.self_attn_scale * new_x)
        x, trend1 = self.decomp1(x)

        # Enhanced cross-attention with residual connection
        residual = x
        new_x = self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0]
        x = residual + self.dropout(self.cross_attn_scale * new_x)
        x, trend2 = self.decomp2(x)

        # MoE FFN with residual connection and third decomposition
        residual = x
        y, aux_loss = self.moe_ffn(x)
        x = residual + y
        x, trend3 = self.decomp3(x)
        
        # Aggregate trends (keep in d_model space like EnhancedDecoderLayer)
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
                # Use the existing EnhancedDecoder which already handles 3-tuple returns
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
            
            # Handle different return signatures: MoE layers return 3, standard EnhancedDecoder returns 2
            decoder_output = decoder(x_aligned, cross_feats, x_mask, cross_mask, trend=trend_aligned)
            
            if len(decoder_output) == 3:
                # MoE decoder returns (seasonal, trend, aux_loss)
                s, t, loss = decoder_output
                if isinstance(loss, torch.Tensor):
                    aux_loss += loss
            else:
                # Standard EnhancedDecoder returns (seasonal, trend)
                s, t = decoder_output
            
            # Accumulate results from the current resolution level
            seasonals.append(s)
            trends.append(t)
                
        return seasonals, trends, aux_loss

    def _align(self, t, target_len):
        if t is None:
            return None
        return F.interpolate(t.transpose(1, 2), size=target_len, mode='linear').transpose(1, 2)
