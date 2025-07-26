"""
Hierarchical Enhanced Autoformer

This module implements a hierarchical architecture that leverages existing
wavelet infrastructure (MultiWaveletCorrelation.py and DWT_Decomposition.py)
to process time series at multiple resolutions simultaneously.

This version is significantly refactored to include:
- A symmetric HierarchicalEncoder and HierarchicalDecoder.
- A robust, standalone CustomMultiHeadAttention layer.
- An optional Gated Mixture of Experts (MoE) FFN for increased capacity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple

from models.EnhancedAutoformer import EnhancedAutoformer, LearnableSeriesDecomp, EnhancedEncoder, EnhancedDecoder, EnhancedEncoderLayer, EnhancedDecoderLayer
from layers.enhancedcomponents.EnhancedEncoder import HierarchicalEncoder
from layers.enhancedcomponents.EnhancedDecoder import HierarchicalDecoder
from layers.MultiWaveletCorrelation import MultiWaveletTransform, MultiWaveletCross
from layers.enhancedcomponents.EnhancedDecomposer import WaveletHierarchicalDecomposer
from layers.enhancedcomponents.EnhancedAttention import CrossResolutionAttention
from layers.Embed import DataEmbedding_wo_pos
from layers.CustomMultiHeadAttention import CustomMultiHeadAttention
from layers.enhancedcomponents.EnhancedFusion import HierarchicalFusion
from layers.GatedMoEFFN import GatedMoEFFN
from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelationLayer, AdaptiveAutoCorrelation
from layers.Normalization import get_norm_layer
from utils.logger import logger






class HierarchicalEnhancedAutoformer(nn.Module):
    """Complete Hierarchical Enhanced Autoformer with optional MoE FFNs."""
    def __init__(self, configs):
        super(HierarchicalEnhancedAutoformer, self).__init__()
        self.configs = configs
        self._get_params()
        
        self.enc_embedding = DataEmbedding_wo_pos(self.enc_in, self.d_model, self.embed, self.freq, self.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(self.dec_in, self.d_model, self.embed, self.freq, self.dropout)
        self.decomposer = WaveletHierarchicalDecomposer(self.seq_len, self.d_model, self.wavelet_type, self.n_levels)
        self.encoder = HierarchicalEncoder(configs, self.n_levels, self.share_encoder_weights)
        self.decoder = HierarchicalDecoder(configs, self.trend_dim, self.n_levels, self.share_encoder_weights)
        self.decomp = LearnableSeriesDecomp(self.dec_in, 25)
        self.cross_attention = CrossResolutionAttention(
            self.d_model, self.n_levels, self.use_cross_attention,
            seq_len_kv=getattr(self.configs, 'seq_len_kv', self.seq_len),
            modes=getattr(self.configs, 'modes', 16)
        ) if self.use_cross_attention else None
        self.seasonal_fusion = HierarchicalFusion(self.d_model, self.n_levels, self.fusion_strategy,
                                                 n_heads=self.n_heads)
        self.trend_fusion = HierarchicalFusion(self.d_model, self.n_levels, self.fusion_strategy,
                                              n_heads=self.n_heads)
        self.projection = nn.Linear(self.c_out, self.c_out, bias=True)

    def _get_params(self):
        attrs = ['n_levels', 'wavelet_type', 'fusion_strategy', 'share_encoder_weights', 'use_cross_attention', 'use_moe_ffn', 'num_experts']
        defaults = [3, 'db4', 'weighted_concat', False, True, False, 4]
        for attr, default in zip(attrs, defaults):
            setattr(self, attr, getattr(self.configs, attr, default))
        
        self.enc_in, self.dec_in, self.c_out = self.configs.enc_in, self.configs.dec_in, self.configs.c_out
        self.d_model, self.dropout, self.embed, self.freq = self.configs.d_model, self.configs.dropout, self.configs.embed, self.configs.freq
        self.seq_len, self.pred_len, self.label_len = self.configs.seq_len, self.configs.pred_len, self.configs.label_len

        # Initialize n_heads from configs
        self.n_heads = getattr(self.configs, 'n_heads', 8)  # Default to 8 if not present

        self.num_target_variables = getattr(self.configs, 'c_out_evaluation', self.c_out)
        self.quantiles = getattr(self.configs, 'quantile_levels', [])
        self.is_quantile_mode = bool(self.quantiles)
        self.num_quantiles = len(self.quantiles) if self.is_quantile_mode else 1
        self.trend_dim = self.num_target_variables * self.num_quantiles if self.is_quantile_mode else self.num_target_variables

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        seasonal_init, trend_init = self.decomp(x_dec)
        # --- Quantile-specific trend: expand trend for quantile dimension if needed ---
        if self.is_quantile_mode:
            # trend_init: [B, L, num_target_variables]
            trend_init = trend_init.unsqueeze(-1).repeat(1, 1, 1, self.num_quantiles)  # [B, L, num_target_variables, num_quantiles]
            trend_init = trend_init.view(trend_init.size(0), trend_init.size(1), -1)  # [B, L, num_target_variables * num_quantiles]
        trend_arg = torch.cat([
            trend_init[:, :self.label_len, :],
            torch.mean(x_dec, dim=1, keepdim=True).repeat(1, self.pred_len, trend_init.size(-1))
        ], dim=1)
        seasonal_arg = torch.cat([
            seasonal_init[:, :self.label_len, :],
            torch.zeros_like(seasonal_init[:, :self.pred_len, :])
        ], dim=1)

        enc_emb = self.enc_embedding(x_enc, x_mark_enc)
        dec_emb = self.dec_embedding(seasonal_arg, x_mark_dec)

        multi_res_enc = self.decomposer(enc_emb)
        encoded_features, enc_aux_loss = self.encoder(multi_res_enc, attn_mask=enc_self_mask)

        cross_attended = self.cross_attention(encoded_features) if self.cross_attention else encoded_features

        s_levels, t_levels, dec_aux_loss = self.decoder(dec_emb, cross_attended, dec_self_mask, dec_enc_mask, trend=trend_arg)

        fused_seasonal = self.seasonal_fusion(s_levels, self.pred_len)
        fused_trend = self.trend_fusion(t_levels, self.pred_len)

        # --- Output dimension check and projection ---
        if fused_seasonal.shape != fused_trend.shape:
            # Project both to the same shape (use c_out as target)
            fused_seasonal = nn.Linear(fused_seasonal.shape[-1], self.c_out, bias=True).to(fused_seasonal.device)(fused_seasonal)
            fused_trend = nn.Linear(fused_trend.shape[-1], self.c_out, bias=True).to(fused_trend.device)(fused_trend)
        assert fused_seasonal.shape == fused_trend.shape, f"Shape mismatch: {fused_seasonal.shape} vs {fused_trend.shape}"
        output = fused_seasonal + fused_trend

        if self.training and self.use_moe_ffn:
            return self.projection(output)[:, -self.pred_len:, :], enc_aux_loss + dec_aux_loss
        else:
            return self.projection(output)[:, -self.pred_len:, :]

# Alias for consistency
Model = HierarchicalEnhancedAutoformer