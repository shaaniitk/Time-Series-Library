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
    def __init__(self, configs: object) -> None:
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
        # Fusion layers: operate in d_model space before final projection
        self.seasonal_fusion = HierarchicalFusion(self.d_model, self.n_levels, self.fusion_strategy,
                                                 n_heads=self.n_heads)
        self.trend_fusion = HierarchicalFusion(self.d_model, self.n_levels, self.fusion_strategy,
                                              n_heads=self.n_heads)
        self.projection = nn.Linear(self.d_model, self.c_out, bias=True)

    def _get_params(self) -> None:
        """
        Extracts and sets model parameters from the configs object.
        Ensures all required attributes are set with appropriate defaults.
        """
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

    def forward(
        self,
        x_enc: torch.Tensor,
        x_mark_enc: torch.Tensor,
        x_dec: torch.Tensor,
        x_mark_dec: torch.Tensor,
        enc_self_mask: Optional[torch.Tensor] = None,
        dec_self_mask: Optional[torch.Tensor] = None,
        dec_enc_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass for HierarchicalEnhancedAutoformer.
        Handles quantile and non-quantile forecasting modes, ensuring correct shape for downstream modules.

        Args:
            x_enc: Encoder input tensor of shape [B, seq_len, enc_in].
            x_mark_enc: Encoder time features.
            x_dec: Decoder input tensor of shape [B, label_len, dec_in].
            x_mark_dec: Decoder time features.
            enc_self_mask: Optional encoder self-attention mask.
            dec_self_mask: Optional decoder self-attention mask.
            dec_enc_mask: Optional decoder-encoder attention mask.

        Returns:
            Model output tensor of shape [B, pred_len, c_out].
        """
        seasonal_init, trend_init = self.decomp(x_dec)

        if self.is_quantile_mode:
            # Expand trend_init for quantile dimension: [B, L, num_target_variables] -> [B, L, num_target_variables, num_quantiles]
            trend_init_expanded = trend_init.unsqueeze(-1).repeat(1, 1, 1, self.num_quantiles)
            # Compute mean trend for prediction window, expand for quantiles
            mean_trend = torch.mean(x_dec, dim=1, keepdim=True)  # [B, 1, num_target_variables]
            mean_trend_expanded = mean_trend.unsqueeze(-1).repeat(1, self.pred_len, 1, self.num_quantiles)  # [B, pred_len, num_target_variables, num_quantiles]
            # Concatenate along time (label + pred), then flatten last two dims for decoder
            trend_arg = torch.cat([
                trend_init_expanded[:, :self.label_len, :, :],
                mean_trend_expanded
            ], dim=1)
            trend_arg = trend_arg.reshape(trend_arg.size(0), self.label_len + self.pred_len, -1)
        else:
            # Standard mode: [B, L, num_target_variables]
            # Fix: Use expand() instead of repeat() to avoid dimension multiplication
            mean_trend = torch.mean(x_dec, dim=1, keepdim=True)  # [B, 1, dec_in]
            mean_trend = mean_trend.expand(-1, self.pred_len, -1)  # [B, pred_len, dec_in]
            
            trend_arg = torch.cat([
                trend_init[:, :self.label_len, :],
                mean_trend
            ], dim=1)

        seasonal_arg = torch.cat([
            seasonal_init[:, :self.label_len, :],
            torch.zeros_like(seasonal_init[:, :self.pred_len, :])
        ], dim=1)

        # For hierarchical processing, embed both seasonal and trend components
        enc_emb = self.enc_embedding(x_enc, x_mark_enc)
        dec_emb = self.dec_embedding(seasonal_arg, x_mark_dec)
        
        # Also embed the trend component for hierarchical processing
        trend_emb = self.dec_embedding(trend_arg, x_mark_dec)

        multi_res_enc = self.decomposer(enc_emb)
        encoded_features, enc_aux_loss = self.encoder(multi_res_enc, attn_mask=enc_self_mask)

        cross_attended = self.cross_attention(encoded_features) if self.cross_attention else encoded_features

        # Process both seasonal and trend through hierarchical decoder
        s_levels, _, dec_aux_loss_s = self.decoder(dec_emb, cross_attended, dec_self_mask, dec_enc_mask, trend=None)
        _, t_levels, dec_aux_loss_t = self.decoder(trend_emb, cross_attended, dec_self_mask, dec_enc_mask, trend=None)
        
        dec_aux_loss = dec_aux_loss_s + dec_aux_loss_t

        fused_seasonal = self.seasonal_fusion(s_levels, self.pred_len)
        fused_trend = self.trend_fusion(t_levels, self.pred_len)

        # Handle None fusion outputs (can happen when all levels are None)
        if fused_seasonal is None:
            fused_seasonal = torch.zeros(dec_emb.size(0), self.pred_len, self.configs.d_model, device=dec_emb.device)
        if fused_trend is None:
            fused_trend = torch.zeros(dec_emb.size(0), self.pred_len, self.configs.d_model, device=dec_emb.device)

        # Both fusion outputs should be in d_model space
        assert fused_seasonal.shape == fused_trend.shape, f"Shape mismatch: {fused_seasonal.shape} vs {fused_trend.shape}"
        output = fused_seasonal + fused_trend

        if self.training and self.use_moe_ffn:
            return self.projection(output)[:, -self.pred_len:, :], enc_aux_loss + dec_aux_loss
        else:
            return self.projection(output)[:, -self.pred_len:, :]

# Alias for consistency
Model = HierarchicalEnhancedAutoformer