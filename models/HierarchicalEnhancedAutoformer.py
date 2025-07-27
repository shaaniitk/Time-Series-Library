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
        
        # Debug logging
        from utils.logger import logger
        debug_enabled = logger.isEnabledFor(10)  # DEBUG level is 10
        
        if debug_enabled:
            logger.debug(f"HierarchicalEnhancedAutoformer Forward - Input shapes:")
            logger.debug(f"  x_enc: {x_enc.shape}, x_mark_enc: {x_mark_enc.shape}")
            logger.debug(f"  x_dec: {x_dec.shape}, x_mark_dec: {x_mark_dec.shape}")
            logger.debug(f"Series decomposition - seasonal: {seasonal_init.shape}, trend: {trend_init.shape}")

        # The logic for preparing trend and seasonal arguments is the same for both quantile and standard modes.
        # The quantile dimension is introduced by the final projection layer, not by manipulating the input to the embedding.
        # The embedding layer expects an input with `dec_in` features.

        # Prepare trend argument for the decoder's trend processing path
        mean_trend = torch.mean(x_dec, dim=1, keepdim=True)
        mean_trend = mean_trend.expand(-1, self.pred_len, -1)
        trend_arg = torch.cat([
            trend_init[:, :self.label_len, :],
            mean_trend
        ], dim=1)

        # Prepare seasonal argument for the decoder's seasonal processing path
        seasonal_arg = torch.cat([
            seasonal_init[:, :self.label_len, :],
            torch.zeros_like(seasonal_init[:, :self.pred_len, :])
        ], dim=1)

        # For hierarchical processing, embed both seasonal and trend components
        enc_emb = self.enc_embedding(x_enc, x_mark_enc)
        dec_emb = self.dec_embedding(seasonal_arg, x_mark_dec)
        
        # Also embed the trend component for hierarchical processing
        trend_emb = self.dec_embedding(trend_arg, x_mark_dec)
        
        if debug_enabled:
            logger.debug(f"Embeddings - enc: {enc_emb.shape}, dec: {dec_emb.shape}, trend: {trend_emb.shape}")

        multi_res_enc = self.decomposer(enc_emb)
        encoded_features, enc_aux_loss = self.encoder(multi_res_enc, attn_mask=enc_self_mask)

        cross_attended = self.cross_attention(encoded_features) if self.cross_attention else encoded_features
        
        if debug_enabled:
            logger.debug(f"Encoder output - encoded_features len: {len(encoded_features)}, aux_loss: {enc_aux_loss}")
            for i, feat in enumerate(encoded_features):
                logger.debug(f"  Level {i}: {feat.shape}")

        # Process both seasonal and trend through hierarchical decoder
        # For hierarchical processing, we want both components to generate outputs
        # Initialize proper trend for accumulation
        initial_trend = torch.zeros_like(trend_emb)
        
        # Pass the embedded components as decoder inputs with proper trend initialization
        s_levels, s_trends, dec_aux_loss_s = self.decoder(dec_emb, cross_attended, dec_self_mask, dec_enc_mask, trend=initial_trend)
        t_levels, t_trends, dec_aux_loss_t = self.decoder(trend_emb, cross_attended, dec_self_mask, dec_enc_mask, trend=initial_trend)
        
        if debug_enabled:
            logger.debug(f"Decoder output - s_levels len: {len(s_levels)}, t_levels len: {len(t_levels)}")
            for i, (s, t, s_tr, t_tr) in enumerate(zip(s_levels, t_levels, s_trends, t_trends)):
                s_shape = s.shape if s is not None else "None"
                t_shape = t.shape if t is not None else "None"
                s_tr_shape = s_tr.shape if s_tr is not None else "None"
                t_tr_shape = t_tr.shape if t_tr is not None else "None"
                logger.debug(f"  Level {i}: seasonal={s_shape}, trend={t_shape}, s_trend={s_tr_shape}, t_trend={t_tr_shape}")
        
        # Use the seasonal outputs for seasonal fusion and trend outputs for trend fusion
        # The key insight: s_levels contains seasonal components, s_trends contains trend components from seasonal processing
        # Similarly: t_levels contains seasonal-like processing of trend input, t_trends contains trend components
        dec_aux_loss = dec_aux_loss_s + dec_aux_loss_t

        fused_seasonal = self.seasonal_fusion(s_levels, self.pred_len)
        # Use the trend components from both decoders - combine s_trends and t_trends
        combined_trends = []
        for s_tr, t_tr in zip(s_trends, t_trends):
            if s_tr is not None and t_tr is not None:
                combined_trends.append(s_tr + t_tr)  # Combine trend components
            elif s_tr is not None:
                combined_trends.append(s_tr)
            elif t_tr is not None:
                combined_trends.append(t_tr)
            else:
                combined_trends.append(None)
        
        fused_trend = self.trend_fusion(combined_trends, self.pred_len)

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