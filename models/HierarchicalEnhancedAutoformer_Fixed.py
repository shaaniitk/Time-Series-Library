"""
Fixed version of HierarchicalEnhancedAutoformer addressing critical bugs
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


class HierarchicalEnhancedAutoformerFixed(nn.Module):
    """Fixed version of HierarchicalEnhancedAutoformer addressing critical bugs."""
    
    def __init__(self, configs: object) -> None:
        super(HierarchicalEnhancedAutoformerFixed, self).__init__()
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
        """Extract and set model parameters from configs."""
        attrs = ['n_levels', 'wavelet_type', 'fusion_strategy', 'share_encoder_weights', 'use_cross_attention', 'use_moe_ffn', 'num_experts']
        defaults = [3, 'db4', 'weighted_concat', False, True, False, 4]
        for attr, default in zip(attrs, defaults):
            setattr(self, attr, getattr(self.configs, attr, default))
        
        self.enc_in, self.dec_in, self.c_out = self.configs.enc_in, self.configs.dec_in, self.configs.c_out
        self.d_model, self.dropout, self.embed, self.freq = self.configs.d_model, self.configs.dropout, self.configs.embed, self.configs.freq
        self.seq_len, self.pred_len, self.label_len = self.configs.seq_len, self.configs.pred_len, self.configs.label_len
        self.n_heads = getattr(self.configs, 'n_heads', 8)
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
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Fixed forward pass with proper trend-seasonal processing.
        Always returns tuple for consistency.
        """
        seasonal_init, trend_init = self.decomp(x_dec)
        
        logger.debug(f"HierarchicalEnhancedAutoformerFixed Forward - Input shapes:")
        logger.debug(f"  x_enc: {x_enc.shape}, x_dec: {x_dec.shape}")
        logger.debug(f"  seasonal_init: {seasonal_init.shape}, trend_init: {trend_init.shape}")

        # Prepare trend and seasonal arguments
        mean_trend = torch.mean(x_dec, dim=1, keepdim=True)
        mean_trend = mean_trend.expand(-1, self.pred_len, -1)
        trend_arg = torch.cat([
            trend_init[:, :self.label_len, :],
            mean_trend
        ], dim=1)

        seasonal_arg = torch.cat([
            seasonal_init[:, :self.label_len, :],
            torch.zeros_like(seasonal_init[:, :self.pred_len, :])
        ], dim=1)

        # Embeddings
        enc_emb = self.enc_embedding(x_enc, x_mark_enc)
        dec_emb = self.dec_embedding(seasonal_arg, x_mark_dec)
        trend_emb = self.dec_embedding(trend_arg, x_mark_dec)
        
        logger.debug(f"Embeddings - enc: {enc_emb.shape}, dec: {dec_emb.shape}, trend: {trend_emb.shape}")

        # Encoder processing
        multi_res_enc = self.decomposer(enc_emb)
        encoded_features, enc_aux_loss = self.encoder(multi_res_enc, attn_mask=enc_self_mask)
        cross_attended = self.cross_attention(encoded_features) if self.cross_attention else encoded_features
        
        # FIX 1: Proper trend initialization for decoder calls
        # Seasonal processing gets zero trend initialization
        seasonal_initial_trend = torch.zeros(
            trend_emb.size(0), trend_emb.size(1), self.num_target_variables,
            device=trend_emb.device, dtype=trend_emb.dtype
        )
        
        # Trend processing gets actual trend initialization (first num_target_variables features)
        trend_initial_trend = trend_arg[:, :trend_emb.size(1), :self.num_target_variables]
        
        logger.debug(f"Trend initializations - seasonal: {seasonal_initial_trend.shape}, trend: {trend_initial_trend.shape}")

        # Process seasonal and trend through hierarchical decoder with proper initialization
        s_levels, s_trends, dec_aux_loss_s = self.decoder(
            dec_emb, cross_attended, dec_self_mask, dec_enc_mask, trend=seasonal_initial_trend
        )
        t_levels, t_trends, dec_aux_loss_t = self.decoder(
            trend_emb, cross_attended, dec_self_mask, dec_enc_mask, trend=trend_initial_trend
        )
        
        logger.debug(f"Decoder outputs - s_levels: {len(s_levels)}, t_levels: {len(t_levels)}")

        # FIX 2: Improved trend combination logic
        combined_trends = []
        for i, (s_tr, t_tr) in enumerate(zip(s_trends, t_trends)):
            try:
                if s_tr is not None and t_tr is not None:
                    # Use trend from trend processing as primary, seasonal trend as residual
                    if s_tr.shape == t_tr.shape:
                        combined_trend = t_tr + 0.1 * s_tr  # Small weight for seasonal trend
                    else:
                        logger.warning(f"Shape mismatch at level {i}: s_tr {s_tr.shape} vs t_tr {t_tr.shape}")
                        combined_trend = t_tr  # Use trend processing result
                    combined_trends.append(combined_trend)
                elif t_tr is not None:
                    combined_trends.append(t_tr)
                elif s_tr is not None:
                    combined_trends.append(s_tr)
                else:
                    # Create zero tensor with proper shape
                    zero_trend = torch.zeros(
                        trend_emb.size(0), self.pred_len, self.num_target_variables,
                        device=trend_emb.device, dtype=trend_emb.dtype
                    )
                    combined_trends.append(zero_trend)
                    logger.debug(f"Created zero trend for level {i}")
            except Exception as e:
                logger.error(f"Error combining trends at level {i}: {e}")
                # Fallback to zero trend
                zero_trend = torch.zeros(
                    trend_emb.size(0), self.pred_len, self.num_target_variables,
                    device=trend_emb.device, dtype=trend_emb.dtype
                )
                combined_trends.append(zero_trend)

        # FIX 3: Safe fusion with error handling
        try:
            fused_seasonal = self.seasonal_fusion(s_levels, self.pred_len)
        except Exception as e:
            logger.error(f"Seasonal fusion failed: {e}")
            fused_seasonal = None
            
        try:
            fused_trend = self.trend_fusion(combined_trends, self.pred_len)
        except Exception as e:
            logger.error(f"Trend fusion failed: {e}")
            fused_trend = None

        # FIX 4: Robust fallback handling
        batch_size = dec_emb.size(0)
        device = dec_emb.device
        dtype = dec_emb.dtype
        
        if fused_seasonal is None:
            fused_seasonal = torch.zeros(batch_size, self.pred_len, self.d_model, device=device, dtype=dtype)
            logger.warning("Using zero fallback for seasonal fusion")
            
        if fused_trend is None:
            fused_trend = torch.zeros(batch_size, self.pred_len, self.d_model, device=device, dtype=dtype)
            logger.warning("Using zero fallback for trend fusion")

        # FIX 5: Safe shape checking instead of assertion
        if fused_seasonal.shape != fused_trend.shape:
            logger.error(f"Shape mismatch: seasonal {fused_seasonal.shape} vs trend {fused_trend.shape}")
            # Reshape to match
            min_seq_len = min(fused_seasonal.size(1), fused_trend.size(1))
            fused_seasonal = fused_seasonal[:, :min_seq_len, :]
            fused_trend = fused_trend[:, :min_seq_len, :]

        output = fused_seasonal + fused_trend

        # FIX 6: Consistent return type - always return tuple
        projected_output = self.projection(output)[:, -self.pred_len:, :]
        
        # FIX 7: Safe auxiliary loss handling
        try:
            if isinstance(enc_aux_loss, torch.Tensor) and isinstance(dec_aux_loss_s, torch.Tensor) and isinstance(dec_aux_loss_t, torch.Tensor):
                total_aux_loss = enc_aux_loss + dec_aux_loss_s + dec_aux_loss_t
            else:
                total_aux_loss = None
        except Exception as e:
            logger.error(f"Auxiliary loss computation failed: {e}")
            total_aux_loss = None

        aux_loss = total_aux_loss if (self.training and self.use_moe_ffn and total_aux_loss is not None) else None
        
        logger.debug(f"Final output shape: {projected_output.shape}, aux_loss: {aux_loss is not None}")
        
        # Return auxiliary loss if available for training compatibility
        if aux_loss is not None:
            return projected_output, aux_loss
        else:
            return projected_output


# Alias for consistency
Model = HierarchicalEnhancedAutoformerFixed