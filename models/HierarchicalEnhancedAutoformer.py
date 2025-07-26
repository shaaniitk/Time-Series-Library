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
from layers.MultiWaveletCorrelation import MultiWaveletTransform, MultiWaveletCross
from layers.DWT_Decomposition import DWT1DForward, DWT1DInverse
from layers.Embed import DataEmbedding_wo_pos
from layers.CustomMultiHeadAttention import CustomMultiHeadAttention
from layers.GatedMoEFFN import GatedMoEFFN
from layers.EnhancedAutoCorrelation import AdaptiveAutoCorrelationLayer, AdaptiveAutoCorrelation
from layers.Normalization import get_norm_layer
from utils.logger import logger


class MoEEncoderLayer(nn.Module):
    """An encoder layer that uses a Gated Mixture of Experts FFN."""
    def __init__(self, attention, d_model, d_ff, num_experts=4, dropout=0.1, activation="relu"):
        super(MoEEncoderLayer, self).__init__()
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


class MoEDecoderLayer(nn.Module):
    """A decoder layer that uses a Gated Mixture of Experts FFN."""
    def __init__(self, self_attention, cross_attention, d_model, d_ff, num_experts=4, dropout=0.1, activation="relu"):
        super(MoEDecoderLayer, self).__init__()
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


class WaveletHierarchicalDecomposer(nn.Module):
    """Hierarchical decomposition using existing DWT infrastructure."""
    def __init__(self, seq_len, d_model, wavelet_type='db4', levels=3, use_learnable_weights=True):
        super(WaveletHierarchicalDecomposer, self).__init__()
        logger.info(f"Initializing WaveletHierarchicalDecomposer with {levels} levels")
        self.seq_len, self.d_model, self.levels, self.wavelet_type = seq_len, d_model, levels, wavelet_type
        self.dwt_forward = DWT1DForward(J=levels, wave=wavelet_type, mode='symmetric')
        self.dwt_inverse = DWT1DInverse(wave=wavelet_type, mode='symmetric')
        if use_learnable_weights:
            self.scale_weights = nn.Parameter(torch.ones(levels + 1) / (levels + 1))
        else:
            self.register_buffer('scale_weights', torch.ones(levels + 1) / (levels + 1))
        self.scale_projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(levels + 1)])

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        min_len = 2 ** self.levels
        padding = 0
        if seq_len < min_len:
            logger.warning(f"Sequence length {seq_len} is shorter than DWT minimum of {min_len}. Padding to {min_len}.")
            padding = min_len - seq_len
            x = F.pad(x, (0, 0, 0, padding), 'replicate')
        
        try:
            low_freq, high_freqs = self.dwt_forward(x.transpose(1, 2))
        except (ValueError, RuntimeError) as e:
            logger.error(f"DWT failed: {e}", exc_info=True)
            return self._fallback_decomposition(x[:, :-padding, :] if padding > 0 else x)

        decomposed_scales = []
        low_freq = self._resize_to_target_length(low_freq.transpose(1, 2), seq_len // (2 ** self.levels))
        decomposed_scales.append(self.scale_projections[0](low_freq))
        
        for i, high_freq in enumerate(high_freqs):
            target_length = seq_len // (2 ** (self.levels - i))
            high_freq = self._resize_to_target_length(high_freq.transpose(1, 2), target_length)
            decomposed_scales.append(self.scale_projections[i + 1](high_freq))
        return decomposed_scales

    def _resize_to_target_length(self, x, target_length):
        if x.size(1) == target_length: return x
        mode = 'linear' if x.size(1) < target_length else 'area'
        return F.interpolate(x.transpose(1, 2), size=target_length, mode=mode).transpose(1, 2)

    def _fallback_decomposition(self, x):
        logger.info("Using fallback multi-scale decomposition")
        scales = [x]
        for _ in range(self.levels):
            x = F.avg_pool1d(x.transpose(1, 2), kernel_size=2, stride=2).transpose(1, 2)
            scales.append(x)
        return scales


class CrossResolutionAttention(nn.Module):
    """Cross-resolution attention using MultiWaveletCross or standard attention."""
    def __init__(self, d_model, n_levels, use_multiwavelet=True, n_heads=8):
        super(CrossResolutionAttention, self).__init__()
        logger.info(f"Initializing CrossResolutionAttention for {n_levels} levels")
        self.n_levels, self.d_model, self.n_heads, self.use_multiwavelet = n_levels, d_model, n_heads, use_multiwavelet
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        
        attention_module = CustomMultiHeadAttention if not use_multiwavelet else MultiWaveletCross
        self.cross_res_attention = nn.ModuleList()
        for _ in range(n_levels - 1):
            if use_multiwavelet:
                self.cross_res_attention.append(MultiWaveletCross(self.head_dim, self.head_dim, min(32, self.head_dim // 2), c=64, k=8, ich=d_model, base='legendre', activation='tanh'))
            else:
                self.cross_res_attention.append(CustomMultiHeadAttention(d_model, n_heads, dropout=0.1))
        self.cross_projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_levels - 1)])

    def forward(self, multi_res_features):
        if len(multi_res_features) < 2: return multi_res_features
        attended_features = [multi_res_features[0]]
        for i in range(len(multi_res_features) - 1):
            coarse, fine = multi_res_features[i], multi_res_features[i + 1]
            aligned_coarse = self._align_features(coarse, fine)
            try:
                if self.use_multiwavelet:
                    cross_attended = self._apply_multiwavelet_cross(fine, aligned_coarse, i)
                else:
                    cross_attended = self._apply_standard_attention(fine, aligned_coarse, i)
            except (AttributeError, RuntimeError) as e:
                logger.error(f"Cross-attention failed: {e}", exc_info=True)
                cross_attended = self._apply_standard_attention(fine, aligned_coarse, i)
            attended_features.append(self.cross_projections[i](cross_attended))
        return attended_features

    def _align_features(self, source, target):
        return F.interpolate(source.transpose(1, 2), size=target.size(1), mode='linear', align_corners=False).transpose(1, 2)

    def _apply_multiwavelet_cross(self, q, kv, idx):
        B, L_q, D = q.shape
        q_4d = q.view(B, L_q, self.n_heads, self.head_dim)
        kv_4d = kv.view(B, kv.size(1), self.n_heads, self.head_dim)
        attended, _ = self.cross_res_attention[idx](q_4d, kv_4d, kv_4d, L_q, kv.size(1))
        return attended.view(B, L_q, D)

    def _apply_standard_attention(self, q, kv, idx):
        attended, _ = self.cross_res_attention[idx](query=q, key=kv, value=kv)
        return attended


class HierarchicalEncoder(nn.Module):
    """Hierarchical encoder using standard or MoE layers."""
    def __init__(self, configs, n_levels=3, share_weights=False):
        super(HierarchicalEncoder, self).__init__()
        self.use_moe_ffn = getattr(configs, 'use_moe_ffn', False)
        
        def create_encoder():
            autocorr = AdaptiveAutoCorrelationLayer(AdaptiveAutoCorrelation(True, configs.factor, configs.dropout), configs.d_model, configs.n_heads)
            layer = MoEEncoderLayer(autocorr, configs.d_model, configs.d_ff, getattr(configs, 'num_experts', 4), configs.dropout) if self.use_moe_ffn else EnhancedEncoderLayer(autocorr, configs.d_model, configs.d_ff, configs.dropout)
            return EnhancedEncoder([layer], norm_layer=get_norm_layer('LayerNorm', configs.d_model))
        
        self.resolution_encoders = nn.ModuleList([create_encoder() for _ in range(n_levels)] if not share_weights else [create_encoder()] * n_levels)

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


class HierarchicalDecoder(nn.Module):
    """Hierarchical decoder using standard or MoE layers."""
    def __init__(self, configs, trend_dim, n_levels=3, share_weights=False):
        super(HierarchicalDecoder, self).__init__()
        self.use_moe_ffn = getattr(configs, 'use_moe_ffn', False)

        def create_decoder():
            self_attn = AdaptiveAutoCorrelationLayer(AdaptiveAutoCorrelation(True, configs.factor, configs.dropout), configs.d_model, configs.n_heads)
            cross_attn = AdaptiveAutoCorrelationLayer(AdaptiveAutoCorrelation(False, configs.factor, configs.dropout), configs.d_model, configs.n_heads)
            if self.use_moe_ffn:
                layer = MoEDecoderLayer(self_attn, cross_attn, configs.d_model, configs.d_ff, getattr(configs, 'num_experts', 4), configs.dropout)
                return EnhancedDecoder([layer], norm_layer=get_norm_layer('LayerNorm', configs.d_model))
            else:
                layer = EnhancedDecoderLayer(self_attn, cross_attn, configs.d_model, trend_dim, configs.d_ff, configs.dropout)
                return EnhancedDecoder([layer], norm_layer=get_norm_layer('LayerNorm', configs.d_model), projection=nn.Linear(configs.d_model, configs.c_out, bias=True))

        self.resolution_decoders = nn.ModuleList([create_decoder() for _ in range(n_levels)] if not share_weights else [create_decoder()] * n_levels)

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


class HierarchicalFusion(nn.Module):
    """Fuses multi-resolution features."""
    def __init__(self, d_model, n_levels, fusion_strategy='weighted_concat'):
        super(HierarchicalFusion, self).__init__()
        self.fusion_strategy, self.n_levels, self.d_model = fusion_strategy, n_levels, d_model
        self.fusion_weights = nn.Parameter(torch.ones(n_levels) / n_levels)
        if fusion_strategy == 'weighted_concat':
            self.fusion_projection = nn.Sequential(nn.Linear(d_model * n_levels, d_model * 2), nn.GELU(), nn.Dropout(0.1), nn.Linear(d_model * 2, d_model))
        elif fusion_strategy == 'attention_fusion':
            self.attention_fusion = CustomMultiHeadAttention(d_model, 8, dropout=0.1)
            self.fusion_projection = nn.Linear(d_model, d_model)
        else: # weighted_sum
            self.fusion_projection = nn.Identity()

    def forward(self, features, target_len=None):
        if len(features) == 1: return features[0]
        target_len = target_len or max(f.size(1) for f in features)
        aligned = [self._align(f, target_len) for f in features]
        
        if self.fusion_strategy == 'weighted_concat':
            weighted = [f * w for f, w in zip(aligned, F.softmax(self.fusion_weights, dim=0))]
            return self.fusion_projection(torch.cat(weighted, dim=-1))
        elif self.fusion_strategy == 'weighted_sum':
            return torch.sum(torch.stack([f * w for f, w in zip(aligned, F.softmax(self.fusion_weights, dim=0))]), dim=0)
        elif self.fusion_strategy == 'attention_fusion':
            stacked = torch.stack(aligned, dim=2) # B, L, n_levels, D
            B, L, N, D = stacked.shape
            q = torch.mean(stacked, dim=2) # B, L, D
            fused, _ = self.attention_fusion(q, stacked.view(B, L, N*D), stacked.view(B, L, N*D))
            return self.fusion_projection(fused)

    def _align(self, t, target_len):
        return F.interpolate(t.transpose(1, 2), size=target_len, mode='linear').transpose(1, 2)


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
        self.cross_attention = CrossResolutionAttention(self.d_model, self.n_levels, self.use_cross_attention) if self.use_cross_attention else None
        self.seasonal_fusion = HierarchicalFusion(self.c_out, self.n_levels, self.fusion_strategy)
        self.trend_fusion = HierarchicalFusion(self.trend_dim, self.n_levels, self.fusion_strategy)
        self.projection = nn.Linear(self.c_out, self.c_out, bias=True)

    def _get_params(self):
        attrs = ['n_levels', 'wavelet_type', 'fusion_strategy', 'share_encoder_weights', 'use_cross_attention', 'use_moe_ffn', 'num_experts']
        defaults = [3, 'db4', 'weighted_concat', False, True, False, 4]
        for attr, default in zip(attrs, defaults):
            setattr(self, attr, getattr(self.configs, attr, default))
        
        self.enc_in, self.dec_in, self.c_out = self.configs.enc_in, self.configs.dec_in, self.configs.c_out
        self.d_model, self.dropout, self.embed, self.freq = self.configs.d_model, self.configs.dropout, self.configs.embed, self.configs.freq
        self.seq_len, self.pred_len, self.label_len = self.configs.seq_len, self.configs.pred_len, self.configs.label_len
        
        self.num_target_variables = getattr(self.configs, 'c_out_evaluation', self.c_out)
        self.quantiles = getattr(self.configs, 'quantile_levels', [])
        self.is_quantile_mode = bool(self.quantiles)
        self.num_quantiles = len(self.quantiles) if self.is_quantile_mode else 1
        self.trend_dim = self.num_target_variables * self.num_quantiles if self.is_quantile_mode else self.num_target_variables

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        seasonal_init, trend_init = self.decomp(x_dec)
        trend_arg = torch.cat([trend_init[:, :self.label_len, :], torch.mean(x_dec, dim=1, keepdim=True).repeat(1, self.pred_len, 1)], dim=1)
        seasonal_arg = torch.cat([seasonal_init[:, :self.label_len, :], torch.zeros_like(seasonal_init[:, :self.pred_len, :])], dim=1)
        
        enc_emb = self.enc_embedding(x_enc, x_mark_enc)
        dec_emb = self.dec_embedding(seasonal_arg, x_mark_dec)
        
        multi_res_enc = self.decomposer(enc_emb)
        encoded_features, enc_aux_loss = self.encoder(multi_res_enc, attn_mask=enc_self_mask)
        
        cross_attended = self.cross_attention(encoded_features) if self.cross_attention else encoded_features
        
        s_levels, t_levels, dec_aux_loss = self.decoder(dec_emb, cross_attended, dec_self_mask, dec_enc_mask, trend=trend_arg)
        
        fused_seasonal = self.seasonal_fusion(s_levels, self.pred_len)
        fused_trend = self.trend_fusion(t_levels, self.pred_len)
        
        output = fused_seasonal + fused_trend
        
        if self.training and self.use_moe_ffn:
            return self.projection(output)[:, -self.pred_len:, :], enc_aux_loss + dec_aux_loss
        else:
            return self.projection(output)[:, -self.pred_len:, :]

# Alias for consistency
Model = HierarchicalEnhancedAutoformer