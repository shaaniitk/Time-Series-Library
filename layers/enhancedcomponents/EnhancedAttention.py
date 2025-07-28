"""
Enhanced Cross-Resolution Attention for HierarchicalEnhancedAutoformer

This module provides a modular, reusable implementation of the cross-resolution attention logic.
"""
import torch
import torch.nn as nn
from layers.CustomMultiHeadAttention import CustomMultiHeadAttention
from layers.MultiWaveletCorrelation import MultiWaveletCross
from utils.logger import logger

class CrossResolutionAttention(nn.Module):
    """Cross-resolution attention using MultiWaveletCross or standard attention."""
    def __init__(self, d_model, n_levels, use_multiwavelet=True, n_heads=8, seq_len_kv=None, modes=None):
        super(CrossResolutionAttention, self).__init__()
        logger.info(f"Initializing CrossResolutionAttention for {n_levels} levels")
        self.n_levels, self.d_model, self.n_heads, self.use_multiwavelet = n_levels, d_model, n_heads, use_multiwavelet
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads
        
        attention_module = CustomMultiHeadAttention if not use_multiwavelet else MultiWaveletCross
        self.cross_res_attention = nn.ModuleList()
        for _ in range(n_levels - 1):
            if use_multiwavelet:
                self.cross_res_attention.append(MultiWaveletCross(
                    self.head_dim, self.head_dim, min(32, self.head_dim // 2),
                    seq_len_kv=seq_len_kv, modes=modes,
                    c=64, k=8, ich=d_model, base='legendre', activation='tanh'))
            else:
                self.cross_res_attention.append(CustomMultiHeadAttention(d_model, n_heads, dropout=0.1))
        self.cross_projections = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(n_levels - 1)])

    def forward(self, multi_res_features):
        if len(multi_res_features) < 2: return multi_res_features
        attended_features = [multi_res_features[0]]
        for i in range(len(multi_res_features) - 1):
            coarse_features = multi_res_features[i]
            fine_features = multi_res_features[i + 1]
            try:
                if self.use_multiwavelet:
                    cross_attended = self._apply_multiwavelet_cross(fine_features, coarse_features, i)
                else:
                    cross_attended = self._apply_standard_attention(fine_features, coarse_features, i)
            except (AttributeError, RuntimeError) as e:
                logger.error(f"Cross-attention failed with a critical error: {e}", exc_info=True)
                cross_attended = self._apply_standard_attention(fine_features, coarse_features, i)
            projected = self.cross_projections[i](cross_attended)
            attended_features.append(fine_features + projected)
        return attended_features

    def _align_features(self, source, target):
        if source.size(1) == target.size(1):
            return source
        return torch.nn.functional.interpolate(
            source.transpose(1, 2),
            size=target.size(1),
            mode='linear',
            align_corners=False
        ).transpose(1, 2)

    def _apply_multiwavelet_cross(self, q, kv, idx):
        B, L_q, D = q.shape
        q_4d = q.view(B, L_q, self.n_heads, self.head_dim)
        kv_4d = kv.view(B, kv.size(1), self.n_heads, self.head_dim)
        attended, _ = self.cross_res_attention[idx](q_4d, kv_4d, kv_4d, L_q, kv.size(1))
        return attended.view(B, L_q, D)

    def _apply_standard_attention(self, q, kv, idx):
        attended, _ = self.cross_res_attention[idx](query=q, key=kv, value=kv)
        return attended
