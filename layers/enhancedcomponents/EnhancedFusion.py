"""
Enhanced Hierarchical Fusion for HierarchicalEnhancedAutoformer

This module provides a modular, reusable implementation of the hierarchical fusion logic.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.CustomMultiHeadAttention import CustomMultiHeadAttention

class HierarchicalFusion(nn.Module):
    """Fuses multi-resolution features."""
    def __init__(self, d_model, n_levels, fusion_strategy='weighted_concat', n_heads=8):
        super(HierarchicalFusion, self).__init__()
        self.fusion_strategy, self.n_levels, self.d_model = fusion_strategy, n_levels, d_model
        self.fusion_weights = nn.Parameter(torch.ones(n_levels) / n_levels)
        
        if fusion_strategy == 'weighted_concat':
            self.fusion_projection = nn.Sequential(
                nn.Linear(d_model * n_levels, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model)
            )
        elif fusion_strategy == 'attention_fusion':
            self.attention_fusion = CustomMultiHeadAttention(d_model, n_heads, dropout=0.1)
            # The original implementation used a static, learnable query.
            # A more dynamic approach is to generate the query from the input data itself.
            # We'll generate it from the finest-resolution features.
            # self.fusion_query = nn.Parameter(torch.randn(1, 1, d_model)) # Learnable query for fusion
            self.query_projection = nn.Linear(d_model, d_model) # Project finest level to get query
            self.fusion_projection = nn.Linear(d_model, d_model)
        else: # weighted_sum
            self.fusion_projection = nn.Identity()

    def forward(self, features, target_len=None):
        from utils.logger import logger
        debug_enabled = logger.isEnabledFor(10)  # DEBUG level is 10
        
        if debug_enabled:
            logger.debug(f"HierarchicalFusion Forward - {self.fusion_strategy}")
            logger.debug(f"  Input features: {len(features)} levels")
            for i, feat in enumerate(features):
                feat_shape = feat.shape if feat is not None else "None"
                logger.debug(f"    Level {i}: {feat_shape}")
        
        if len(features) == 1: return features[0]
        target_len = target_len or max(f.size(1) for f in features if f is not None)
        
        # Filter out None values and align features
        aligned = [self._align(f, target_len) for f in features if f is not None]
        
        if not aligned:
            # If all features are None, return None
            if debug_enabled:
                logger.debug("  All features are None, returning None")
            return None
        
        if debug_enabled:
            logger.debug(f"  After alignment: {len(aligned)} features, target_len: {target_len}")
            for i, feat in enumerate(aligned):
                logger.debug(f"    Aligned {i}: {feat.shape}")
        
        if self.fusion_strategy == 'weighted_concat':
            weighted = [f * w for f, w in zip(aligned, torch.nn.functional.softmax(self.fusion_weights, dim=0))]
            result = self.fusion_projection(torch.cat(weighted, dim=-1))
        elif self.fusion_strategy == 'weighted_sum':
            result = torch.sum(torch.stack([f * w for f, w in zip(aligned, torch.nn.functional.softmax(self.fusion_weights, dim=0))]), dim=0)
        elif self.fusion_strategy == 'attention_fusion':
            stacked_features = torch.stack(aligned, dim=2)  # [B, L, n_levels, D]
            B, L, N, D = stacked_features.shape
            # Use the finest resolution level (last in the list) to generate a dynamic query.
            # The original features are ordered from coarsest to finest.
            finest_level_features = aligned[-1] # [B, L, D]
            key_value = stacked_features.view(B * L, N, D)
            query = self.query_projection(finest_level_features).view(B * L, 1, D)
            fused, _ = self.attention_fusion(query=query, key=key_value, value=key_value)
            fused = fused.view(B, L, D)
            result = self.fusion_projection(fused)
        
        if debug_enabled:
            logger.debug(f"  Fusion output: {result.shape}")
        
        return result

    def _align(self, t, target_len):
        if t is None:
            return None
        return F.interpolate(t.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
