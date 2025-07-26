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
            self.fusion_query = nn.Parameter(torch.randn(1, 1, d_model)) # Learnable query for fusion
            self.fusion_projection = nn.Linear(d_model, d_model)
        else: # weighted_sum
            self.fusion_projection = nn.Identity()

    def forward(self, features, target_len=None):
        if len(features) == 1: return features[0]
        target_len = target_len or max(f.size(1) for f in features if f is not None)
        
        # Filter out None values and align features
        aligned = [self._align(f, target_len) for f in features if f is not None]
        
        if not aligned:
            # If all features are None, return None
            return None
        
        if self.fusion_strategy == 'weighted_concat':
            weighted = [f * w for f, w in zip(aligned, torch.nn.functional.softmax(self.fusion_weights, dim=0))]
            return self.fusion_projection(torch.cat(weighted, dim=-1))
        elif self.fusion_strategy == 'weighted_sum':
            return torch.sum(torch.stack([f * w for f, w in zip(aligned, torch.nn.functional.softmax(self.fusion_weights, dim=0))]), dim=0)
        elif self.fusion_strategy == 'attention_fusion':
            stacked_features = torch.stack(aligned, dim=2)  # [B, L, n_levels, D]
            B, L, N, D = stacked_features.shape
            key_value = stacked_features.view(B * L, N, D)
            query = self.fusion_query.expand(B * L, -1, -1)
            fused, _ = self.attention_fusion(query, key_value, key_value)
            fused = fused.view(B, L, D)
            return self.fusion_projection(fused)

    def _align(self, t, target_len):
        if t is None:
            return None
        return F.interpolate(t.transpose(1, 2), size=target_len, mode='linear', align_corners=False).transpose(1, 2)
