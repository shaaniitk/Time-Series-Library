
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import BaseFusion
from utils.logger import logger

class HierarchicalFusion(BaseFusion):
    """
    Fuse multi-resolution features into final output.
    """
    
    def __init__(self, d_model, n_levels, fusion_strategy='weighted_concat'):
        super(HierarchicalFusion, self).__init__()
        logger.info(f"Initializing HierarchicalFusion with strategy: {fusion_strategy}")
        
        self.fusion_strategy = fusion_strategy
        self.n_levels = n_levels
        self.d_model = d_model
        
        # Learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(n_levels) / n_levels)
        
        if fusion_strategy == 'weighted_concat':
            # Concatenate all levels and project down
            self.fusion_projection = nn.Sequential(
                nn.Linear(d_model * n_levels, d_model * 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_model * 2, d_model)
            )
        elif fusion_strategy == 'weighted_sum':
            # Simple weighted sum
            self.fusion_projection = nn.Identity()
        elif fusion_strategy == 'attention_fusion':
            # Attention-based fusion
            self.attention_fusion = nn.MultiheadAttention(
                d_model, num_heads=8, batch_first=True
            )
            self.fusion_projection = nn.Linear(d_model, d_model)
    
    def forward(self, multi_res_features, target_length=None):
        """
        Fuse multi-resolution features.
        
        Args:
            multi_res_features: List of features at different resolutions
            target_length: Target sequence length for output
            
        Returns:
            Fused feature tensor
        """
        if len(multi_res_features) == 1:
            return multi_res_features[0]
        
        # Align all features to the same sequence length
        if target_length is None:
            target_length = max(feat.size(1) for feat in multi_res_features)
        
        aligned_features = []
        for i, features in enumerate(multi_res_features):
            if features.size(1) != target_length:
                features = F.interpolate(
                    features.transpose(1, 2),
                    size=target_length,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            aligned_features.append(features)
        
        # Apply fusion strategy
        if self.fusion_strategy == 'weighted_concat':
            # Apply learnable weights to each aligned feature before concatenation
            weights = F.softmax(self.fusion_weights, dim=0)
            weighted_aligned_features = [
                features * weights[i] for i, features in enumerate(aligned_features)
            ]
            # Concatenate and project
            concatenated = torch.cat(weighted_aligned_features, dim=-1)
            fused = self.fusion_projection(concatenated)
            
        elif self.fusion_strategy == 'weighted_sum':
            # Weighted sum of features
            fused = torch.zeros_like(aligned_features[0])
            weights = F.softmax(self.fusion_weights, dim=0)
            for i, features in enumerate(aligned_features):
                fused += weights[i] * features

        elif self.fusion_strategy == 'attention_fusion':
            # Use attention to fuse features
            stacked_features = torch.stack(aligned_features, dim=2)  # [B, L, n_levels, D]
            B, L, n_levels, D = stacked_features.shape
            
            # Reshape for attention
            query = aligned_features[-1]  # Use finest scale as query
            key_value = stacked_features.view(B, L * n_levels, D)
            
            fused, _ = self.attention_fusion(query, key_value, key_value)
            fused = self.fusion_projection(fused)
        
        return fused
