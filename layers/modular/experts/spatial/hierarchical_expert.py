"""
Hierarchical Spatial Expert

Specialized expert for hierarchical spatial dependencies in time series.
Handles multi-scale spatial structures and hierarchical relationships.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, List
import math

from ..base_expert import SpatialExpert, ExpertOutput


class HierarchicalSpatialExpert(SpatialExpert):
    """Expert specialized in hierarchical spatial pattern detection and modeling."""
    
    def __init__(self, config, hierarchy_levels: Optional[int] = None):
        super().__init__(config, 'hierarchical_spatial_expert')
        
        # Hierarchy parameters
        self.hierarchy_levels = hierarchy_levels or getattr(config, 'spatial_hierarchy_levels', 3)
        
        # Multi-scale spatial processors
        self.spatial_processors = nn.ModuleList()
        for level in range(self.hierarchy_levels):
            scale_factor = 2 ** level
            processor = nn.Sequential(
                nn.Conv1d(self.d_model if level == 0 else self.expert_dim, 
                         self.expert_dim,
                         kernel_size=min(scale_factor * 3, 15),
                         padding=min(scale_factor * 3, 15) // 2),
                nn.ReLU(),
                nn.LayerNorm(self.expert_dim)
            )
            self.spatial_processors.append(processor)
        
        # Hierarchical attention
        self.hierarchical_attention = nn.ModuleList()
        for level in range(self.hierarchy_levels):
            attention = nn.MultiheadAttention(
                self.expert_dim,
                getattr(config, 'hierarchical_heads', 4),
                dropout=self.dropout,
                batch_first=True
            )
            self.hierarchical_attention.append(attention)
        
        # Level fusion mechanism
        self.level_fusion = nn.Sequential(
            nn.Linear(self.expert_dim * self.hierarchy_levels, self.expert_dim),
            nn.ReLU(),
            nn.LayerNorm(self.expert_dim)
        )
        
        # Hierarchy strength estimator
        self.hierarchy_strength = nn.Sequential(
            nn.Linear(self.expert_dim, self.expert_dim // 4),
            nn.ReLU(),
            nn.Linear(self.expert_dim // 4, self.hierarchy_levels),
            nn.Softmax(dim=-1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.expert_dim, self.expert_dim)
        
    def forward(self, x: torch.Tensor, **kwargs) -> ExpertOutput:
        """Process input through hierarchical spatial expert."""
        batch_size, seq_len, d_model = x.shape
        
        # 1. Multi-scale spatial processing
        level_features = []
        level_attention_weights = []
        
        current_input = x
        for level, (processor, attention) in enumerate(zip(self.spatial_processors, self.hierarchical_attention)):
            # Spatial processing at current level
            if level == 0:
                processed = processor(current_input.transpose(1, 2)).transpose(1, 2)
            else:
                processed = processor(current_input.transpose(1, 2)).transpose(1, 2)
            
            # Hierarchical attention at current level
            attended, attn_weights = attention(processed, processed, processed)
            
            level_features.append(attended)
            level_attention_weights.append(attn_weights)
            
            # Prepare input for next level (downsample or use current)
            current_input = attended
        
        # 2. Fuse multi-level features
        if len(level_features) > 1:
            concatenated_features = torch.cat(level_features, dim=-1)
            fused_features = self.level_fusion(concatenated_features)
        else:
            fused_features = level_features[0]
        
        # 3. Apply spatial normalization
        spatial_features = self.spatial_norm(fused_features)
        
        # 4. Output projection
        output = self.output_proj(spatial_features)
        
        # 5. Compute hierarchy strength
        hierarchy_strengths = self.hierarchy_strength(fused_features.mean(dim=1))
        dominant_level = hierarchy_strengths.argmax(dim=-1)
        
        # 6. Compute confidence based on hierarchy coherence
        confidence_score = hierarchy_strengths.max(dim=-1)[0]
        confidence = confidence_score.unsqueeze(1).unsqueeze(1).expand(-1, seq_len, -1)
        
        # 7. Compile metadata
        metadata = {
            'hierarchy_levels': self.hierarchy_levels,
            'hierarchy_strengths': hierarchy_strengths.mean(dim=0).tolist(),
            'dominant_hierarchy_level': dominant_level.mode()[0].item(),
            'hierarchy_coherence': confidence_score.mean().item(),
            'level_attention_patterns': [attn.mean().item() for attn in level_attention_weights]
        }
        
        return ExpertOutput(
            output=output,
            confidence=confidence,
            metadata=metadata,
            attention_weights=level_attention_weights[-1],  # Use top-level attention
            uncertainty=None
        )