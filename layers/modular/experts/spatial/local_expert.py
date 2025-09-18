"""
Local Spatial Expert

Specialized expert for local spatial dependencies in time series.
Focuses on short-range spatial interactions and neighborhood effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import math

from ..base_expert import SpatialExpert, ExpertOutput


class LocalSpatialExpert(SpatialExpert):
    """Expert specialized in local spatial pattern detection and modeling."""
    
    def __init__(self, config, neighborhood_size: Optional[int] = None):
        super().__init__(config, 'local_spatial_expert')
        
        # Local neighborhood parameters
        self.neighborhood_size = neighborhood_size or getattr(config, 'local_neighborhood_size', 3)
        
        # Local convolution layers
        self.local_conv = nn.Sequential(
            nn.Conv1d(self.d_model, self.expert_dim, 
                     kernel_size=self.neighborhood_size, 
                     padding=self.neighborhood_size//2),
            nn.ReLU(),
            nn.Conv1d(self.expert_dim, self.expert_dim, kernel_size=1),
            nn.LayerNorm(self.expert_dim)
        )
        
        # Local attention for spatial relationships
        self.local_attention = nn.MultiheadAttention(
            self.expert_dim, 
            getattr(config, 'local_spatial_heads', 4),
            dropout=self.dropout,
            batch_first=True
        )
        
        # Spatial strength estimator
        self.spatial_strength = nn.Sequential(
            nn.Linear(self.expert_dim, self.expert_dim // 4),
            nn.ReLU(),
            nn.Linear(self.expert_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.expert_dim, self.expert_dim)
        
    def forward(self, x: torch.Tensor, **kwargs) -> ExpertOutput:
        """Process input through local spatial expert."""
        batch_size, seq_len, d_model = x.shape
        
        # 1. Local spatial convolution
        local_features = self.local_conv(x.transpose(1, 2)).transpose(1, 2)
        
        # 2. Local spatial attention
        attended_features, attention_weights = self.local_attention(
            local_features, local_features, local_features
        )
        
        # 3. Apply spatial normalization
        spatial_features = self.spatial_norm(attended_features)
        
        # 4. Output projection
        output = self.output_proj(spatial_features)
        
        # 5. Compute confidence based on spatial strength
        spatial_strength = self.spatial_strength(spatial_features.mean(dim=1))
        confidence = spatial_strength.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 6. Compile metadata
        metadata = {
            'neighborhood_size': self.neighborhood_size,
            'spatial_strength_score': spatial_strength.mean().item(),
            'local_connectivity': attention_weights.mean().item()
        }
        
        return ExpertOutput(
            output=output,
            confidence=confidence,
            metadata=metadata,
            attention_weights=attention_weights,
            uncertainty=None
        )