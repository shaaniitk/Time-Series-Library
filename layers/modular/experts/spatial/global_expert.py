"""
Global Spatial Expert

Specialized expert for global spatial dependencies in time series.
Focuses on long-range spatial interactions and global patterns.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import math

from ..base_expert import SpatialExpert, ExpertOutput


class GlobalSpatialExpert(SpatialExpert):
    """Expert specialized in global spatial pattern detection and modeling."""
    
    def __init__(self, config):
        super().__init__(config, 'global_spatial_expert')
        
        # Global attention mechanism
        self.global_attention = nn.MultiheadAttention(
            self.expert_dim,
            getattr(config, 'global_spatial_heads', 8),
            dropout=self.dropout,
            batch_first=True
        )
        
        # Global feature aggregator
        self.global_aggregator = nn.Sequential(
            nn.Linear(self.d_model, self.expert_dim),
            nn.ReLU(),
            nn.Linear(self.expert_dim, self.expert_dim),
            nn.LayerNorm(self.expert_dim)
        )
        
        # Global pattern detector
        self.pattern_detector = nn.Sequential(
            nn.Linear(self.expert_dim, self.expert_dim // 2),
            nn.ReLU(),
            nn.Linear(self.expert_dim // 2, self.expert_dim),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.expert_dim, self.expert_dim)
        
    def forward(self, x: torch.Tensor, **kwargs) -> ExpertOutput:
        """Process input through global spatial expert."""
        batch_size, seq_len, d_model = x.shape
        
        # 1. Global feature aggregation
        global_features = self.global_aggregator(x)
        
        # 2. Global spatial attention
        attended_features, attention_weights = self.global_attention(
            global_features, global_features, global_features
        )
        
        # 3. Global pattern detection
        pattern_scores = self.pattern_detector(attended_features)
        enhanced_features = attended_features * pattern_scores
        
        # 4. Apply spatial normalization
        spatial_features = self.spatial_norm(enhanced_features)
        
        # 5. Output projection
        output = self.output_proj(spatial_features)
        
        # 6. Compute confidence based on global patterns
        global_strength = pattern_scores.mean(dim=1, keepdim=True)
        confidence = global_strength.expand(-1, seq_len, -1)
        
        # 7. Compile metadata
        metadata = {
            'global_pattern_strength': pattern_scores.mean().item(),
            'global_connectivity': attention_weights.mean().item(),
            'attention_entropy': self._compute_attention_entropy(attention_weights)
        }
        
        return ExpertOutput(
            output=output,
            confidence=confidence,
            metadata=metadata,
            attention_weights=attention_weights,
            uncertainty=None
        )
    
    def _compute_attention_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention weights to measure focus."""
        # attention_weights: [batch_size, num_heads, seq_len, seq_len]
        avg_attention = attention_weights.mean(dim=(0, 1))  # [seq_len, seq_len]
        
        # Compute entropy for each query position
        entropies = []
        for i in range(avg_attention.size(0)):
            attn_dist = avg_attention[i]
            entropy = -torch.sum(attn_dist * torch.log(attn_dist + 1e-8))
            entropies.append(entropy)
        
        return torch.stack(entropies).mean().item()