"""
Trend Pattern Expert

Specialized expert for detecting and modeling trend patterns in time series.
Handles various trend types: linear, exponential, polynomial, and regime-based trends.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Tuple
import math

from ..base_expert import TemporalExpert, ExpertOutput


class TrendDecomposition(nn.Module):
    """Learnable trend decomposition using multiple methods."""
    
    def __init__(self, d_model: int, trend_types: list = ['linear', 'exponential', 'polynomial']):
        super().__init__()
        self.d_model = d_model
        self.trend_types = trend_types
        
        # Linear trend extractor
        if 'linear' in trend_types:
            self.linear_trend = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=1)
            )
        
        # Exponential trend extractor
        if 'exponential' in trend_types:
            self.exp_trend = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=1)
            )
        
        # Polynomial trend extractor
        if 'polynomial' in trend_types:
            self.poly_trend = nn.Sequential(
                nn.Conv1d(d_model, d_model, kernel_size=7, padding=3),
                nn.ReLU(),
                nn.Conv1d(d_model, d_model, kernel_size=1)
            )
        
        # Trend fusion
        self.trend_fusion = nn.Sequential(
            nn.Linear(d_model * len(trend_types), d_model),
            nn.ReLU(),
            nn.LayerNorm(d_model)
        )
        
        # Trend strength estimator
        self.trend_strength = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, len(trend_types)),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract trend components using multiple methods."""
        batch_size, seq_len, d_model = x.shape
        x_transposed = x.transpose(1, 2)  # [batch_size, d_model, seq_len]
        
        trend_components = []
        trend_dict = {}
        
        # Extract different trend types
        if 'linear' in self.trend_types:
            linear_trend = self.linear_trend(x_transposed).transpose(1, 2)
            trend_components.append(linear_trend)
            trend_dict['linear'] = linear_trend
        
        if 'exponential' in self.trend_types:
            exp_trend = self.exp_trend(x_transposed).transpose(1, 2)
            trend_components.append(exp_trend)
            trend_dict['exponential'] = exp_trend
        
        if 'polynomial' in self.trend_types:
            poly_trend = self.poly_trend(x_transposed).transpose(1, 2)
            trend_components.append(poly_trend)
            trend_dict['polynomial'] = poly_trend
        
        # Fuse trend components
        if len(trend_components) > 1:
            trend_concat = torch.cat(trend_components, dim=-1)
            fused_trend = self.trend_fusion(trend_concat)
        else:
            fused_trend = trend_components[0]
        
        # Estimate trend strengths
        trend_strengths = self.trend_strength(fused_trend.mean(dim=1))
        
        trend_dict.update({
            'fused_trend': fused_trend,
            'trend_strengths': trend_strengths,
            'components': trend_components
        })
        
        return trend_dict


class TrendPatternExpert(TemporalExpert):
    """Expert specialized in trend pattern detection and modeling."""
    
    def __init__(self, config, trend_types: Optional[list] = None):
        super().__init__(config, 'trend_expert')
        
        # Default trend types
        if trend_types is None:
            trend_types = ['linear', 'exponential', 'polynomial']
        self.trend_types = trend_types
        
        # Trend decomposition
        self.trend_decomp = TrendDecomposition(self.d_model, trend_types)
        
        # Trend strength estimator
        self.trend_strength = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, 1),
            nn.Sigmoid()
        )
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, self.expert_dim)
        
    def forward(self, x: torch.Tensor, **kwargs) -> ExpertOutput:
        """Process input through trend pattern expert."""
        batch_size, seq_len, d_model = x.shape
        
        # 1. Trend decomposition
        trend_results = self.trend_decomp(x)
        trend_component = trend_results['fused_trend']
        
        # 2. Apply temporal convolution and normalization
        temporal_features = self.temporal_conv(trend_component.transpose(1, 2)).transpose(1, 2)
        temporal_features = self.temporal_norm(temporal_features)
        
        # 3. Output projection
        output = self.output_proj(temporal_features)
        
        # 4. Compute confidence based on trend strength
        trend_strength = self.trend_strength(trend_component.mean(dim=1))
        confidence = trend_strength.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 5. Compile metadata
        metadata = {
            'trend_types': self.trend_types,
            'trend_strengths': trend_results['trend_strengths'],
            'trend_strength_score': trend_strength.mean().item(),
            'dominant_trend_type': self.trend_types[trend_results['trend_strengths'].mean(dim=0).argmax().item()]
        }
        
        return ExpertOutput(
            output=output,
            confidence=confidence,
            metadata=metadata,
            attention_weights=None,
            uncertainty=None
        )