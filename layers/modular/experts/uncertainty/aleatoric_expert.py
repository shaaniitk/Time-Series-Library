"""
Aleatoric Uncertainty Expert

Specialized expert for modeling aleatoric (data) uncertainty in time series.
Handles observation noise, measurement errors, and inherent data variability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional, Tuple
import math

from ..base_expert import UncertaintyExpert, ExpertOutput


class AleatoricUncertaintyExpert(UncertaintyExpert):
    """Expert specialized in aleatoric uncertainty quantification."""
    
    def __init__(self, config):
        super().__init__(config, 'aleatoric_uncertainty_expert')
        
        # Noise level estimator
        self.noise_estimator = nn.Sequential(
            nn.Linear(self.d_model, self.expert_dim),
            nn.ReLU(),
            nn.Linear(self.expert_dim, self.expert_dim // 2),
            nn.ReLU(),
            nn.Linear(self.expert_dim // 2, 1),
            nn.Softplus()  # Ensure positive noise estimates
        )
        
        # Heteroscedastic variance predictor
        self.variance_predictor = nn.Sequential(
            nn.Linear(self.d_model, self.expert_dim),
            nn.ReLU(),
            nn.Linear(self.expert_dim, self.expert_dim),
            nn.ReLU(),
            nn.Linear(self.expert_dim, 1),
            nn.Softplus()
        )
        
        # Temporal variance modeling
        self.temporal_variance = nn.LSTM(
            self.d_model, self.expert_dim // 2, 
            batch_first=True, dropout=self.dropout
        )
        
        # Uncertainty fusion
        self.uncertainty_fusion = nn.Sequential(
            nn.Linear(self.expert_dim // 2 + 2, self.expert_dim),  # LSTM + noise + variance
            nn.ReLU(),
            nn.Linear(self.expert_dim, 1),
            nn.Softplus()
        )
        
        # Output projection
        self.output_proj = nn.Linear(1, self.expert_dim)
        
    def forward(self, x: torch.Tensor, **kwargs) -> ExpertOutput:
        """Process input through aleatoric uncertainty expert."""
        batch_size, seq_len, d_model = x.shape
        
        # 1. Estimate noise levels
        noise_levels = self.noise_estimator(x)  # [batch_size, seq_len, 1]
        
        # 2. Predict heteroscedastic variance
        variance_pred = self.variance_predictor(x)  # [batch_size, seq_len, 1]
        
        # 3. Temporal variance modeling
        temporal_var, _ = self.temporal_variance(x)  # [batch_size, seq_len, expert_dim//2]
        
        # 4. Fuse uncertainty components
        uncertainty_input = torch.cat([
            temporal_var,
            noise_levels,
            variance_pred
        ], dim=-1)
        
        fused_uncertainty = self.uncertainty_fusion(uncertainty_input)
        
        # 5. Apply uncertainty normalization
        normalized_uncertainty = self.uncertainty_norm(fused_uncertainty)
        
        # 6. Output projection
        output = self.output_proj(normalized_uncertainty)
        
        # 7. Compute confidence (inverse of uncertainty)
        confidence = 1.0 / (1.0 + normalized_uncertainty)
        
        # 8. Compile metadata
        metadata = {
            'average_noise_level': noise_levels.mean().item(),
            'average_variance': variance_pred.mean().item(),
            'temporal_variance_magnitude': temporal_var.abs().mean().item(),
            'uncertainty_range': (normalized_uncertainty.min().item(), normalized_uncertainty.max().item())
        }
        
        return ExpertOutput(
            output=output,
            confidence=confidence,
            metadata=metadata,
            attention_weights=None,
            uncertainty=normalized_uncertainty
        )