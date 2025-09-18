"""
Base Expert Classes for Mixture of Experts Framework

Provides abstract base classes and common functionality for all expert types.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class ExpertOutput:
    """Standardized output format for all experts."""
    
    # Primary output
    output: torch.Tensor
    
    # Confidence/certainty score [0, 1]
    confidence: torch.Tensor
    
    # Expert-specific metadata
    metadata: Dict[str, Any]
    
    # Attention weights (if applicable)
    attention_weights: Optional[torch.Tensor] = None
    
    # Uncertainty estimates (if applicable)
    uncertainty: Optional[torch.Tensor] = None


class BaseExpert(nn.Module, ABC):
    """
    Abstract base class for all experts in the MoE framework.
    
    All experts must implement the forward method and provide
    standardized outputs through ExpertOutput.
    """
    
    def __init__(self, config, expert_type: str, expert_name: str):
        """
        Initialize base expert.
        
        Args:
            config: Model configuration
            expert_type: Type of expert ('temporal', 'spatial', 'uncertainty')
            expert_name: Unique name for this expert
        """
        super().__init__()
        self.config = config
        self.expert_type = expert_type
        self.expert_name = expert_name
        
        # Common parameters
        self.d_model = getattr(config, 'd_model', 512)
        self.dropout = getattr(config, 'dropout', 0.1)
        
        # Expert-specific parameters
        self.expert_dim = getattr(config, f'{expert_name}_dim', self.d_model)
        self.expert_layers = getattr(config, f'{expert_name}_layers', 2)
        
        # Common components
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.dropout_layer = nn.Dropout(self.dropout)
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 4),
            nn.ReLU(),
            nn.Linear(self.d_model // 4, 1),
            nn.Sigmoid()
        )
        
    @abstractmethod
    def forward(self, x: torch.Tensor, **kwargs) -> ExpertOutput:
        """
        Forward pass for the expert.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            **kwargs: Expert-specific additional arguments
            
        Returns:
            ExpertOutput with standardized format
        """
        pass
    
    def compute_confidence(self, x: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        """
        Compute confidence score for the expert's output.
        
        Args:
            x: Input tensor
            output: Expert's output tensor
            
        Returns:
            Confidence scores [batch_size, seq_len, 1]
        """
        # Use output features for confidence estimation
        confidence_input = output if output.dim() == x.dim() else x
        confidence = self.confidence_estimator(confidence_input)
        return confidence
    
    def get_expert_info(self) -> Dict[str, Any]:
        """Get information about this expert."""
        return {
            'expert_type': self.expert_type,
            'expert_name': self.expert_name,
            'expert_dim': self.expert_dim,
            'expert_layers': self.expert_layers,
            'parameters': sum(p.numel() for p in self.parameters()),
            'trainable_parameters': sum(p.numel() for p in self.parameters() if p.requires_grad)
        }


class TemporalExpert(BaseExpert):
    """Base class for temporal pattern experts."""
    
    def __init__(self, config, expert_name: str):
        super().__init__(config, 'temporal', expert_name)
        
        # Temporal-specific components
        self.temporal_conv = nn.Conv1d(
            self.d_model, self.expert_dim, 
            kernel_size=getattr(config, f'{expert_name}_kernel_size', 3),
            padding=1
        )
        
        self.temporal_norm = nn.LayerNorm(self.expert_dim)


class SpatialExpert(BaseExpert):
    """Base class for spatial relationship experts."""
    
    def __init__(self, config, expert_name: str):
        super().__init__(config, 'spatial', expert_name)
        
        # Spatial-specific components
        self.spatial_attention = nn.MultiheadAttention(
            self.expert_dim, 
            getattr(config, f'{expert_name}_heads', 4),
            dropout=self.dropout
        )
        
        self.spatial_norm = nn.LayerNorm(self.expert_dim)


class UncertaintyExpert(BaseExpert):
    """Base class for uncertainty quantification experts."""
    
    def __init__(self, config, expert_name: str):
        super().__init__(config, 'uncertainty', expert_name)
        
        # Uncertainty-specific components
        self.uncertainty_head = nn.Sequential(
            nn.Linear(self.d_model, self.expert_dim),
            nn.ReLU(),
            nn.Linear(self.expert_dim, 2)  # mean and variance
        )
        
        self.uncertainty_norm = nn.LayerNorm(2)


class MultiModalExpert(BaseExpert):
    """Expert that can handle multiple modalities (temporal + spatial)."""
    
    def __init__(self, config, expert_name: str):
        super().__init__(config, 'multimodal', expert_name)
        
        # Multi-modal fusion
        self.temporal_branch = nn.Sequential(
            nn.Linear(self.d_model, self.expert_dim),
            nn.ReLU(),
            nn.LayerNorm(self.expert_dim)
        )
        
        self.spatial_branch = nn.Sequential(
            nn.Linear(self.d_model, self.expert_dim),
            nn.ReLU(),
            nn.LayerNorm(self.expert_dim)
        )
        
        self.fusion_attention = nn.MultiheadAttention(
            self.expert_dim, 4, dropout=self.dropout
        )
        
        self.fusion_norm = nn.LayerNorm(self.expert_dim)


class AdaptiveExpert(BaseExpert):
    """Expert that adapts its behavior based on input characteristics."""
    
    def __init__(self, config, expert_name: str):
        super().__init__(config, 'adaptive', expert_name)
        
        # Adaptation mechanism
        self.adaptation_detector = nn.Sequential(
            nn.Linear(self.d_model, self.d_model // 2),
            nn.ReLU(),
            nn.Linear(self.d_model // 2, 4),  # 4 adaptation modes
            nn.Softmax(dim=-1)
        )
        
        # Multiple processing paths
        self.processing_paths = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.d_model, self.expert_dim),
                nn.ReLU(),
                nn.LayerNorm(self.expert_dim)
            ) for _ in range(4)
        ])
        
    def forward(self, x: torch.Tensor, **kwargs) -> ExpertOutput:
        """Adaptive forward pass based on input characteristics."""
        # Detect adaptation mode
        adaptation_weights = self.adaptation_detector(x.mean(dim=1))  # [batch, 4]
        
        # Process through all paths
        path_outputs = []
        for path in self.processing_paths:
            path_output = path(x)
            path_outputs.append(path_output)
        
        # Weighted combination
        path_outputs = torch.stack(path_outputs, dim=-1)  # [batch, seq, d_model, 4]
        adaptation_weights = adaptation_weights.unsqueeze(1).unsqueeze(1)  # [batch, 1, 1, 4]
        
        output = torch.sum(path_outputs * adaptation_weights, dim=-1)
        
        # Compute confidence
        confidence = self.compute_confidence(x, output)
        
        return ExpertOutput(
            output=output,
            confidence=confidence,
            metadata={
                'adaptation_weights': adaptation_weights.squeeze(),
                'expert_type': self.expert_type,
                'expert_name': self.expert_name
            }
        )