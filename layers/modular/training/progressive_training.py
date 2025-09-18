"""
Progressive Training Components

Modular components for progressive training strategies that gradually
increase model complexity during training.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List
from abc import ABC, abstractmethod


class ProgressiveTrainer:
    """Progressive trainer that gradually increases model complexity."""
    
    def __init__(self, model: nn.Module, growth_strategy: str = 'layer_wise'):
        self.model = model
        self.growth_strategy = growth_strategy
        self.current_complexity = 0
        self.max_complexity = 1
        
    def increase_complexity(self):
        """Increase model complexity."""
        if self.current_complexity < self.max_complexity:
            self.current_complexity += 1
            print(f"Increased model complexity to {self.current_complexity}")
    
    def get_current_complexity(self) -> float:
        """Get current complexity ratio [0, 1]."""
        return self.current_complexity / self.max_complexity if self.max_complexity > 0 else 1.0


class ModelGrowthStrategy:
    """Strategy for growing model complexity."""
    
    def __init__(self, strategy_type: str = 'linear'):
        self.strategy_type = strategy_type
    
    def get_growth_params(self, epoch: int, total_epochs: int) -> Dict[str, Any]:
        """Get growth parameters for current epoch."""
        progress = epoch / total_epochs
        
        return {
            'progress': progress,
            'strategy': self.strategy_type
        }


class FeatureGrowthStrategy:
    """Strategy for growing feature complexity."""
    
    def __init__(self, min_features: int = 1, max_features: int = 10):
        self.min_features = min_features
        self.max_features = max_features
    
    def get_feature_count(self, progress: float) -> int:
        """Get number of features for current progress."""
        return int(self.min_features + progress * (self.max_features - self.min_features))