"""
Unified Base Architecture for Time Series Models
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Unified configuration for all time series models"""
    model_family: str = "autoformer"
    model_variant: str = "base" 
    backbone_type: str = "inhouse"
    task_name: str = "long_term_forecast"
    seq_len: int = 96
    pred_len: int = 24
    enc_in: int = 7
    c_out: int = 7
    d_model: int = 512


class BaseTimeSeriesModel(nn.Module, ABC):
    """Abstract base class for all time series models"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model_family = config.model_family
        self.backbone_type = config.backbone_type
    
    @abstractmethod
    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """Unified forward pass"""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        return {
            'model_class': self.__class__.__name__,
            'model_family': self.model_family,
            'backbone_type': self.backbone_type,
            'task_name': self.config.task_name
        }


class InHouseModelBase(BaseTimeSeriesModel):
    """Base for in-house models"""
    pass


class HFModelBase(BaseTimeSeriesModel):
    """Base for HF-backed models"""
    pass


class ModelFactory:
    """Factory for creating unified models"""
    
    @staticmethod
    def create_model(model_type: str, config: ModelConfig):
        """Create model based on type"""
        if model_type == "autoformer_fixed":
            from models.Autoformer_Fixed import Model
            return Model(config)
        elif model_type == "enhanced_autoformer":
            from models.EnhancedAutoformer_Fixed import Model
            return Model(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")