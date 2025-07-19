"""
Unified Base Framework for Time Series Forecasting

This module provides a unified base class for both custom and HF-based time series models,
enabling seamless integration while maintaining backward compatibility.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
from abc import ABC, abstractmethod
from argparse import Namespace
import logging

logger = logging.getLogger(__name__)


class BaseTimeSeriesForecaster(nn.Module, ABC):
    """
    Unified base class for all time series forecasting models.
    
    This base class provides common interfaces and functionality that can be used
    by both custom implementations (ModularAutoformer) and HF-based implementations
    (HFAutoformerSuite, HFChronosXAutoformer).
    """
    
    def __init__(self, configs: Union[Namespace, Dict]):
        super().__init__()
        if isinstance(configs, dict):
            configs = Namespace(**configs)
        self.configs = configs
        
        # Common attributes
        self.task_name = getattr(configs, 'task_name', 'long_term_forecast')
        self.seq_len = getattr(configs, 'seq_len', 96)
        self.pred_len = getattr(configs, 'pred_len', 24)
        self.label_len = getattr(configs, 'label_len', 48)
        
        # Model dimensions
        self.enc_in = getattr(configs, 'enc_in', 7)
        self.dec_in = getattr(configs, 'dec_in', 7)
        self.c_out = getattr(configs, 'c_out', 7)
        self.d_model = getattr(configs, 'd_model', 512)
        
        # Framework identification
        self.framework_type = 'base'  # Override in subclasses
        self.model_type = 'base'      # Override in subclasses
        
        # Results storage for uncertainty and sampling
        self.last_uncertainty_results = None
        self.last_sampling_results = None
    
    @abstractmethod
    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor, 
                x_dec: torch.Tensor, x_mark_dec: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for time series forecasting.
        
        Args:
            x_enc: Encoder input [batch_size, seq_len, enc_in]
            x_mark_enc: Encoder time features [batch_size, seq_len, time_features]
            x_dec: Decoder input [batch_size, label_len + pred_len, dec_in]
            x_mark_dec: Decoder time features [batch_size, label_len + pred_len, time_features]
            mask: Optional attention mask
            
        Returns:
            Prediction tensor [batch_size, pred_len, c_out]
        """
        pass
    
    def get_uncertainty_results(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get uncertainty quantification results if available.
        
        Returns:
            Dictionary containing uncertainty information:
            - 'prediction': Mean prediction
            - 'std': Standard deviation (if available)
            - 'quantiles': Quantile predictions (if available)
            - 'samples': Raw samples (if available)
        """
        return getattr(self, 'last_uncertainty_results', None)
    
    def get_sampling_results(self) -> Optional[Dict[str, torch.Tensor]]:
        """
        Get sampling results if available.
        
        Returns:
            Dictionary containing sampling information
        """
        return getattr(self, 'last_sampling_results', None)
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary containing model metadata
        """
        return {
            'framework_type': self.framework_type,
            'model_type': self.model_type,
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'enc_in': self.enc_in,
            'dec_in': self.dec_in,
            'c_out': self.c_out,
            'd_model': self.d_model,
            'task_name': self.task_name,
            'supports_uncertainty': self.supports_uncertainty(),
            'supports_quantiles': self.supports_quantiles(),
            'parameter_count': sum(p.numel() for p in self.parameters())
        }
    
    def supports_uncertainty(self) -> bool:
        """Check if model supports uncertainty quantification."""
        return False  # Override in subclasses
    
    def supports_quantiles(self) -> bool:
        """Check if model supports quantile predictions."""
        return False  # Override in subclasses
    
    def supports_multivariate(self) -> bool:
        """Check if model supports multivariate time series."""
        return self.enc_in > 1 or self.c_out > 1
    
    def get_framework_type(self) -> str:
        """Get the framework type (custom, hf, chronosx)."""
        return self.framework_type
    
    def get_model_type(self) -> str:
        """Get the specific model type."""
        return self.model_type
    
    def validate_input_shapes(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor,
                            x_dec: torch.Tensor, x_mark_dec: torch.Tensor) -> None:
        """
        Validate input tensor shapes.
        
        Raises:
            ValueError: If input shapes are invalid
        """
        batch_size = x_enc.shape[0]
        
        # Validate encoder inputs
        if x_enc.shape != (batch_size, self.seq_len, self.enc_in):
            raise ValueError(f"Expected x_enc shape {(batch_size, self.seq_len, self.enc_in)}, "
                           f"got {x_enc.shape}")
        
        if x_mark_enc.shape[:-1] != (batch_size, self.seq_len):
            raise ValueError(f"Expected x_mark_enc batch and sequence dimensions "
                           f"{(batch_size, self.seq_len)}, got {x_mark_enc.shape[:-1]}")
        
        # Validate decoder inputs
        expected_dec_len = self.label_len + self.pred_len
        if x_dec.shape != (batch_size, expected_dec_len, self.dec_in):
            raise ValueError(f"Expected x_dec shape {(batch_size, expected_dec_len, self.dec_in)}, "
                           f"got {x_dec.shape}")
        
        if x_mark_dec.shape[:-1] != (batch_size, expected_dec_len):
            raise ValueError(f"Expected x_mark_dec batch and sequence dimensions "
                           f"{(batch_size, expected_dec_len)}, got {x_mark_dec.shape[:-1]}")
    
    def prepare_inference_data(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor,
                             x_dec: torch.Tensor, x_mark_dec: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Prepare data for inference, applying any necessary preprocessing.
        
        Returns:
            Dictionary containing prepared tensors
        """
        return {
            'x_enc': x_enc,
            'x_mark_enc': x_mark_enc,
            'x_dec': x_dec,
            'x_mark_dec': x_mark_dec
        }
    
    def postprocess_prediction(self, prediction: torch.Tensor) -> torch.Tensor:
        """
        Apply any necessary postprocessing to predictions.
        
        Args:
            prediction: Raw model output
            
        Returns:
            Processed prediction tensor
        """
        # Ensure prediction has correct shape
        if prediction.shape[-2] != self.pred_len:
            prediction = prediction[:, -self.pred_len:, :]
        
        return prediction


class CustomFrameworkMixin:
    """Mixin for custom framework specific functionality."""
    
    def get_component_info(self) -> Dict[str, Any]:
        """Get information about modular components."""
        return {}
    
    def get_backbone_info(self) -> Dict[str, Any]:
        """Get information about backbone component if available."""
        return {}


class HFFrameworkMixin:
    """Mixin for HuggingFace framework specific functionality."""
    
    def get_hf_model_info(self) -> Dict[str, Any]:
        """Get HuggingFace specific model information."""
        return {}
    
    def get_backbone_model_name(self) -> Optional[str]:
        """Get the name of the HF backbone model."""
        return None


class ChronosXMixin:
    """Mixin for ChronosX specific functionality."""
    
    def get_chronos_info(self) -> Dict[str, Any]:
        """Get ChronosX specific information."""
        return {}
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get ChronosPipeline information."""
        return {}


def create_unified_model(configs: Union[Namespace, Dict], 
                        framework_type: str = 'auto') -> BaseTimeSeriesForecaster:
    """
    Factory function to create unified time series models.
    
    Args:
        configs: Model configuration
        framework_type: Framework type ('auto', 'custom', 'hf', 'chronosx')
        
    Returns:
        Appropriate model instance based on configuration
    """
    if isinstance(configs, dict):
        configs = Namespace(**configs)
    
    # Auto-detect framework type if not specified
    if framework_type == 'auto':
        if getattr(configs, 'use_chronosx_backbone', False):
            framework_type = 'chronosx'
        elif getattr(configs, 'use_hf_backbone', False):
            framework_type = 'hf'
        else:
            framework_type = 'custom'
    
    # Import and create appropriate model
    if framework_type == 'custom':
        from models.modular_autoformer import ModularAutoformer
        return ModularAutoformer(configs)
    elif framework_type == 'hf':
        from models.HFAutoformerSuite import create_hf_model
        return create_hf_model(configs)
    elif framework_type == 'chronosx':
        from models.chronosx_autoformer import ChronosXAutoformer
        return ChronosXAutoformer(configs)
    else:
        raise ValueError(f"Unknown framework type: {framework_type}")
