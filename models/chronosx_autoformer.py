"""
ChronosX-based Autoformer with HuggingFace Integration

This module provides a unified ChronosX implementation that integrates with the
HuggingFace ecosystem while maintaining compatibility with the custom framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from argparse import Namespace
import logging

# HuggingFace imports
try:
    from transformers import AutoModel, AutoConfig, AutoTokenizer
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    AutoModel = AutoConfig = AutoTokenizer = None

# ChronosX imports
try:
    from chronos import ChronosPipeline
    CHRONOS_AVAILABLE = True
except ImportError:
    CHRONOS_AVAILABLE = False
    ChronosPipeline = None

# Unified base framework
from models.base_forecaster import BaseTimeSeriesForecaster, HFFrameworkMixin, ChronosXMixin
from utils.logger import logger


class ChronosXAutoformer(BaseTimeSeriesForecaster, HFFrameworkMixin, ChronosXMixin):
    """
    ChronosX-based Autoformer with HuggingFace integration.
    
    This model provides a seamless integration between ChronosX models and the
    HuggingFace ecosystem, offering both uncertainty quantification and
    traditional forecasting capabilities.
    """
    
    def __init__(self, configs: Union[Namespace, Dict]):
        super().__init__(configs)
        
        # Set framework identification
        self.framework_type = 'chronosx'
        self.model_type = 'chronosx_autoformer'
        
        # ChronosX configuration
        self.model_size = getattr(configs, 'chronosx_model_size', 'base')
        self.uncertainty_enabled = getattr(configs, 'uncertainty_enabled', True)
        self.num_samples = getattr(configs, 'num_samples', 20)
        self.device_target = getattr(configs, 'device', 'auto')
        
        # Model name mapping
        model_names = {
            'tiny': 'amazon/chronos-t5-tiny',
            'mini': 'amazon/chronos-t5-mini',
            'small': 'amazon/chronos-t5-small',
            'base': 'amazon/chronos-t5-base',
            'large': 'amazon/chronos-t5-large'
        }
        self.model_name = model_names.get(self.model_size, 'amazon/chronos-t5-base')
        
        # Initialize ChronosX pipeline
        self._initialize_chronosx_pipeline()
        
        # HF integration components
        self._initialize_hf_integration()
        
        logger.info(f"ChronosXAutoformer initialized with {self.model_name}")
    
    def _initialize_chronosx_pipeline(self):
        """Initialize the ChronosX pipeline with fallback options."""
        self.is_mock = False
        
        if not CHRONOS_AVAILABLE:
            logger.warning("ChronosPipeline not available. Install with: pip install chronos-forecasting")
            self._create_mock_pipeline()
            return
        
        try:
            # Set device
            if self.device_target == 'auto':
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
            else:
                device = self.device_target
            
            # Initialize ChronosX pipeline
            self.chronos_pipeline = ChronosPipeline.from_pretrained(
                self.model_name,
                device_map=device if device != 'auto' else "auto",
                torch_dtype=torch.bfloat16 if device == 'cuda' else torch.float32,
            )
            
            logger.info(f"Successfully loaded ChronosX pipeline: {self.model_name} on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load ChronosX model: {e}")
            self._create_mock_pipeline()
    
    def _create_mock_pipeline(self):
        """Create a mock pipeline for testing when ChronosX is not available."""
        self.is_mock = True
        logger.warning("Using mock ChronosX pipeline for testing")
        
        # Create a simple mock model
        class MockChronosPipeline:
            def __init__(self, d_model=512):
                self.d_model = d_model
                self.linear = nn.Linear(1, d_model)
                self.output_projection = nn.Linear(d_model, 1)
            
            def predict(self, context, prediction_length, num_samples=1, **kwargs):
                # Mock prediction logic
                batch_size = len(context) if isinstance(context, list) else context.shape[0]
                
                # Generate mock forecasts with slight noise
                mean_pred = torch.randn(batch_size, prediction_length)
                
                if num_samples > 1:
                    # Generate multiple samples for uncertainty
                    samples = torch.randn(batch_size, num_samples, prediction_length)
                    samples = samples + mean_pred.unsqueeze(1)
                    return samples
                else:
                    return mean_pred.unsqueeze(1)  # Add sample dimension
        
        self.chronos_pipeline = MockChronosPipeline(self.d_model)
    
    def _initialize_hf_integration(self):
        """Initialize HuggingFace integration components."""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace transformers not available")
            return
        
        try:
            # Initialize tokenizer for potential text integration
            self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
            
            # Input/output adaptation layers
            self.input_adapter = nn.Linear(self.enc_in, self.d_model)
            self.output_adapter = nn.Linear(self.d_model, self.c_out)
            
            # Uncertainty processing layers
            self.uncertainty_processor = nn.Sequential(
                nn.Linear(self.d_model, self.d_model // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(self.d_model // 2, self.c_out)
            )
            
        except Exception as e:
            logger.warning(f"HF integration initialization failed: {e}")
    
    def forward(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor,
                x_dec: torch.Tensor, x_mark_dec: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass using ChronosX pipeline.
        
        Args:
            x_enc: Encoder input [batch_size, seq_len, enc_in]
            x_mark_enc: Encoder time features [batch_size, seq_len, time_features]
            x_dec: Decoder input [batch_size, label_len + pred_len, dec_in]
            x_mark_dec: Decoder time features [batch_size, label_len + pred_len, time_features]
            mask: Optional attention mask
            
        Returns:
            Prediction tensor [batch_size, pred_len, c_out]
        """
        self.validate_input_shapes(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        # Prepare data for ChronosX
        prepared_data = self._prepare_chronosx_input(x_enc, x_mark_enc)
        
        # Generate predictions
        if self.uncertainty_enabled:
            predictions = self._predict_with_uncertainty(prepared_data)
        else:
            predictions = self._predict_deterministic(prepared_data)
        
        # Process output
        output = self._process_chronosx_output(predictions)
        
        return self.postprocess_prediction(output)
    
    def _prepare_chronosx_input(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> List[torch.Tensor]:
        """
        Prepare input data for ChronosX pipeline.
        
        Args:
            x_enc: Encoder input [batch_size, seq_len, enc_in]
            x_mark_enc: Time features (not used by ChronosX directly)
            
        Returns:
            List of context tensors for ChronosX
        """
        batch_size = x_enc.shape[0]
        
        if self.enc_in == 1:
            # Univariate case: use data directly
            context_list = [x_enc[i, :, 0] for i in range(batch_size)]
        else:
            # Multivariate case: use first channel or aggregate
            if hasattr(self.configs, 'chronosx_input_channel'):
                channel = self.configs.chronosx_input_channel
            else:
                channel = 0  # Use first channel by default
            
            context_list = [x_enc[i, :, channel] for i in range(batch_size)]
        
        return context_list
    
    def _predict_deterministic(self, context_list: List[torch.Tensor]) -> torch.Tensor:
        """Generate deterministic predictions."""
        predictions = self.chronos_pipeline.predict(
            context=context_list,
            prediction_length=self.pred_len,
            num_samples=1
        )
        
        if isinstance(predictions, list):
            predictions = torch.stack(predictions)
        
        # Remove sample dimension for deterministic prediction
        if predictions.dim() == 3:
            predictions = predictions.squeeze(1)
        
        return predictions
    
    def _predict_with_uncertainty(self, context_list: List[torch.Tensor]) -> torch.Tensor:
        """Generate predictions with uncertainty quantification."""
        # Generate multiple samples
        samples = self.chronos_pipeline.predict(
            context=context_list,
            prediction_length=self.pred_len,
            num_samples=self.num_samples
        )
        
        if isinstance(samples, list):
            samples = torch.stack(samples)
        
        # samples shape: [batch_size, num_samples, pred_len]
        
        # Calculate uncertainty statistics
        mean_prediction = samples.mean(dim=1)  # [batch_size, pred_len]
        std_prediction = samples.std(dim=1)    # [batch_size, pred_len]
        
        # Calculate quantiles
        quantiles = torch.quantile(samples, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]), dim=1)
        
        # Store uncertainty results
        self.last_uncertainty_results = {
            'prediction': mean_prediction,
            'std': std_prediction,
            'samples': samples,
            'quantiles': {
                'q10': quantiles[0],
                'q25': quantiles[1],
                'q50': quantiles[2],
                'q75': quantiles[3],
                'q90': quantiles[4]
            }
        }
        
        return mean_prediction
    
    def _process_chronosx_output(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Process ChronosX output to match target dimensions.
        
        Args:
            predictions: Raw ChronosX predictions [batch_size, pred_len]
            
        Returns:
            Processed predictions [batch_size, pred_len, c_out]
        """
        batch_size, pred_len = predictions.shape
        
        if self.c_out == 1:
            # Univariate output: add feature dimension
            output = predictions.unsqueeze(-1)
        else:
            # Multivariate output: expand or project
            if hasattr(self, 'output_adapter'):
                # Use neural adapter
                predictions_expanded = predictions.unsqueeze(-1).expand(-1, -1, self.d_model)
                output = self.output_adapter(predictions_expanded)
            else:
                # Simple replication
                output = predictions.unsqueeze(-1).expand(-1, -1, self.c_out)
        
        return output
    
    def supports_uncertainty(self) -> bool:
        """Check if model supports uncertainty quantification."""
        return self.uncertainty_enabled
    
    def supports_quantiles(self) -> bool:
        """Check if model supports quantile predictions."""
        return self.uncertainty_enabled
    
    def get_chronos_info(self) -> Dict[str, Any]:
        """Get ChronosX specific information."""
        return {
            'model_name': self.model_name,
            'model_size': self.model_size,
            'uncertainty_enabled': self.uncertainty_enabled,
            'num_samples': self.num_samples,
            'is_mock': self.is_mock,
            'chronos_available': CHRONOS_AVAILABLE
        }
    
    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get ChronosPipeline information."""
        if hasattr(self, 'chronos_pipeline') and not self.is_mock:
            return {
                'pipeline_type': 'ChronosPipeline',
                'model_path': self.model_name,
                'device': getattr(self.chronos_pipeline, 'device', 'unknown')
            }
        return {'pipeline_type': 'MockPipeline'}
    
    def get_hf_model_info(self) -> Dict[str, Any]:
        """Get HuggingFace specific model information."""
        return {
            'hf_available': HF_AVAILABLE,
            'model_name': self.model_name,
            'integration_type': 'chronosx_pipeline'
        }
    
    def get_backbone_model_name(self) -> Optional[str]:
        """Get the name of the HF backbone model."""
        return self.model_name


def create_chronosx_model(configs: Union[Namespace, Dict]) -> ChronosXAutoformer:
    """
    Factory function to create ChronosX models with different configurations.
    
    Args:
        configs: Model configuration
        
    Returns:
        ChronosXAutoformer instance
    """
    if isinstance(configs, dict):
        configs = Namespace(**configs)
    
    return ChronosXAutoformer(configs)
