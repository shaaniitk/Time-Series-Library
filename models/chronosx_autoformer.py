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
        # Fast-test / performance flags
        self.force_mock_pipeline: bool = getattr(configs, 'force_mock_pipeline', False) or getattr(configs, 'force_mock_chronosx', False)
        self.disable_hf_integration: bool = getattr(configs, 'disable_hf_integration', False)

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
        self.chronos_init_error: Optional[str] = None
        if self.force_mock_pipeline:
            self.chronos_init_error = "force_mock_pipeline=True"
            self._create_mock_pipeline()
        else:
            self._initialize_chronosx_pipeline()
        # HF integration components (optional / skippable in tests)
        if not self.disable_hf_integration:
            self._initialize_hf_integration()
        # Multivariate / covariate adapter flags
        self.enable_multivariate_adapter: bool = getattr(configs, "enable_multivariate_adapter", True)
        self.enable_covariate_residual: bool = getattr(configs, "enable_covariate_residual", True)
        self.adapter_context_channel_map: Optional[List[int]] = getattr(configs, "adapter_context_channel_map", None)
        self.cov_residual_lookback: int = getattr(configs, "cov_residual_lookback", 8)
        self._build_multivariate_and_covariate_adapters()
        logger.info(f"ChronosXAutoformer initialized with {self.model_name}")
    
    def _initialize_chronosx_pipeline(self):
        """Initialize the ChronosX pipeline with fallback options."""
        self.is_mock = False
        if getattr(self, 'force_mock_chronosx', False):
            logger.info("force_mock_chronosx=True -> using mock ChronosX pipeline")
            self._create_mock_pipeline()
            return
        
        if not CHRONOS_AVAILABLE:
            msg = (
                "ChronosPipeline import failed. Install or verify version: pip install -U chronos-forecasting. "
                "Python version compatibility may also be an issue."
            )
            logger.warning(msg)
            self.chronos_init_error = "IMPORT_ERROR"
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
            err = f"Failed to load ChronosX model: {e}"
            logger.error(err)
            self.chronos_init_error = str(e)
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

    # ----------------------------------------------------------------------------------
    # Adapters: Multivariate forecasting + covariate residual correction
    # ----------------------------------------------------------------------------------
    def _build_multivariate_and_covariate_adapters(self) -> None:
        """Create optional multivariate adapter and covariate residual modules.

        Multivariate adapter strategy:
        - For c_out == 1 or adapter disabled: no-op.
        - If a channel mapping is provided (list of length c_out) and each index < enc_in,
          we use those channels directly per target.
        - Else we construct a 1x1 Conv aggregator (enc_in -> c_out) producing per-target contexts.

        Covariate residual (optional): small MLP over last k timesteps of (x_enc || x_mark_enc)
        to produce a residual [B, pred_len, c_out]. Added to adapter output (not to direct
        pipeline univariate path).
        """
        self.multivariate_adapter: Optional[nn.Module] = None
        self.covariate_residual: Optional[nn.Module] = None

        # Ensure required attributes exist (safe-guard for legacy configs)
        if not hasattr(self, 'enc_in') or not hasattr(self, 'c_out') or not hasattr(self, 'pred_len'):
            return

        if self.c_out <= 1 or not self.enable_multivariate_adapter:
            return

        class _PipelineMultivariateAdapter(nn.Module):
            """Per-target Chronos pipeline invocation with optional feature aggregation.

            Produces multivariate forecasts by invoking the underlying Chronos
            pipeline once per target channel.
            """
            def __init__(self, chronos_pipeline, enc_in: int, c_out: int,
                         pred_len: int, channel_map: Optional[List[int]] = None):
                super().__init__()
                self.chronos_pipeline = chronos_pipeline
                self.enc_in = enc_in
                self.c_out = c_out
                self.pred_len = pred_len
                self.channel_map = channel_map
                use_direct = channel_map is not None and len(channel_map) == c_out \
                    and all(0 <= ch < enc_in for ch in channel_map)
                self.use_direct = use_direct
                if not use_direct:
                    # 1x1 Conv aggregator: (B, enc_in, L) -> (B, c_out, L)
                    self.aggregator = nn.Conv1d(enc_in, c_out, kernel_size=1, bias=True)
                else:
                    self.aggregator = None

            def forward(self, x_enc: torch.Tensor) -> torch.Tensor:
                # x_enc: [B, L, enc_in]
                B, L, _ = x_enc.shape
                device = x_enc.device
                # Prepare per-target contexts per batch
                if self.use_direct:
                    # Direct channel extraction
                    per_target_series = [x_enc[:, :, ch] for ch in self.channel_map]
                else:
                    # Learned aggregation
                    agg = self.aggregator(x_enc.permute(0, 2, 1))  # [B, c_out, L]
                    per_target_series = [agg[:, i, :] for i in range(self.c_out)]

                # For each target call pipeline.predict with context list length B
                target_preds: List[torch.Tensor] = []
                for t_idx, target_context in enumerate(per_target_series):
                    context_list = [target_context[b] for b in range(B)]  # list len B, each [L]
                    try:
                        preds = self.chronos_pipeline.predict(
                            context=context_list,
                            prediction_length=self.pred_len,
                            num_samples=1
                        )
                    except Exception as e:  # graceful degrade
                        # Fallback: random noise (only in exceptional failure cases)
                        preds = torch.randn(B, self.pred_len, device=device)
                    if isinstance(preds, list):
                        # Chronos returns list of tensors shape [pred_len]
                        preds = torch.stack(preds, dim=0)  # [B, pred_len]
                    if preds.dim() == 3:
                        # [B, 1, pred_len] -> squeeze
                        preds = preds.squeeze(1)
                    target_preds.append(preds)  # each [B, pred_len]

                # Stack into [B, pred_len, c_out]
                stacked = torch.stack(target_preds, dim=-1)  # [B, pred_len, c_out]
                return stacked

        self.multivariate_adapter = _PipelineMultivariateAdapter(
            getattr(self, 'chronos_pipeline', None),
            enc_in=self.enc_in,
            c_out=self.c_out,
            pred_len=self.pred_len,
            channel_map=self.adapter_context_channel_map
        )

        if self.enable_covariate_residual:
            # Residual MLP over flattened last k timesteps of (x_enc || x_mark_enc)
            k = max(1, min(self.cov_residual_lookback, getattr(self, 'seq_len', 0) or 512))
            # Feature dimension unknown at build time; we parameterize lazily via nn.LazyLinear
            hidden = max(64, min(512, self.pred_len * self.c_out))
            self.covariate_residual = nn.Sequential(
                nn.LazyLinear(hidden),
                nn.GELU(),
                nn.Linear(hidden, self.pred_len * self.c_out)
            )
            self.cov_residual_lookback = k

    def _apply_covariate_residual(self, base: torch.Tensor, x_enc: torch.Tensor,
                                  x_mark_enc: Optional[torch.Tensor]) -> torch.Tensor:
        """Add covariate residual correction if enabled.

        base: [B, pred_len, c_out]
        Returns adjusted tensor with same shape.
        """
        if self.covariate_residual is None:
            return base
        try:
            B = x_enc.size(0)
            k = min(self.cov_residual_lookback, x_enc.size(1))
            enc_slice = x_enc[:, -k:, :]
            if x_mark_enc is not None and x_mark_enc.size(1) >= k:
                mark_slice = x_mark_enc[:, -k:, :]
                feat = torch.cat([enc_slice, mark_slice], dim=-1)  # [B, k, enc_in+Tm]
            else:
                feat = enc_slice
            flat = feat.reshape(B, -1)
            residual_flat = self.covariate_residual(flat)  # [B, pred_len*c_out]
            residual = residual_flat.view(B, self.pred_len, self.c_out)
            return base + residual
        except Exception:
            return base  # fail-safe
    
    def _initialize_hf_integration(self):
        """Initialize HuggingFace integration components."""
        if not HF_AVAILABLE:
            logger.warning("HuggingFace transformers not available")
            return
        if getattr(self, 'disable_hf_integration', False):
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
        
        # Multivariate adapter path (produces [B, pred_len, c_out])
        if self.multivariate_adapter is not None:
            multivar_pred = self.multivariate_adapter(x_enc)
            multivar_pred = self._apply_covariate_residual(multivar_pred, x_enc, x_mark_enc)
            return self.postprocess_prediction(multivar_pred)

        # Univariate or legacy path
        prepared_data = self._prepare_chronosx_input(x_enc, x_mark_enc)

        if self.uncertainty_enabled:
            predictions = self._predict_with_uncertainty(prepared_data)
        else:
            predictions = self._predict_deterministic(prepared_data)

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
            'chronos_available': CHRONOS_AVAILABLE,
            'init_error': getattr(self, 'chronos_init_error', None)
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
