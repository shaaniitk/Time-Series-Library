"""
Processor Components for Modular Framework

This module provides processor implementations that transform data between
components in the modular pipeline.
"""

import logging
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple

# Import existing wavelet infrastructure
try:
    from ....DWT_Decomposition import DWT1DForward, DWT1DInverse, Decomposition
    from ....MultiWaveletCorrelation import MultiWaveletTransform
    WAVELET_AVAILABLE = True
except ImportError as e:
    WAVELET_AVAILABLE = False
    DWT1DForward = None
    DWT1DInverse = None
    Decomposition = None
    MultiWaveletTransform = None
    logging.warning(f"Wavelet components not available: {e}")

from ..base_interfaces import BaseProcessor

logger = logging.getLogger(__name__)


class WaveletProcessor(BaseProcessor):
    """
    Wavelet decomposition processor leveraging existing DWT infrastructure.
    
    Uses the existing layers.DWT_Decomposition module for robust wavelet processing.
    Implements the hybrid/residual fitting approach for better learning.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize WaveletProcessor
        
        Args:
            config: Configuration dictionary with:
                - wavelet_type: Type of wavelet ('db4', 'haar', etc.)
                - decomposition_levels: Number of decomposition levels
                - mode: Decomposition mode ('symmetric', 'periodization', etc.)
                - residual_learning: Whether to use residual learning approach
                - baseline_method: Method for baseline forecast ('simple', 'arima', 'none')
                - device: Device for computations
        """
        # Create a mock config object for BaseComponent
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        if not WAVELET_AVAILABLE:
            raise ImportError("Wavelet components not available. Please install pywt and check DWT_Decomposition.py")
        
        self.wavelet_type = config.get('wavelet_type', 'db4')
        self.decomposition_levels = config.get('decomposition_levels', 3)
        self.mode = config.get('mode', 'symmetric')
        self.residual_learning = config.get('residual_learning', True)
        self.baseline_method = config.get('baseline_method', 'simple')
        self.device = config.get('device', 'cpu')
        
        # Initialize DWT components using existing infrastructure
        self.dwt_forward = DWT1DForward(
            J=self.decomposition_levels,
            wave=self.wavelet_type,
            mode=self.mode
        ).to(self.device)
        
        self.dwt_inverse = DWT1DInverse(
            wave=self.wavelet_type,
            mode=self.mode
        ).to(self.device)
        
        # For more complex scenarios, we can use the full Decomposition class
        self.use_full_decomposition = config.get('use_full_decomposition', False)
        if self.use_full_decomposition:
            # This requires more parameters - simplified for now
            self.decomposition = None  # Can be initialized when needed
        
        # Baseline forecasting components (if residual learning is enabled)
        if self.residual_learning:
            self._init_baseline_components()
        
        logger.info(f"WaveletProcessor initialized:")
        logger.info(f"  - Wavelet type: {self.wavelet_type}")
        logger.info(f"  - Levels: {self.decomposition_levels}")
        logger.info(f"  - Residual learning: {self.residual_learning}")
        logger.info(f"  - Baseline method: {self.baseline_method}")
    
    def _init_baseline_components(self):
        """Initialize components for baseline forecasting"""
        
        if self.baseline_method == 'simple':
            # Simple baseline: use low-frequency components for trend prediction
            self.baseline_predictor = SimpleWaveletBaseline()
            
        elif self.baseline_method == 'arima':
            # ARIMA-like baseline using wavelet coefficients
            self.baseline_predictor = WaveletARIMABaseline()
            
        elif self.baseline_method == 'none':
            # No baseline, just decompose and let model learn everything
            self.baseline_predictor = None
        
        else:
            logger.warning(f"Unknown baseline method: {self.baseline_method}, using 'simple'")
            self.baseline_predictor = SimpleWaveletBaseline()
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process input through wavelet decomposition
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            
        Returns:
            Dictionary containing:
                - 'processed': Main output for downstream models
                - 'components': Wavelet components (optional)
                - 'baseline': Baseline forecast (if residual learning)
                - 'residuals': Residuals for model to learn (if residual learning)
        """
        batch_size, seq_len, n_features = x.shape
        
        # Handle multi-feature case by processing each feature separately
        if n_features == 1:
            return self._process_single_feature(x.squeeze(-1))
        else:
            return self._process_multi_feature(x)
    
    def _process_single_feature(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process single feature time series
        
        Args:
            x: [batch_size, seq_len]
            
        Returns:
            Processed outputs dictionary
        """
        batch_size, seq_len = x.shape
        
        # Step 1: Wavelet decomposition using existing infrastructure
        try:
            # Apply DWT - note the input format expected by DWT1DForward
            # It expects [batch, channels, seq_len], so we add a channel dimension
            x_for_dwt = x.unsqueeze(1)  # [batch_size, 1, seq_len]
            
            # Perform forward DWT
            yl, yh = self.dwt_forward(x_for_dwt)
            # yl: low-frequency (approximation) coefficients [batch, 1, reduced_len]
            # yh: list of high-frequency (detail) coefficients at each level
            
        except Exception as e:
            logger.error(f"DWT forward failed: {e}")
            # Fallback: return original data
            return {
                'processed': x.unsqueeze(-1),  # Add feature dimension back
                'components': {'original': x},
                'baseline': None,
                'residuals': None
            }
        
        # Step 2: Create baseline forecast if residual learning is enabled
        baseline_forecast = None
        residuals = None
        
        if self.residual_learning and self.baseline_predictor is not None:
            try:
                # Create baseline forecast using wavelet components
                baseline_forecast = self.baseline_predictor.predict(yl, yh, seq_len)
                
                # Calculate residuals
                residuals = x - baseline_forecast
                
                # The main output for downstream models is the residuals
                processed_output = residuals.unsqueeze(-1)  # [batch_size, seq_len, 1]
                
            except Exception as e:
                logger.warning(f"Baseline prediction failed: {e}, using original data")
                processed_output = x.unsqueeze(-1)
                baseline_forecast = None
                residuals = None
        else:
            # No residual learning, just use original data
            processed_output = x.unsqueeze(-1)
        
        # Step 3: Prepare components for analysis/debugging
        components = {
            'low_freq': yl.squeeze(1) if yl is not None else None,  # [batch_size, reduced_len]
            'high_freq': [h.squeeze(1) for h in yh] if yh else [],  # List of [batch_size, reduced_len]
            'original': x
        }
        
        return {
            'processed': processed_output,
            'components': components,
            'baseline': baseline_forecast.unsqueeze(-1) if baseline_forecast is not None else None,
            'residuals': residuals.unsqueeze(-1) if residuals is not None else None
        }
    
    def _process_multi_feature(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Process multi-feature time series by handling each feature separately
        
        Args:
            x: [batch_size, seq_len, n_features]
            
        Returns:
            Processed outputs dictionary
        """
        batch_size, seq_len, n_features = x.shape
        
        processed_features = []
        all_components = []
        all_baselines = []
        all_residuals = []
        
        # Process each feature independently
        for feature_idx in range(n_features):
            feature_data = x[:, :, feature_idx]  # [batch_size, seq_len]
            
            # Process this feature
            feature_result = self._process_single_feature(feature_data)
            
            processed_features.append(feature_result['processed'])
            all_components.append(feature_result['components'])
            
            if feature_result['baseline'] is not None:
                all_baselines.append(feature_result['baseline'])
            if feature_result['residuals'] is not None:
                all_residuals.append(feature_result['residuals'])
        
        # Combine results
        processed_output = torch.cat(processed_features, dim=-1)  # [batch_size, seq_len, n_features]
        
        # Aggregate baselines and residuals if available
        baseline_output = None
        residuals_output = None
        
        if all_baselines:
            baseline_output = torch.cat(all_baselines, dim=-1)
        if all_residuals:
            residuals_output = torch.cat(all_residuals, dim=-1)
        
        return {
            'processed': processed_output,
            'components': all_components,  # List of component dicts, one per feature
            'baseline': baseline_output,
            'residuals': residuals_output
        }
    
    def inverse_transform(self, 
                         processed_output: torch.Tensor,
                         components: Dict[str, Any],
                         baseline: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Reconstruct original signal from processed output
        
        Args:
            processed_output: Output from downstream model [batch_size, seq_len, features]
            components: Components from forward pass
            baseline: Baseline forecast if available
            
        Returns:
            Reconstructed signal [batch_size, seq_len, features]
        """
        
        if baseline is not None and self.residual_learning:
            # Add baseline back to residual predictions
            reconstructed = processed_output + baseline
        else:
            reconstructed = processed_output
        
        return reconstructed
    
    def get_output_dim(self) -> int:
        """Return the output dimension of this processor"""
        # WaveletProcessor preserves input dimension
        return None  # Dynamic based on input
    
    def process_sequence(self, 
                        embedded_input: torch.Tensor,
                        backbone_output: torch.Tensor,
                        target_length: int,
                        **kwargs) -> torch.Tensor:
        """
        Process sequence using wavelet decomposition
        
        Args:
            embedded_input: Embedded input sequence [batch, seq_len, d_model]
            backbone_output: Output from backbone model [batch, seq_len, d_model]
            target_length: Desired output sequence length
            **kwargs: Additional processing parameters
            
        Returns:
            Processed sequence [batch, target_length, d_model]
        """
        # Apply wavelet processing to backbone output
        wavelet_result = self.forward(backbone_output)
        
        # Get the main processed output
        processed = wavelet_result['processed']
        
        # Handle target length adjustment if needed
        if processed.size(1) != target_length:
            if processed.size(1) > target_length:
                # Truncate
                processed = processed[:, :target_length, :]
            else:
                # Pad or extrapolate
                diff = target_length - processed.size(1)
                padding = processed[:, -1:, :].repeat(1, diff, 1)
                processed = torch.cat([processed, padding], dim=1)
        
        return processed

    def get_processor_type(self) -> str:
        """Return processor type"""
        return "wavelet"
    
    def get_config(self) -> Dict[str, Any]:
        """Return processor configuration"""
        return {
            'type': 'wavelet',
            'wavelet_type': self.wavelet_type,
            'decomposition_levels': self.decomposition_levels,
            'residual_learning': self.residual_learning,
            'baseline_method': self.baseline_method
        }


class SimpleWaveletBaseline(nn.Module):
    """
    Simple baseline predictor using wavelet coefficients
    
    Creates predictions using low-frequency components and simple extrapolation
    """
    
    def __init__(self):
        super().__init__()
    
    def predict(self, yl: torch.Tensor, yh: List[torch.Tensor], target_length: int) -> torch.Tensor:
        """
        Create baseline forecast from wavelet components
        
        Args:
            yl: Low-frequency coefficients [batch_size, 1, reduced_len]
            yh: High-frequency coefficients (list)
            target_length: Target sequence length
            
        Returns:
            Baseline forecast [batch_size, target_length]
        """
        batch_size = yl.size(0)
        
        # Simple approach: use the trend from low-frequency coefficients
        low_freq_data = yl.squeeze(1)  # [batch_size, reduced_len]
        
        # Extrapolate the low-frequency trend
        # Simple linear extrapolation from last few points
        if low_freq_data.size(1) >= 2:
            # Calculate slope from last two points
            last_point = low_freq_data[:, -1]
            second_last = low_freq_data[:, -2]
            slope = last_point - second_last
            
            # Create linear trend forecast
            time_steps = torch.arange(target_length, device=yl.device, dtype=yl.dtype)
            baseline = last_point.unsqueeze(-1) + slope.unsqueeze(-1) * time_steps.unsqueeze(0)
        else:
            # Fallback: constant forecast
            baseline = low_freq_data[:, -1:].expand(-1, target_length)
        
        return baseline


class WaveletARIMABaseline(nn.Module):
    """
    More sophisticated baseline using ARIMA-like approach on wavelet coefficients
    """
    
    def __init__(self):
        super().__init__()
        # This would implement a more sophisticated baseline
        # For now, fallback to simple baseline
        self.simple_baseline = SimpleWaveletBaseline()
    
    def predict(self, yl: torch.Tensor, yh: List[torch.Tensor], target_length: int) -> torch.Tensor:
        """Create ARIMA-like baseline forecast"""
        # For now, use simple baseline
        # This could be extended with proper ARIMA implementation
        return self.simple_baseline.predict(yl, yh, target_length)


class NormalizationProcessor(BaseProcessor):
    """
    Data normalization processor with multiple strategies
    """
    
    def __init__(self, config: Dict[str, Any]):
        # Create a mock config object for BaseComponent  
        from dataclasses import dataclass
        @dataclass
        class MockConfig:
            pass
        
        super().__init__(MockConfig())
        
        self.normalization_type = config.get('type', 'standard')  # 'standard', 'minmax', 'robust'
        self.feature_wise = config.get('feature_wise', True)
        self.eps = config.get('eps', 1e-8)
        
        # Statistics storage
        self.stats = {}
        self.fitted = False
        
        logger.info(f"NormalizationProcessor initialized: {self.normalization_type}")
    
    def forward(self, x: torch.Tensor, fit: bool = True) -> Dict[str, torch.Tensor]:
        """
        Normalize input data
        
        Args:
            x: Input tensor [batch_size, seq_len, features]
            fit: Whether to fit statistics on this data
            
        Returns:
            Dictionary with normalized data
        """
        
        if fit or not self.fitted:
            self._fit_statistics(x)
        
        normalized = self._normalize(x)
        
        return {
            'processed': normalized,
            'original': x,
            'stats': self.stats
        }
    
    def _fit_statistics(self, x: torch.Tensor):
        """Fit normalization statistics"""
        
        if self.feature_wise:
            # Compute statistics per feature
            if self.normalization_type == 'standard':
                self.stats['mean'] = x.mean(dim=(0, 1), keepdim=True)
                self.stats['std'] = x.std(dim=(0, 1), keepdim=True) + self.eps
            elif self.normalization_type == 'minmax':
                self.stats['min'] = x.min(dim=0, keepdim=True)[0].min(dim=1, keepdim=True)[0]
                self.stats['max'] = x.max(dim=0, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        else:
            # Global statistics
            if self.normalization_type == 'standard':
                self.stats['mean'] = x.mean()
                self.stats['std'] = x.std() + self.eps
            elif self.normalization_type == 'minmax':
                self.stats['min'] = x.min()
                self.stats['max'] = x.max()
        
        self.fitted = True
    
    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization"""
        
        if self.normalization_type == 'standard':
            return (x - self.stats['mean']) / self.stats['std']
        elif self.normalization_type == 'minmax':
            return (x - self.stats['min']) / (self.stats['max'] - self.stats['min'] + self.eps)
        else:
            return x
    
    def inverse_transform(self, normalized_x: torch.Tensor) -> torch.Tensor:
        """Inverse normalization"""
        
        if not self.fitted:
            logger.warning("Processor not fitted, returning input")
            return normalized_x
        
        if self.normalization_type == 'standard':
            return normalized_x * self.stats['std'] + self.stats['mean']
        elif self.normalization_type == 'minmax':
            return normalized_x * (self.stats['max'] - self.stats['min']) + self.stats['min']
        else:
            return normalized_x
    
    def get_output_dim(self) -> int:
        """Return the output dimension of this processor"""
        # NormalizationProcessor preserves input dimension
        return None  # Dynamic based on input
    
    def process_sequence(self, 
                        embedded_input: torch.Tensor,
                        backbone_output: torch.Tensor,
                        target_length: int,
                        **kwargs) -> torch.Tensor:
        """
        Process sequence with normalization
        
        Args:
            embedded_input: Embedded input sequence [batch, seq_len, d_model]
            backbone_output: Output from backbone model [batch, seq_len, d_model]
            target_length: Desired output sequence length
            **kwargs: Additional processing parameters
            
        Returns:
            Processed sequence [batch, target_length, d_model]
        """
        # Apply normalization to backbone output
        norm_result = self.forward(backbone_output, fit=kwargs.get('fit', False))
        processed = norm_result['processed']
        
        # Handle target length adjustment if needed
        if processed.size(1) != target_length:
            if processed.size(1) > target_length:
                # Truncate
                processed = processed[:, :target_length, :]
            else:
                # Pad or extrapolate
                diff = target_length - processed.size(1)
                padding = processed[:, -1:, :].repeat(1, diff, 1)
                processed = torch.cat([processed, padding], dim=1)
        
        return processed

    def get_processor_type(self) -> str:
        return "normalization"