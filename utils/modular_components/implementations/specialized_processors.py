"""
Specialized Signal Processors Integration

Extracts and integrates specialized signal processing components from existing implementations.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ProcessorConfig:
    """Base configuration for processors"""
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.float32


@dataclass 
class FrequencyProcessorConfig(ProcessorConfig):
    """Configuration for frequency domain processors"""
    fft_size: Optional[int] = None
    window_size: int = 64
    hop_length: int = 32
    freq_weights: Optional[List[float]] = None


@dataclass
class StructuralProcessorConfig(ProcessorConfig):
    """Configuration for structural processors"""
    patch_size: int = 8
    stride: int = 4
    min_length: int = 16


@dataclass
class AlignmentProcessorConfig(ProcessorConfig):
    """Configuration for alignment processors"""
    window_size: int = 10
    distance_metric: str = 'euclidean'
    penalty: float = 1.0


class FrequencyDomainProcessor:
    """Extracts frequency domain processing from existing loss functions"""
    
    def __init__(self, config: FrequencyProcessorConfig):
        self.config = config
        self.device = config.device
        
    def compute_fft_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Extract FFT-based frequency loss computation"""
        try:
            # Attempt to use existing implementation
            from utils.enhanced_losses import FrequencyAwareLoss
            freq_loss = FrequencyAwareLoss()
            return freq_loss._compute_frequency_loss(pred, true)
        except ImportError:
            # Fallback implementation
            return self._fallback_fft_loss(pred, true)
    
    def _fallback_fft_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Fallback FFT loss computation"""
        # Move to correct device
        pred = pred.to(self.device)
        true = true.to(self.device)
        
        # Compute FFT
        pred_fft = torch.fft.fft(pred, dim=-1)
        true_fft = torch.fft.fft(true, dim=-1)
        
        # Compute magnitude and phase losses
        pred_mag = torch.abs(pred_fft)
        true_mag = torch.abs(true_fft)
        mag_loss = nn.functional.mse_loss(pred_mag, true_mag)
        
        pred_phase = torch.angle(pred_fft)
        true_phase = torch.angle(true_fft)
        phase_loss = nn.functional.mse_loss(pred_phase, true_phase)
        
        return mag_loss + 0.1 * phase_loss
    
    def compute_spectral_features(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract spectral feature computation"""
        x = x.to(self.device)
        
        # FFT
        x_fft = torch.fft.fft(x, dim=-1)
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        
        # Power spectral density
        psd = magnitude ** 2
        
        # Spectral centroid
        freqs = torch.fft.fftfreq(x.size(-1)).to(self.device)
        spectral_centroid = torch.sum(freqs * psd, dim=-1) / torch.sum(psd, dim=-1)
        
        return {
            'magnitude': magnitude,
            'phase': phase,
            'psd': psd,
            'spectral_centroid': spectral_centroid
        }


class StructuralPatchProcessor:
    """Extracts patch-based structural processing"""
    
    def __init__(self, config: StructuralProcessorConfig):
        self.config = config
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.device = config.device
        
    def extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Extract overlapping patches from time series"""
        try:
            # Attempt to use existing implementation
            from utils.losses import PSLoss
            ps_loss = PSLoss()
            return ps_loss._extract_patches(x, self.patch_size, self.stride)
        except (ImportError, AttributeError):
            return self._fallback_extract_patches(x)
    
    def _fallback_extract_patches(self, x: torch.Tensor) -> torch.Tensor:
        """Fallback patch extraction"""
        x = x.to(self.device)
        B, L, D = x.shape
        
        patches = []
        for i in range(0, L - self.patch_size + 1, self.stride):
            patch = x[:, i:i + self.patch_size, :]
            patches.append(patch)
        
        if patches:
            return torch.stack(patches, dim=1)  # [B, N_patches, patch_size, D]
        else:
            # Return original if too short
            return x.unsqueeze(1)
    
    def compute_patch_statistics(self, patches: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute statistical features for patches"""
        # patches: [B, N_patches, patch_size, D]
        patch_mean = torch.mean(patches, dim=2)  # [B, N_patches, D]
        patch_std = torch.std(patches, dim=2)
        patch_min = torch.min(patches, dim=2)[0]
        patch_max = torch.max(patches, dim=2)[0]
        
        return {
            'mean': patch_mean,
            'std': patch_std,
            'min': patch_min,
            'max': patch_max,
            'range': patch_max - patch_min
        }
    
    def compute_structural_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute patch-based structural loss"""
        try:
            from utils.losses import PSLoss
            ps_loss = PSLoss(patch_size=self.patch_size)
            return ps_loss(pred, true)
        except ImportError:
            return self._fallback_structural_loss(pred, true)
    
    def _fallback_structural_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Fallback structural loss computation"""
        pred_patches = self.extract_patches(pred)
        true_patches = self.extract_patches(true)
        
        # Compute statistics
        pred_stats = self.compute_patch_statistics(pred_patches)
        true_stats = self.compute_patch_statistics(true_patches)
        
        # Compare statistics
        stat_loss = 0
        for key in pred_stats:
            stat_loss += nn.functional.mse_loss(pred_stats[key], true_stats[key])
        
        return stat_loss / len(pred_stats)


class DTWAlignmentProcessor:
    """Extracts DTW-based alignment processing"""
    
    def __init__(self, config: AlignmentProcessorConfig):
        self.config = config
        self.window_size = config.window_size
        self.device = config.device
        self.penalty = config.penalty
        
    def compute_dtw_alignment(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute DTW alignment loss"""
        try:
            from utils.losses import DTWLoss
            dtw_loss = DTWLoss(window_size=self.window_size)
            return dtw_loss(pred, true)
        except ImportError:
            return self._fallback_dtw_loss(pred, true)
    
    def _fallback_dtw_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Simplified DTW-inspired alignment loss"""
        pred = pred.to(self.device)
        true = true.to(self.device)
        
        B, L, D = pred.shape
        
        # Simple alignment penalty based on temporal shifts
        alignment_loss = 0
        for shift in range(-self.window_size, self.window_size + 1):
            if shift == 0:
                continue
                
            if shift > 0:
                pred_shifted = pred[:, :-shift, :]
                true_aligned = true[:, shift:, :]
            else:
                pred_shifted = pred[:, -shift:, :]
                true_aligned = true[:, :shift, :]
            
            shift_loss = nn.functional.mse_loss(pred_shifted, true_aligned)
            alignment_loss += shift_loss * torch.exp(-torch.tensor(abs(shift) * self.penalty))
        
        # Base MSE loss
        base_loss = nn.functional.mse_loss(pred, true)
        
        return base_loss + 0.1 * alignment_loss


class TrendProcessor:
    """Extracts trend analysis components"""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.device = config.device
        
    def extract_trend_components(self, x: torch.Tensor, scales: List[int] = None) -> Dict[str, torch.Tensor]:
        """Extract multi-scale trend components"""
        if scales is None:
            scales = [1, 3, 7, 14]
        
        x = x.to(self.device)
        components = {}
        
        for scale in scales:
            if scale == 1:
                components[f'trend_{scale}'] = x
            else:
                # Simple moving average for trend extraction
                kernel = torch.ones(scale, device=self.device) / scale
                kernel = kernel.view(1, 1, -1)
                
                # Pad to maintain length
                padding = scale // 2
                x_padded = nn.functional.pad(x.transpose(1, 2), (padding, padding), mode='reflect')
                
                trend = nn.functional.conv1d(x_padded, kernel, groups=1)
                trend = trend.transpose(1, 2)
                
                # Ensure same length
                if trend.size(1) != x.size(1):
                    trend = nn.functional.interpolate(
                        trend.transpose(1, 2), size=x.size(1), mode='linear', align_corners=False
                    ).transpose(1, 2)
                
                components[f'trend_{scale}'] = trend
        
        return components
    
    def compute_trend_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Compute multi-scale trend-aware loss"""
        try:
            from utils.enhanced_losses import MultiScaleTrendAwareLoss
            trend_loss = MultiScaleTrendAwareLoss()
            return trend_loss(pred, true)
        except ImportError:
            return self._fallback_trend_loss(pred, true)
    
    def _fallback_trend_loss(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        """Fallback trend loss computation"""
        pred_trends = self.extract_trend_components(pred)
        true_trends = self.extract_trend_components(true)
        
        trend_loss = 0
        for key in pred_trends:
            scale_weight = 1.0 / (int(key.split('_')[-1]) ** 0.5)  # Weight by scale
            trend_loss += scale_weight * nn.functional.mse_loss(pred_trends[key], true_trends[key])
        
        return trend_loss / len(pred_trends)


class QuantileProcessor:
    """Extracts quantile-based processing"""
    
    def __init__(self, config: ProcessorConfig):
        self.config = config
        self.device = config.device
        
    def compute_quantile_loss(self, pred: torch.Tensor, true: torch.Tensor, 
                            quantiles: List[float] = None) -> torch.Tensor:
        """Compute quantile regression loss"""
        if quantiles is None:
            quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        
        pred = pred.to(self.device)
        true = true.to(self.device)
        
        # Assume pred contains quantile predictions
        if pred.size(-1) != len(quantiles):
            # If not quantile predictions, use standard MSE
            return nn.functional.mse_loss(pred, true)
        
        quantile_loss = 0
        for i, q in enumerate(quantiles):
            pred_q = pred[..., i]
            errors = true.squeeze(-1) - pred_q
            quantile_loss += torch.mean(torch.max(q * errors, (q - 1) * errors))
        
        return quantile_loss / len(quantiles)


class IntegratedSignalProcessor:
    """Integrated processor combining all specialized processors"""
    
    def __init__(self, 
                 freq_config: Optional[FrequencyProcessorConfig] = None,
                 struct_config: Optional[StructuralProcessorConfig] = None,
                 align_config: Optional[AlignmentProcessorConfig] = None,
                 trend_config: Optional[ProcessorConfig] = None):
        
        self.freq_processor = FrequencyDomainProcessor(freq_config or FrequencyProcessorConfig())
        self.struct_processor = StructuralPatchProcessor(struct_config or StructuralProcessorConfig())
        self.align_processor = DTWAlignmentProcessor(align_config or AlignmentProcessorConfig())
        self.trend_processor = TrendProcessor(trend_config or ProcessorConfig())
        self.quantile_processor = QuantileProcessor(trend_config or ProcessorConfig())
        
        logger.info("IntegratedSignalProcessor initialized with all specialized processors")
    
    def process_signal(self, x: torch.Tensor, mode: str = 'full') -> Dict[str, Any]:
        """Comprehensive signal processing"""
        results = {}
        
        if mode in ['full', 'frequency']:
            results['frequency'] = self.freq_processor.compute_spectral_features(x)
        
        if mode in ['full', 'structural']:
            patches = self.struct_processor.extract_patches(x)
            results['structural'] = self.struct_processor.compute_patch_statistics(patches)
        
        if mode in ['full', 'trend']:
            results['trend'] = self.trend_processor.extract_trend_components(x)
        
        return results
    
    def compute_comprehensive_loss(self, pred: torch.Tensor, true: torch.Tensor,
                                 weights: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        """Compute all specialized losses"""
        if weights is None:
            weights = {
                'frequency': 0.2,
                'structural': 0.2, 
                'alignment': 0.2,
                'trend': 0.2,
                'base': 0.2
            }
        
        losses = {}
        
        # Base MSE loss
        losses['base'] = nn.functional.mse_loss(pred, true)
        
        # Specialized losses
        if weights.get('frequency', 0) > 0:
            losses['frequency'] = self.freq_processor.compute_fft_loss(pred, true)
        
        if weights.get('structural', 0) > 0:
            losses['structural'] = self.struct_processor.compute_structural_loss(pred, true)
        
        if weights.get('alignment', 0) > 0:
            losses['alignment'] = self.align_processor.compute_dtw_alignment(pred, true)
        
        if weights.get('trend', 0) > 0:
            losses['trend'] = self.trend_processor.compute_trend_loss(pred, true)
        
        # Weighted combination
        total_loss = sum(weights.get(k, 0) * v for k, v in losses.items())
        losses['total'] = total_loss
        
        return losses
