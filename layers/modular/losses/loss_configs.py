"""
Configuration dataclasses for all modular loss functions.
"""
from dataclasses import dataclass
from typing import Optional, Any, Dict, Union, List, Tuple

@dataclass
class LossConfig:
    """Base configuration for loss functions"""
    reduction: str = 'mean'
    weight: Optional[Any] = None
    ignore_index: int = -100

@dataclass
class BayesianLossConfig(LossConfig):
    kl_weight: float = 1e-5
    uncertainty_weight: float = 0.1
    base_loss_type: str = 'mse'
    calibration_weight: float = 0.0

@dataclass
class AdaptiveLossConfig(LossConfig):
    base_loss: str = 'mse'
    moving_avg: int = 25
    initial_trend_weight: float = 1.0
    initial_seasonal_weight: float = 1.0
    adaptive_weights: bool = True
    mse_weight: float = 1.0
    mae_weight: float = 0.5
    autocorr_weight: float = 0.3
    frequency_weight: float = 0.2
    trend_weight: float = 0.1

@dataclass
class FrequencyLossConfig(LossConfig):
    freq_weight: float = 0.1
    base_loss: str = 'mse'
    time_weight: float = 1.0
    high_freq_penalty: float = 0.1

@dataclass
class StructuralLossConfig(LossConfig):
    pred_len: int = 96
    mse_weight: float = 0.5
    w_corr: float = 1.0
    w_var: float = 1.0
    w_mean: float = 1.0
    k_dominant_freqs: int = 3
    min_patch_len: int = 5
    trend_weight: float = 1.0
    seasonal_weight: float = 1.0
    residual_weight: float = 1.0
    patch_size: int = 16
    stride: int = 8
    structural_weight: float = 1.0
    patch_weight: float = 1.0

@dataclass
class QuantileConfig(LossConfig):
    quantiles: Optional[List[float]] = None
    kl_weight: float = 1e-5
    uncertainty_weight: float = 0.1

@dataclass
class FocalLossConfig(LossConfig):
    alpha: float = 1.0
    gamma: float = 2.0
    num_classes: Optional[int] = None
    smooth: float = 1e-8

@dataclass
class DTWConfig(LossConfig):
    gamma: float = 1.0
    normalize: bool = True

@dataclass
class CustomLossConfig(LossConfig):
    custom_param: float = 0.5
