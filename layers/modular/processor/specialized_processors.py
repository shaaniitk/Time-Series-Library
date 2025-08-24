"""
Specialized Processors for Modular Framework

This module provides specialized processor implementations migrated from utils/.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass
from ..base_interfaces import BaseProcessor

logger = logging.getLogger(__name__)

@dataclass
class ProcessorConfig:
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype: torch.dtype = torch.float32

@dataclass
class FrequencyProcessorConfig(ProcessorConfig):
    fft_size: Optional[int] = None
    window_size: int = 64
    hop_length: int = 32
    freq_weights: Optional[List[float]] = None

@dataclass
class StructuralProcessorConfig(ProcessorConfig):
    patch_size: int = 8
    stride: int = 4
    min_length: int = 16

@dataclass
class AlignmentProcessorConfig(ProcessorConfig):
    window_size: int = 10
    distance_metric: str = 'euclidean'
    penalty: float = 1.0

class FrequencyDomainProcessor(BaseProcessor):
    def __init__(self, config: FrequencyProcessorConfig):
        super().__init__(config)
        self.config = config
        self.device = config.device

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.to(self.device)
        x_fft = torch.fft.fft(x, dim=-1)
        magnitude = torch.abs(x_fft)
        phase = torch.angle(x_fft)
        psd = magnitude ** 2
        freqs = torch.fft.fftfreq(x.size(-1)).to(self.device)
        spectral_centroid = torch.sum(freqs * psd, dim=-1) / torch.sum(psd, dim=-1)
        return {
            'magnitude': magnitude,
            'phase': phase,
            'psd': psd,
            'spectral_centroid': spectral_centroid
        }

class StructuralPatchProcessor(BaseProcessor):
    def __init__(self, config: StructuralProcessorConfig):
        super().__init__(config)
        self.config = config
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.device = config.device

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = x.to(self.device)
        B, L, D = x.shape
        patches = []
        for i in range(0, L - self.patch_size + 1, self.stride):
            patch = x[:, i:i + self.patch_size, :]
            patches.append(patch)
        if patches:
            patches = torch.stack(patches, dim=1)
        else:
            patches = x.unsqueeze(1)
        patch_mean = torch.mean(patches, dim=2)
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

class DTWAlignmentProcessor(BaseProcessor):
    def __init__(self, config: AlignmentProcessorConfig):
        super().__init__(config)
        self.config = config
        self.window_size = config.window_size
        self.device = config.device
        self.penalty = config.penalty

    def forward(self, pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
        pred = pred.to(self.device)
        true = true.to(self.device)
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
        base_loss = nn.functional.mse_loss(pred, true)
        return base_loss + 0.1 * alignment_loss

class TrendProcessor(BaseProcessor):
    def __init__(self, config: ProcessorConfig):
        super().__init__(config)
        self.config = config
        self.device = config.device

    def forward(self, x: torch.Tensor, scales: List[int] = None) -> Dict[str, torch.Tensor]:
        if scales is None:
            scales = [1, 3, 7, 14]
        x = x.to(self.device)
        components = {}
        # Add trend extraction logic here
        return components