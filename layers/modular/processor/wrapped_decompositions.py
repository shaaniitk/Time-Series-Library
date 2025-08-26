"""
Wrapped decomposition processors registered under the unified registry.

These adapters expose decomposition components as 'processor' family entries
with a uniform process_sequence API, without importing from utils/*.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base_interfaces import BaseProcessor
from ..decomposition.series_decomposition import SeriesDecomposition
from ..decomposition.stable_decomposition import StableSeriesDecomposition
from ..decomposition.learnable_decomposition import LearnableSeriesDecomposition
from ..decomposition.wavelet_decomposition import WaveletHierarchicalDecomposition
from ..core.registry import register_component


@dataclass
class ProcessorConfig:
    d_model: int = 16
    dropout: float = 0.0
    seq_len: int = 8
    pred_len: int = 8
    label_len: int = 0
    custom_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionProcessorConfig(ProcessorConfig):
    component_selection: str = "seasonal"  # seasonal | trend | concat | both
    kernel_size: int = 25
    ensure_odd_kernel: bool = False
    wavelet_type: str = "db4"
    levels: int = 3
    use_learnable_weights: bool = True


class _ResizeMixin:
    @staticmethod
    def _resize_to_target_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
        if x.size(1) == target_length:
            return x
        if x.size(1) < target_length:
            return F.interpolate(x.transpose(1, 2), size=target_length, mode="linear", align_corners=False).transpose(1, 2)
        return F.adaptive_avg_pool1d(x.transpose(1, 2), target_length).transpose(1, 2)


class SeriesDecompositionProcessor(BaseProcessor, _ResizeMixin):
    def __init__(self, config: DecompositionProcessorConfig):
        super().__init__(config)
        self.config = config
        k = config.kernel_size
        if self.config.ensure_odd_kernel:
            k = k + (1 - k % 2)
        self.decomp = SeriesDecomposition(kernel_size=k)
        self.concat_proj = nn.Linear(config.d_model * 2, config.d_model) if config.component_selection == "concat" else None

    def forward(self, embedded_input: torch.Tensor, backbone_output: Optional[torch.Tensor] = None, target_length: Optional[int] = None, **kwargs) -> torch.Tensor:
        return self.process_sequence(embedded_input, backbone_output, target_length or embedded_input.size(1), **kwargs)

    def process_sequence(self, embedded_input: torch.Tensor, backbone_output: Optional[torch.Tensor], target_length: int, **kwargs) -> torch.Tensor:
        seasonal, trend = self.decomp(embedded_input)
        seasonal = self._resize_to_target_length(seasonal, target_length)
        trend = self._resize_to_target_length(trend, target_length)
        sel = self.config.component_selection
        if sel == "seasonal":
            return seasonal
        if sel == "trend":
            return trend
        if sel == "concat":
            x = torch.cat([seasonal, trend], dim=-1)
            return self.concat_proj(x) if self.concat_proj is not None else x
        return seasonal

    def get_processor_type(self) -> str:
        return "decomposition_series"

    def get_output_dim(self) -> int:
        return self.config.d_model


class StableSeriesDecompositionProcessor(SeriesDecompositionProcessor):
    def __init__(self, config: DecompositionProcessorConfig):
        config.ensure_odd_kernel = True
        super().__init__(config)

    def get_processor_type(self) -> str:
        return "decomposition_series_stable"


class LearnableDecompositionProcessor(BaseProcessor, _ResizeMixin):
    def __init__(self, config: DecompositionProcessorConfig):
        super().__init__(config)
        self.config = config
        self.decomp = LearnableSeriesDecomposition(input_dim=config.d_model, init_kernel_size=config.kernel_size, max_kernel_size=max(config.kernel_size, 50))
        self.concat_proj = nn.Linear(config.d_model * 2, config.d_model) if config.component_selection == "concat" else None

    def forward(self, embedded_input: torch.Tensor, backbone_output: Optional[torch.Tensor] = None, target_length: Optional[int] = None, **kwargs) -> torch.Tensor:
        return self.process_sequence(embedded_input, backbone_output, target_length or embedded_input.size(1), **kwargs)

    def process_sequence(self, embedded_input: torch.Tensor, backbone_output: Optional[torch.Tensor], target_length: int, **kwargs) -> torch.Tensor:
        seasonal, trend = self.decomp(embedded_input)
        seasonal = self._resize_to_target_length(seasonal, target_length)
        trend = self._resize_to_target_length(trend, target_length)
        sel = self.config.component_selection
        if sel == "seasonal":
            return seasonal
        if sel == "trend":
            return trend
        if sel == "concat":
            x = torch.cat([seasonal, trend], dim=-1)
            return self.concat_proj(x) if self.concat_proj is not None else x
        return seasonal

    def get_processor_type(self) -> str:
        return "decomposition_learnable"

    def get_output_dim(self) -> int:
        return self.config.d_model


class WaveletDecompositionProcessor(BaseProcessor, _ResizeMixin):
    def __init__(self, config: DecompositionProcessorConfig):
        super().__init__(config)
        self.config = config
        self.decomp = WaveletHierarchicalDecomposition(
            seq_len=config.seq_len,
            d_model=config.d_model,
            wavelet_type=config.wavelet_type,
            levels=config.levels,
            use_learnable_weights=config.use_learnable_weights,
        )
        self.concat_proj = nn.Linear(config.d_model * 2, config.d_model) if config.component_selection == "concat" else None

    def forward(self, embedded_input: torch.Tensor, backbone_output: Optional[torch.Tensor] = None, target_length: Optional[int] = None, **kwargs) -> torch.Tensor:
        return self.process_sequence(embedded_input, backbone_output, target_length or embedded_input.size(1), **kwargs)

    def process_sequence(self, embedded_input: torch.Tensor, backbone_output: Optional[torch.Tensor], target_length: int, **kwargs) -> torch.Tensor:
        seasonal, trend = self.decomp(embedded_input)
        seasonal = self._resize_to_target_length(seasonal, target_length)
        trend = self._resize_to_target_length(trend, target_length)
        sel = self.config.component_selection
        if sel == "seasonal":
            return seasonal
        if sel == "trend":
            return trend
        if sel == "concat":
            x = torch.cat([seasonal, trend], dim=-1)
            return self.concat_proj(x) if self.concat_proj is not None else x
        return seasonal

    def get_processor_type(self) -> str:
        return "decomposition_wavelet"

    def get_output_dim(self) -> int:
        return self.config.d_model


def register_layers_decompositions() -> None:
    register_component(
        "processor",
        "series_decomposition_processor",
        SeriesDecompositionProcessor,
        metadata={
            "source": "layers.modular.decomposition.series_decomposition",
            "features": ["moving_average", "trend", "seasonal"],
            "domain": "decomposition",
            "sophistication_level": "medium",
        },
    )
    register_component(
        "processor",
        "stable_series_decomposition_processor",
        StableSeriesDecompositionProcessor,
        metadata={
            "source": "layers.modular.decomposition.stable_decomposition",
            "features": ["moving_average", "stability_fix"],
            "domain": "decomposition",
            "sophistication_level": "medium",
            "deprecated": True,
            "note": "Use series_decomposition_processor with ensure_odd_kernel=True",
        },
    )
    register_component(
        "processor",
        "learnable_series_decomposition_processor",
        LearnableDecompositionProcessor,
        metadata={
            "source": "layers.modular.decomposition.learnable_decomposition",
            "features": ["learnable_weights", "adaptive_kernel", "feature_specific"],
            "domain": "decomposition",
            "sophistication_level": "high",
        },
    )
    register_component(
        "processor",
        "wavelet_hierarchical_decomposition_processor",
        WaveletDecompositionProcessor,
        metadata={
            "source": "layers.modular.decomposition.wavelet_decomposition",
            "features": ["dwt", "multi_resolution", "wavelet"],
            "domain": "decomposition",
            "sophistication_level": "high",
        },
    )
