"""
Adapters that wrap legacy decomposition components under layers/modular/decomposition
into utils 'processor' components, preserving behavior while providing a unified API.

Each wrapper implements BaseProcessor.process_sequence and registers into the
ComponentRegistry via register_layers_decompositions().
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modular_components.base_interfaces import BaseProcessor
from ...modular_components.config_schemas import ProcessorConfig
from ...modular_components.registry import register_component

# Legacy implementations
from layers.modular.decomposition.series_decomposition import SeriesDecomposition
from layers.modular.decomposition.stable_decomposition import StableSeriesDecomposition
from layers.modular.decomposition.learnable_decomposition import LearnableSeriesDecomposition
from layers.modular.decomposition.wavelet_decomposition import WaveletHierarchicalDecomposition


@dataclass
class DecompositionProcessorConfig(ProcessorConfig):
    """Configuration for decomposition processors wrapped as processors.
    
    Attributes:
        component_selection: which part to output ('seasonal', 'trend', 'concat', 'both').
        kernel_size: for moving-average based decompositions.
        ensure_odd_kernel: enforce odd kernel for stability (stable variant).
        wavelet_type: wavelet family (for wavelet decomposition).
        levels: DWT levels (for wavelet decomposition).
        use_learnable_weights: learnable scale weights (for wavelet decomposition).
    """
    component_selection: str = "seasonal"  # seasonal | trend | concat | both
    kernel_size: int = 25
    ensure_odd_kernel: bool = False
    wavelet_type: str = "db4"
    levels: int = 3
    use_learnable_weights: bool = True


class _ResizeMixin:
    @staticmethod
    def _resize_to_target_length(x: torch.Tensor, target_length: int) -> torch.Tensor:
        """Resize [B, L, D] to desired L while preserving features."""
        if x.size(1) == target_length:
            return x
        if x.size(1) < target_length:
            return F.interpolate(x.transpose(1, 2), size=target_length, mode="linear", align_corners=False).transpose(1, 2)
        return F.adaptive_avg_pool1d(x.transpose(1, 2), target_length).transpose(1, 2)


class SeriesDecompositionProcessor(BaseProcessor, _ResizeMixin):
    """Processor wrapper around SeriesDecomposition."""

    def __init__(self, config: DecompositionProcessorConfig):
        super().__init__(config)
        self.config = config
        k = config.kernel_size
        if self.config.ensure_odd_kernel:
            k = k + (1 - k % 2)
        self.decomp = SeriesDecomposition(kernel_size=k)
        # Optional projection for concat
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
        # both -> default to seasonal (could return tuple, but processors should return Tensor)
        return seasonal

    def get_processor_type(self) -> str:
        return "decomposition_series"

    def get_output_dim(self) -> int:
        return self.config.d_model


class StableSeriesDecompositionProcessor(SeriesDecompositionProcessor):
    """Stable variant that enforces odd kernel; kept for backward compatibility."""

    def __init__(self, config: DecompositionProcessorConfig):
        config.ensure_odd_kernel = True
        super().__init__(config)

    def get_processor_type(self) -> str:
        return "decomposition_series_stable"


class LearnableDecompositionProcessor(BaseProcessor, _ResizeMixin):
    """Processor wrapper around LearnableSeriesDecomposition."""

    def __init__(self, config: DecompositionProcessorConfig):
        super().__init__(config)
        self.config = config
        # Align input_dim with d_model by default
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
    """Processor wrapper around WaveletHierarchicalDecomposition."""

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
    """Register wrapped decomposition processors into the utils registry."""
    # Series decomposition
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
    # Stable variant (kept for backward-compat; essentially same with enforced odd kernel)
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
    # Learnable decomposition
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
    # Wavelet hierarchical decomposition
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
