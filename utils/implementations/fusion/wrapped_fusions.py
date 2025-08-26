"""
Fusion wrappers that adapt legacy layers.modular.fusion components into the utils registry
as 'processor' components with a uniform process_sequence API.

This file also registers the wrapped fusions into the global ComponentRegistry.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List

import torch
import torch.nn as nn

from layers.modular.base_interfaces import BaseProcessor
from layers.modular.core.registry import register_component

@dataclass
class ProcessorConfig:
    """Minimal processor config base to avoid legacy modular_components import."""
    d_model: int = 16
    dropout: float = 0.0
    seq_len: int = 8
    pred_len: int = 8
    label_len: int = 0
    custom_params: dict = field(default_factory=dict)

@dataclass
class FusionProcessorConfig(ProcessorConfig):
    """Configuration for wrapped fusion processors.
    
    Attributes:
        n_levels: number of multi-resolution levels.
        fusion_strategy: fusion method (weighted_concat, weighted_sum, attention_fusion).
    """
    n_levels: int = 3
    fusion_strategy: str = 'weighted_concat'

class HierarchicalFusionProcessor(BaseProcessor):
    """Wraps HierarchicalFusion as a processor component."""
    def __init__(self, config: FusionProcessorConfig):
        super().__init__(config)
        from layers.modular.fusion.hierarchical_fusion import HierarchicalFusion
        self.fusion = HierarchicalFusion(
            d_model=config.d_model,
            n_levels=config.n_levels,
            fusion_strategy=config.fusion_strategy
        )
        self._d_model = config.d_model

    def process_sequence(self, multi_res_features: List[torch.Tensor], backbone_output: Optional[torch.Tensor], target_length: int, **kwargs) -> torch.Tensor:
        return self.fusion(multi_res_features, target_length=target_length)

    def forward(self, multi_res_features: List[torch.Tensor], backbone_output: Optional[torch.Tensor] = None, target_length: Optional[int] = None, **kwargs) -> torch.Tensor:
        return self.process_sequence(multi_res_features, backbone_output, target_length or multi_res_features[0].size(1), **kwargs)

    def get_processor_type(self) -> str:
        return "fusion_hierarchical"

    def get_output_dim(self) -> int:
        return self._d_model


def register_layers_fusions() -> None:
    """Register wrapped fusion processors into the utils registry."""
    register_component(
        "processor",
        "fusion_hierarchical_processor",
        HierarchicalFusionProcessor,
        metadata={
            "source": "layers.modular.fusion.hierarchical_fusion",
            "features": ["multi_resolution", "weighted_concat", "weighted_sum", "attention_fusion"],
            "domain": "fusion",
            "sophistication_level": "high",
        },
    )
