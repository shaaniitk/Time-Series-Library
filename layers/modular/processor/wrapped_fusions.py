"""Wrapped fusion processors for unified registry (no utils imports)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any

import torch

from ..base_interfaces import BaseProcessor
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
class FusionProcessorConfig(ProcessorConfig):
    n_levels: int = 3
    fusion_strategy: str = 'weighted_concat'


class HierarchicalFusionProcessor(BaseProcessor):
    def __init__(self, config: FusionProcessorConfig):
        super().__init__(config)
        from ..fusion.hierarchical_fusion import HierarchicalFusion
        self.fusion = HierarchicalFusion(d_model=config.d_model, n_levels=config.n_levels, fusion_strategy=config.fusion_strategy)
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
    register_component("processor", "fusion_hierarchical_processor", HierarchicalFusionProcessor, metadata={
        "source": "layers.modular.fusion.hierarchical_fusion",
        "features": ["multi_resolution", "weighted_concat", "weighted_sum", "attention_fusion"],
        "domain": "fusion",
        "sophistication_level": "high",
    })
