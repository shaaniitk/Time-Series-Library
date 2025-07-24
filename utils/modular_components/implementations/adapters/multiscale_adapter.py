"""
MultiScaleAdapter: Adapter that applies multiple scales of processing to the same backbone.
"""
import logging
import torch
import torch.nn as nn
from typing import Dict, Any
from .base import BaseAdapter

logger = logging.getLogger(__name__)

class MultiScaleAdapter(BaseAdapter):
    """
    Adapter that applies multiple scales of processing to the same backbone.
    Useful for hierarchical decomposition approaches where we want to
    process different frequency components separately.
    """
    def __init__(self, backbone: BaseAdapter, scale_config: Dict[str, Any]):
        super().__init__(backbone.config)
        self.backbone = backbone
        self.scales = scale_config.get('scales', [1, 2, 4])
        self.aggregation_method = scale_config.get('aggregation', 'concat')
        self.scale_processors = nn.ModuleList([
            nn.Linear(self.backbone.get_d_model(), self.backbone.get_d_model())
            for _ in self.scales
        ])
        if self.aggregation_method == 'concat':
            self.aggregator = nn.Linear(
                len(self.scales) * self.backbone.get_d_model(),
                self.backbone.get_d_model()
            )
        elif self.aggregation_method == 'add':
            self.aggregator = nn.Identity()
        logger.info(f"MultiScaleAdapter initialized with scales: {self.scales}")

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        scale_outputs = []
        for i, scale in enumerate(self.scales):
            if scale > 1:
                x_scaled = x[:, ::scale, :]
            else:
                x_scaled = x
            scale_output = self.backbone.forward(x_scaled, **kwargs)
            scale_output = self.scale_processors[i](scale_output)
            if scale > 1 and scale_output.size(1) != x.size(1):
                scale_output = torch.nn.functional.interpolate(
                    scale_output.transpose(1, 2),
                    size=x.size(1),
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            scale_outputs.append(scale_output)
        if self.aggregation_method == 'concat':
            combined = torch.cat(scale_outputs, dim=-1)
            output = self.aggregator(combined)
        elif self.aggregation_method == 'add':
            output = torch.stack(scale_outputs).sum(dim=0)
        return output

    def get_d_model(self) -> int:
        return self.backbone.get_d_model()

    def supports_seq2seq(self) -> bool:
        return self.backbone.supports_seq2seq()

    def get_backbone_type(self) -> str:
        return f"multiscale_{self.backbone.get_backbone_type()}"
