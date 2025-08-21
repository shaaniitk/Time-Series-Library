"""Local TimeDomainProcessor implementation.

Simplified processing strategy replicating behaviour of the earlier example
component while removing dependency on ``utils.modular_components``.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Optional


@dataclass
class TimeDomainProcessorConfig:
    d_model: int = 512
    pred_len: int = 24


class TimeDomainProcessor(nn.Module):
    def __init__(self, config: TimeDomainProcessorConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.pred_len = config.pred_len
        self.projection = nn.Linear(self.d_model, self.d_model)

    def forward(self, embedded_input: torch.Tensor, backbone_output: torch.Tensor, target_length: Optional[int] = None, **kwargs):  # type: ignore[override]
        target = target_length or self.pred_len
        if backbone_output.size(1) != target:
            pooled = backbone_output.mean(dim=1, keepdim=True)
            out = pooled.repeat(1, target, 1)
        else:
            out = backbone_output
        return self.projection(out)

    def process_sequence(self, embedded_input: torch.Tensor, backbone_output: torch.Tensor, target_length: int, **kwargs):  # helper alias
        return self.forward(embedded_input, backbone_output, target_length, **kwargs)

    def get_output_dim(self) -> int:
        return self.d_model

__all__ = ["TimeDomainProcessor", "TimeDomainProcessorConfig"]
