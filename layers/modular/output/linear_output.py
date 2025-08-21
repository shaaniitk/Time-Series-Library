"""Local LinearOutput implementation.

Replaces previous dependency on example_components for a simple projection head.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class LinearOutputConfig:
    d_model: int = 512
    output_dim: int = 1


class LinearOutput(nn.Module):
    def __init__(self, config: LinearOutputConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.output_dim = config.output_dim
        self.projection = nn.Linear(self.d_model, self.output_dim)

    def forward(self, x: torch.Tensor):  # type: ignore[override]
        return self.projection(x)

    def get_output_dim(self) -> int:
        return self.output_dim

__all__ = ["LinearOutput", "LinearOutputConfig"]
