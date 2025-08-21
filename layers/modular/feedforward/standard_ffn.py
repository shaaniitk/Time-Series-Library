"""Local standard position-wise feed-forward network (FFN).

Provides an internal implementation mirroring the legacy StandardFFN so that
no external import from ``utils.modular_components`` is required.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
from typing import Dict


@dataclass
class FFNConfig:
    d_model: int = 512
    d_ff: int = 2048
    dropout: float = 0.1
    activation: str = "relu"
    use_bias: bool = True
    layer_norm: bool = False


class StandardFFN(nn.Module):
    def __init__(self, config: FFNConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_ff = config.d_ff
        act = self._act(config.activation)
        self.linear1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.linear2 = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.activation = act
        self.layer_norm = nn.LayerNorm(config.d_model) if config.layer_norm else None

    def _act(self, name: str) -> nn.Module:
        return {
            'relu': nn.ReLU(),
            'gelu': nn.GELU(),
            'swish': nn.SiLU(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(),
        }.get(name.lower(), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        h = self.linear1(x)
        h = self.activation(h)
        h = self.dropout1(h)
        h = self.linear2(h)
        h = self.dropout2(h)
        if self.layer_norm is not None:
            h = self.layer_norm(h)
        return h

    # Compatibility accessors
    def get_output_dim(self) -> int:
        return self.d_model

    def get_capabilities(self) -> Dict[str, int]:
        return {"d_model": self.d_model, "d_ff": self.d_ff}

__all__ = ["StandardFFN", "FFNConfig"]
