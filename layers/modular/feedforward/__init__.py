"""Feedforward module wrappers for unified registry.

Provides thin adaptation over legacy feedforward blocks.
"""
from __future__ import annotations
from typing import Any

from .standard_ffn import StandardFFN as _FFLegacy  # local implementation

class PositionwiseFeedForwardWrapper:
    """Adapter around StandardFFN (position-wise feedforward) legacy implementation."""
    def __init__(
        self,
        d_model: int = 512,
        d_ff: int = 2048,
        dropout: float = 0.1,
        activation: str = 'relu',
        use_bias: bool = True,
        layer_norm: bool = False,
        **kwargs: Any,
    ) -> None:
        if _FFLegacy is None:  # pragma: no cover
            raise RuntimeError("Feedforward implementation unavailable")
        cfg = type('Cfg', (), {
            'd_model': d_model,
            'd_ff': d_ff,
            'dropout': dropout,
            'activation': activation,
            'use_bias': use_bias,
            'layer_norm': layer_norm,
            **kwargs,
        })
        self._impl = _FFLegacy(cfg)  # type: ignore[arg-type]
        self.d_model = d_model

    def forward(self, x):  # type: ignore
        return self._impl.forward(x)

__all__ = ['PositionwiseFeedForwardWrapper']
