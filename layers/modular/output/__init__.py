"""Generic output projection wrappers for unified registry (beyond output_head).
"""
from __future__ import annotations
from typing import Any

from .linear_output import LinearOutput as _LinearLegacy  # local implementation

class LinearOutputWrapper:
    def __init__(self, d_model: int = 512, output_dim: int = 1, **kwargs: Any):
        if _LinearLegacy is None:  # pragma: no cover
            raise RuntimeError("LinearOutput implementation unavailable")
        cfg = type('Cfg', (), {'d_model': d_model, 'output_dim': output_dim, **kwargs})
        self._impl = _LinearLegacy(cfg)  # type: ignore[arg-type]
        self.d_model = d_model
        self.c_out = output_dim

    def forward(self, x):  # type: ignore
        return self._impl.forward(x)

__all__ = ['LinearOutputWrapper']
