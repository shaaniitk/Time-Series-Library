"""Processor family wrappers for unified registry.

Processors transform backbone outputs; thin wrappers adapt simple kwargs.
"""
from __future__ import annotations
from typing import Any

from .time_domain_processor import TimeDomainProcessor as _TimeProcLegacy  # local implementation

class TimeDomainProcessorWrapper:
    def __init__(self, d_model: int = 512, pred_len: int = 24, **kwargs: Any):
        if _TimeProcLegacy is None:  # pragma: no cover
            raise RuntimeError("TimeDomainProcessor implementation unavailable")
        cfg = type('Cfg', (), {'d_model': d_model, 'pred_len': pred_len, **kwargs})
        self._impl = _TimeProcLegacy(cfg)  # type: ignore[arg-type]
        self.d_model = d_model

    def forward(self, embedded_input, backbone_output, target_length: int, **kwargs):  # type: ignore
        return self._impl.forward(embedded_input, backbone_output, target_length, **kwargs)

__all__ = ['TimeDomainProcessorWrapper']
