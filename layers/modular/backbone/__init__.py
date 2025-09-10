"""Backbone family implementations (Chronos / ChronosX etc.)

Thin wrappers over backbone implementations so they can be
instantiated through the unified registry with direct kwargs instead of a
config object. These are deliberately minimal – they adapt the __init__
signature to (d_model, **extras) and internally construct the config
object if needed later.
"""
from __future__ import annotations
from typing import Any

from .chronos_backbone import ChronosBackbone as _ChronosLegacy  # local implementation

class ChronosBackboneWrapper:
    """Thin adapter for Chronos backbone (or fallback) usable via unified registry.

    Automatically supplies minimal config attributes expected by the legacy
    implementation. Optional arguments are passed through. If the advanced
    Chronos implementation (transformers dependency) isn't available, it
    silently falls back to the lightweight example implementation.

    Parameters
    ----------
    d_model: int (default 512)
        Target internal model dimension.
    dropout: float (default 0.1)
        Dropout probability.
    model_name: str (default 'amazon/chronos-t5-small')
        HuggingFace model identifier (if transformers Chronos variant present).
    pretrained: bool (default True)
        Whether to load pretrained weights when available.
    """
    def __init__(
        self,
        d_model: int = 512,
        dropout: float = 0.1,
    n_heads: int = 8,
    num_layers: int = 3,
    d_ff: int = 2048,
        model_name: str | None = None,
        pretrained: bool = True,
        seq_len: int = 24,
        **kwargs: Any,
    ) -> None:
        if _ChronosLegacy is None:  # pragma: no cover
            raise RuntimeError("Chronos backbone implementation unavailable")
        cfg_dict = {
            'd_model': d_model,
            'dropout': dropout,
            'n_heads': n_heads,
            'num_layers': num_layers,
            'd_ff': d_ff,
            'model_name': model_name or 'amazon/chronos-t5-small',
            'pretrained': pretrained,
            'seq_len': seq_len,
        }
        cfg_dict.update(kwargs)
        impl = None
        try:
            impl = _ChronosLegacy(type('Cfg', (), cfg_dict))  # type: ignore[arg-type]
        except Exception as e:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed to instantiate Chronos backbone: {e}")
        self._impl = impl
        self.d_model = d_model

    def forward(self, x, *a, **k):  # type: ignore
        return self._impl.forward(x, *a, **k)

    def get_output_dim(self) -> int:  # compatibility hook
        return getattr(self._impl, 'd_model', self.d_model)

from .registry import BackboneRegistry, get_backbone_component

__all__ = [
    'ChronosBackboneWrapper',
    'BackboneRegistry',
    'get_backbone_component'
]
