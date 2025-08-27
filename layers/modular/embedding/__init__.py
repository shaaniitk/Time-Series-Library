"""Embedding family wrappers for unified registry.

Wrap legacy embedding implementations with simple direct kwargs.
Expose concrete embedding classes and a small shim used by tests to register
them into the global registry without importing tooling directly.
"""
from __future__ import annotations
from typing import Any

try:
    from .temporal_embedding import TemporalEmbedding as _TemporalLegacy  # local implementation
except Exception:  # pragma: no cover
    _TemporalLegacy = None  # type: ignore
from .temporal_embedding import TemporalEmbedding
from .value_embedding import ValueEmbedding
from .covariate_embedding import CovariateEmbedding
from .hybrid_embedding import HybridEmbedding

class TemporalEmbeddingWrapper:
    """Adapter for temporal (or fallback standard) embedding implementations.

    Supplies expected config attributes (d_model, dropout, max_len, temp_feature_dim).
    """
    def __init__(
        self,
        d_model: int = 512,
        dropout: float = 0.1,
        max_len: int = 5000,
        temp_feature_dim: int = 4,
        **kwargs: Any,
    ) -> None:
        if _TemporalLegacy is None:  # pragma: no cover
            raise RuntimeError("TemporalEmbedding implementation unavailable")
        cfg = type('Cfg', (), {
            'd_model': d_model,
            'dropout': dropout,
            'max_len': max_len,
            'temp_feature_dim': temp_feature_dim,
            **kwargs,
        })
        self._impl = _TemporalLegacy(cfg)  # type: ignore[arg-type]
        self.d_model = d_model

    def forward(self, x, *args, **kwargs):  # type: ignore
        # Legacy TemporalEmbedding signature expects (input_embeddings, temporal_features=None, positions=None)
        return self._impl.forward(x, *args, **kwargs)

    def register_utils_embeddings():  # pragma: no cover - keep for compatibility; no-op now
        return None

__all__ = [
    'TemporalEmbeddingWrapper',
    'TemporalEmbedding', 'ValueEmbedding', 'CovariateEmbedding', 'HybridEmbedding',
    'register_utils_embeddings',
]
