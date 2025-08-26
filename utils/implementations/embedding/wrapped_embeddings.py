"""
Embedding registrations that expose embedding implementations to both registries.

This helper registers embeddings into:
- utils.modular_components.registry (legacy/utils global registry used by tests)
- layers.modular.core.unified_registry (unified core registry), when available
"""
from __future__ import annotations

from layers.modular.embedding.temporal_embedding import TemporalEmbedding
from layers.modular.embedding.value_embedding import ValueEmbedding
from layers.modular.embedding.covariate_embedding import CovariateEmbedding
from layers.modular.embedding.hybrid_embedding import HybridEmbedding


def _register_both(component_type: str, name: str, cls, metadata: dict | None = None) -> None:
    md = metadata or {}
    # 1) Register into utils.modular_components registry (the one tests query)
    try:
        from ...modular_components.registry import register_component as _utils_register  # type: ignore
        _utils_register(component_type, name, cls, md)
    except Exception:
        pass

    # 2) Also try to register into unified core registry (best-effort)
    try:
        from layers.modular.core.registry import unified_registry, ComponentFamily  # type: ignore
        # Map utils component_type string to ComponentFamily
        fam = getattr(ComponentFamily, component_type.upper(), None)
        if fam is not None:
            unified_registry.register(fam, name, cls, metadata=md)
    except Exception:
        pass


def register_utils_embeddings() -> None:
    """Register embedding implementations into the global registries."""
    _register_both(
        "embedding",
        "temporal_embedding",
        TemporalEmbedding,
        metadata={
            "domain": "embedding",
            "features": ["positional", "temporal_features"],
            "source": "layers.modular.embedding",
        },
    )
    _register_both(
        "embedding",
        "value_embedding",
        ValueEmbedding,
        metadata={
            "domain": "embedding",
            "features": ["continuous_projection", "optional_binning"],
            "source": "layers.modular.embedding",
        },
    )
    _register_both(
        "embedding",
        "covariate_embedding",
        CovariateEmbedding,
        metadata={
            "domain": "embedding",
            "features": ["categorical", "numerical"],
            "source": "layers.modular.embedding",
        },
    )
    _register_both(
        "embedding",
        "hybrid_embedding",
        HybridEmbedding,
        metadata={
            "domain": "embedding",
            "features": ["combine_temporal_value_covariate"],
            "source": "layers.modular.embedding",
        },
    )
