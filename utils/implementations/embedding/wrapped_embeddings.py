"""
Embedding registrations that expose utils embedding implementations to the unified registry.
"""
from __future__ import annotations

from layers.modular.core.registry import register_component
from layers.modular.embeddings import (
    TemporalEmbedding,
    ValueEmbedding,
    CovariateEmbedding,
    HybridEmbedding,
)


def register_utils_embeddings() -> None:
    """Register embedding implementations into the global ComponentRegistry."""
    register_component(
        "embedding",
        "temporal_embedding",
        TemporalEmbedding,
        metadata={
            "domain": "embedding",
            "features": ["positional", "temporal_features"],
            "source": "layers.modular.embeddings",
        },
    )
    register_component(
        "embedding",
        "value_embedding",
        ValueEmbedding,
        metadata={
            "domain": "embedding",
            "features": ["continuous_projection", "optional_binning"],
            "source": "layers.modular.embeddings",
        },
    )
    register_component(
        "embedding",
        "covariate_embedding",
        CovariateEmbedding,
        metadata={
            "domain": "embedding",
            "features": ["categorical", "numerical"],
            "source": "layers.modular.embeddings",
        },
    )
    register_component(
        "embedding",
        "hybrid_embedding",
        HybridEmbedding,
        metadata={
            "domain": "embedding",
            "features": ["combine_temporal_value_covariate"],
            "source": "layers.modular.embeddings",
        },
    )
