"""
Embedding registrations that expose utils embedding implementations to the unified registry.
"""
from __future__ import annotations

from utils.modular_components.registry import register_component
from utils.modular_components.implementations.embeddings import (
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
            "source": "utils.modular_components.implementations.embeddings",
        },
    )
    register_component(
        "embedding",
        "value_embedding",
        ValueEmbedding,
        metadata={
            "domain": "embedding",
            "features": ["continuous_projection", "optional_binning"],
            "source": "utils.modular_components.implementations.embeddings",
        },
    )
    register_component(
        "embedding",
        "covariate_embedding",
        CovariateEmbedding,
        metadata={
            "domain": "embedding",
            "features": ["categorical", "numerical"],
            "source": "utils.modular_components.implementations.embeddings",
        },
    )
    register_component(
        "embedding",
        "hybrid_embedding",
        HybridEmbedding,
        metadata={
            "domain": "embedding",
            "features": ["combine_temporal_value_covariate"],
            "source": "utils.modular_components.implementations.embeddings",
        },
    )
