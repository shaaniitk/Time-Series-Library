"""
Adapter registrations that expose utils adapter implementations to the unified registry.
"""
from __future__ import annotations

from utils.modular_components.registry import register_component
from utils.modular_components.implementations.adapters import (
    CovariateAdapter,
    MultiScaleAdapter,
)


def register_utils_adapters() -> None:
    """Register Adapter components into the global ComponentRegistry."""
    register_component(
        "adapter",
        "covariate_adapter",
        CovariateAdapter,
        metadata={
            "domain": "adapter",
            "features": ["covariates", "fusion:project|concat|add", "tokenizer_aware"],
            "source": "utils.modular_components.implementations.adapters",
        },
    )
    register_component(
        "adapter",
        "multiscale_adapter",
        MultiScaleAdapter,
        metadata={
            "domain": "adapter",
            "features": ["multi_scale", "concat|add_aggregation"],
            "source": "utils.modular_components.implementations.adapters",
        },
    )
