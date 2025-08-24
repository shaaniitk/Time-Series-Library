"""
Adapter registrations that expose utils adapter implementations to the unified registry.
"""
from __future__ import annotations

from layers.modular.core.registry import register_component
from layers.modular.backbone.adapters import (
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
            "source": "layers.modular.backbone.adapters",
        },
    )
    register_component(
        "adapter",
        "multiscale_adapter",
        MultiScaleAdapter,
        metadata={
            "domain": "adapter",
            "features": ["multi_scale", "concat|add_aggregation"],
            "source": "layers.modular.backbone.adapters",
        },
    )
