
from .base import BaseDecomposition
from .series_decomposition import SeriesDecomposition
from .stable_decomposition import StableSeriesDecomposition
from .learnable_decomposition import LearnableSeriesDecomposition
from .wavelet_decomposition import WaveletHierarchicalDecomposition
from .registry import DecompositionRegistry, get_decomposition_component

# Compatibility re-exports for processor tests now point to local wrappers
try:  # Prefer utils wrappers for processor-style registration
    from layers.modular.processor.wrapped_decompositions import (  # type: ignore
        register_layers_decompositions,
        DecompositionProcessorConfig,
    )
except Exception:  # pragma: no cover
    register_layers_decompositions = None  # type: ignore
    DecompositionProcessorConfig = None  # type: ignore

__all__ = [
    "BaseDecomposition",
    "SeriesDecomposition",
    "StableSeriesDecomposition",
    "LearnableSeriesDecomposition",
    "WaveletHierarchicalDecomposition",
    "DecompositionRegistry",
    "get_decomposition_component",
    # Re-exported for tests expecting these under layers.modular.decomposition
    "register_layers_decompositions",
    "DecompositionProcessorConfig",
]
