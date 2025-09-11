"""Processor Component Registry

This module provides a unified registry for processor components that transform
data between pipeline stages.
"""

from typing import Dict, Any, Type, Optional, List
from ..core.registry import ComponentFamily, register_component, get_component, list_components
from ..base_interfaces import BaseProcessor

# Import processor implementations
try:
    from .processors import WaveletProcessor, NormalizationProcessor
    PROCESSORS_AVAILABLE = True
except ImportError as e:
    PROCESSORS_AVAILABLE = False
    WaveletProcessor = None
    NormalizationProcessor = None
    print(f"Warning: Processor components not available: {e}")


# Register processor components
if PROCESSORS_AVAILABLE:
    # Wavelet Processor
    register_component(
        ComponentFamily.PROCESSOR,
        "wavelet",
        WaveletProcessor,
        description="Wavelet decomposition processor for time series analysis",
        config_schema={
            "wavelet_type": {"type": "string", "default": "db4", "description": "Type of wavelet (db4, haar, etc.)"},
            "decomposition_levels": {"type": "integer", "default": 3, "description": "Number of decomposition levels"},
            "mode": {"type": "string", "default": "symmetric", "description": "Decomposition mode"},
            "residual_learning": {"type": "boolean", "default": True, "description": "Enable residual learning approach"},
            "baseline_method": {"type": "string", "default": "simple", "description": "Baseline forecast method (simple, arima, none)"},
            "device": {"type": "string", "default": "cpu", "description": "Device for computations"}
        },
        tags=["wavelet", "decomposition", "time-series", "signal-processing"]
    )
    
    # Normalization Processor
    register_component(
        ComponentFamily.PROCESSOR,
        "normalization",
        NormalizationProcessor,
        description="Data normalization processor with multiple strategies",
        config_schema={
            "type": {"type": "string", "default": "standard", "description": "Normalization type (standard, minmax, robust)"},
            "feature_wise": {"type": "boolean", "default": True, "description": "Apply normalization per feature"},
            "eps": {"type": "number", "default": 1e-8, "description": "Small epsilon for numerical stability"}
        },
        tags=["normalization", "preprocessing", "scaling"]
    )


# Factory functions
def create_wavelet_processor(config: Optional[Dict[str, Any]] = None) -> BaseProcessor:
    """Create a wavelet processor with the given configuration."""
    if not PROCESSORS_AVAILABLE or WaveletProcessor is None:
        raise ImportError("WaveletProcessor not available")
    
    config = config or {}
    return WaveletProcessor(config)


def create_normalization_processor(config: Optional[Dict[str, Any]] = None) -> BaseProcessor:
    """Create a normalization processor with the given configuration."""
    if not PROCESSORS_AVAILABLE or NormalizationProcessor is None:
        raise ImportError("NormalizationProcessor not available")
    
    config = config or {}
    return NormalizationProcessor(config)


# Registry query functions
def list_processor_components() -> List[str]:
    """List all registered processor components."""
    return list_components(ComponentFamily.PROCESSOR)


def get_processor_component(name: str) -> Type[BaseProcessor]:
    """Get a processor component by name."""
    return get_component(ComponentFamily.PROCESSOR, name)


def create_processor(name: str, config: Optional[Dict[str, Any]] = None) -> BaseProcessor:
    """Create a processor component by name with the given configuration."""
    component_class = get_processor_component(name)
    config = config or {}
    return component_class(config)


# Export public interface
__all__ = [
    'create_wavelet_processor',
    'create_normalization_processor',
    'list_processor_components',
    'get_processor_component',
    'create_processor'
]