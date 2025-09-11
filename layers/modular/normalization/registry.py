"""Registry for normalization components.

This module provides a centralized registry for all normalization-related components
including layer normalization, data normalization, and preprocessing normalization.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union
import logging

from ..core.registry import ComponentRegistry, ComponentFamily
from ..base_interfaces import BaseComponent

# Import normalization components
try:
    from ...Normalization import RMSNorm, get_norm_layer
except ImportError:
    RMSNorm = None
    get_norm_layer = None
    logging.warning("Could not import RMSNorm from layers.Normalization")

try:
    from ....utils.normalization import TSNormalizer
except ImportError:
    TSNormalizer = None
    logging.warning("Could not import TSNormalizer from utils.normalization")

try:
    from ..processor.processors import NormalizationProcessor
except ImportError:
    NormalizationProcessor = None
    logging.warning("Could not import NormalizationProcessor")

logger = logging.getLogger(__name__)

# Create normalization registry
normalization_registry = ComponentRegistry()


# ============================================================================
# Layer Normalization Components
# ============================================================================

class LayerNormWrapper(BaseComponent):
    """Wrapper for PyTorch LayerNorm to fit component interface."""
    
    def __init__(self, config):
        super().__init__(config)
        self.d_model = getattr(config, 'd_model', 512)
        self.eps = getattr(config, 'eps', 1e-5)
        self.layer_norm = nn.LayerNorm(self.d_model, eps=self.eps)
    
    def forward(self, x):
        return self.layer_norm(x)


class RMSNormWrapper(BaseComponent):
    """Wrapper for RMSNorm to fit component interface."""
    
    def __init__(self, config):
        super().__init__(config)
        self.d_model = getattr(config, 'd_model', 512)
        self.eps = getattr(config, 'eps', 1e-5)
        
        if RMSNorm is not None:
            self.rms_norm = RMSNorm(self.d_model, self.eps)
        else:
            raise ImportError("RMSNorm not available")
    
    def forward(self, x):
        return self.rms_norm(x)


# Register layer normalization components
if RMSNorm is not None:
    normalization_registry.register(
        'rmsnorm',
        RMSNormWrapper,
        ComponentFamily.NORMALIZATION,
        {"description": "Root Mean Square Layer Normalization"}
    )

normalization_registry.register(
    'layernorm',
    LayerNormWrapper,
    ComponentFamily.NORMALIZATION,
    {"description": "Standard Layer Normalization"}
)


# ============================================================================
# Data Normalization Components
# ============================================================================

class TSNormalizerWrapper(BaseComponent):
    """Wrapper for TSNormalizer to fit component interface."""
    
    def __init__(self, config):
        super().__init__(config)
        self.mode = getattr(config, 'mode', 'standard')
        self.eps = getattr(config, 'eps', 1e-8)
        
        if TSNormalizer is not None:
            self.normalizer = TSNormalizer(mode=self.mode, eps=self.eps)
        else:
            raise ImportError("TSNormalizer not available")
    
    def fit(self, df):
        """Fit normalizer on DataFrame."""
        return self.normalizer.fit(df)
    
    def transform(self, df):
        """Transform DataFrame."""
        return self.normalizer.transform(df)
    
    def inverse_transform(self, df):
        """Inverse transform DataFrame."""
        return self.normalizer.inverse_transform(df)
    
    def forward(self, x):
        """For compatibility with BaseComponent interface."""
        # This would need adaptation based on input type
        return x


class NormalizationProcessorWrapper(BaseComponent):
    """Wrapper for NormalizationProcessor to fit component interface."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Convert config to dict format expected by NormalizationProcessor
        processor_config = {
            'type': getattr(config, 'normalization_type', 'standard'),
            'feature_wise': getattr(config, 'feature_wise', True),
            'eps': getattr(config, 'eps', 1e-8)
        }
        
        if NormalizationProcessor is not None:
            self.processor = NormalizationProcessor(processor_config)
        else:
            raise ImportError("NormalizationProcessor not available")
    
    def forward(self, x, fit=True):
        """Process input tensor."""
        return self.processor.forward(x, fit=fit)


# Register data normalization components
if TSNormalizer is not None:
    normalization_registry.register(
        'ts_normalizer',
        TSNormalizerWrapper,
        ComponentFamily.NORMALIZATION,
        {"description": "Time Series DataFrame Normalizer"}
    )

if NormalizationProcessor is not None:
    normalization_registry.register(
        'normalization_processor',
        NormalizationProcessorWrapper,
        ComponentFamily.NORMALIZATION,
        {"description": "Tensor-based Normalization Processor"}
    )


# ============================================================================
# Factory Functions
# ============================================================================

def create_layer_norm(d_model: int, eps: float = 1e-5, norm_type: str = 'layernorm') -> BaseComponent:
    """Create a layer normalization component.
    
    Args:
        d_model: Model dimension
        eps: Epsilon for numerical stability
        norm_type: Type of normalization ('layernorm' or 'rmsnorm')
        
    Returns:
        Normalization component
    """
    from dataclasses import dataclass
    
    @dataclass
    class NormConfig:
        d_model: int = d_model
        eps: float = eps
    
    config = NormConfig()
    
    if norm_type.lower() == 'rmsnorm' and RMSNorm is not None:
        return RMSNormWrapper(config)
    else:
        return LayerNormWrapper(config)


def create_data_normalizer(mode: str = 'standard', eps: float = 1e-8) -> BaseComponent:
    """Create a data normalization component.
    
    Args:
        mode: Normalization mode ('standard', 'minmax', etc.)
        eps: Epsilon for numerical stability
        
    Returns:
        Data normalization component
    """
    from dataclasses import dataclass
    
    @dataclass
    class NormConfig:
        mode: str = mode
        eps: float = eps
    
    config = NormConfig()
    
    if TSNormalizer is not None:
        return TSNormalizerWrapper(config)
    else:
        raise ImportError("TSNormalizer not available")


def create_tensor_normalizer(normalization_type: str = 'standard', 
                           feature_wise: bool = True, 
                           eps: float = 1e-8) -> BaseComponent:
    """Create a tensor normalization processor.
    
    Args:
        normalization_type: Type of normalization ('standard', 'minmax', 'robust')
        feature_wise: Whether to normalize per feature
        eps: Epsilon for numerical stability
        
    Returns:
        Tensor normalization component
    """
    from dataclasses import dataclass
    
    @dataclass
    class NormConfig:
        normalization_type: str = normalization_type
        feature_wise: bool = feature_wise
        eps: float = eps
    
    config = NormConfig()
    
    if NormalizationProcessor is not None:
        return NormalizationProcessorWrapper(config)
    else:
        raise ImportError("NormalizationProcessor not available")


# ============================================================================
# Registry Access Functions
# ============================================================================

def get_normalization_component(name: str) -> type:
    """Get a normalization component class by name.
    
    Args:
        name: Component name
        
    Returns:
        Component class
    """
    return normalization_registry.get_component(name)


def list_normalization_components() -> Dict[str, Dict[str, Any]]:
    """List all registered normalization components.
    
    Returns:
        Dictionary of component information
    """
    return normalization_registry.list_components()


def register_custom_normalization(name: str, component_class: type, description: str = ""):
    """Register a custom normalization component.
    
    Args:
        name: Component name
        component_class: Component class
        description: Component description
    """
    register_component(normalization_registry, name, component_class, description)


# ============================================================================
# Module Exports
# ============================================================================

__all__ = [
    'normalization_registry',
    'LayerNormWrapper',
    'RMSNormWrapper', 
    'TSNormalizerWrapper',
    'NormalizationProcessorWrapper',
    'create_layer_norm',
    'create_data_normalizer',
    'create_tensor_normalizer',
    'get_normalization_component',
    'list_normalization_components',
    'register_custom_normalization'
]


if __name__ == "__main__":
    # Example usage and testing
    print("Normalization Registry Components:")
    components = list_normalization_components()
    for name, info in components.items():
        print(f"  - {name}: {info.get('description', 'No description')}")
    
    print(f"\nTotal normalization components: {len(components)}")