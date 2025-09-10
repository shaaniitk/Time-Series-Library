"""Modular components for time series models.

This package provides modular components that can be mixed and matched
to build different time series architectures.
"""

# Import all modular components
from .attention import *
from .decomposition import *
from .backbone import *
from .feedforward import *
from .output import *
from .output_heads import *
from .core import *

# Import and initialize the component registry
from .core.register_components import component_registry

__all__ = [
    # Core registry
    'component_registry',
    # Component families will be imported via their respective modules
]