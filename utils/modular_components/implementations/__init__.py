"""
Concrete Implementations of Modular Components

This package contains concrete implementations of all the abstract
component interfaces for building modular time series models.
"""

# Import implementation modules with fallbacks
try:
    from . import backbones
    BACKBONES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import advanced backbones: {e}")
    BACKBONES_AVAILABLE = False

# Always import simple backbones (no external dependencies)
from . import simple_backbones

from . import embeddings  
from . import attentions

__all__ = [
    'simple_backbones',
    'embeddings', 
    'attentions'
]

if BACKBONES_AVAILABLE:
    __all__.append('backbones')
