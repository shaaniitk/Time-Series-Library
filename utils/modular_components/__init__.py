"""
Modular Components for Time Series Models

This package provides abstract interfaces and base classes for building
modular, reusable time series model components that can be mixed and matched
across different architectures (HF Autoformer, Transformer, etc.).
"""

# Core interfaces
from .base_interfaces import (
    BaseComponent,
    BaseBackbone,
    BaseEmbedding,
    BaseAttention,
    BaseProcessor,
    BaseFeedForward,
    BaseLoss,
    BaseOutput,
    BaseAdapter
)

# Registry and factory systems
from .registry import ComponentRegistry
from .factory import ComponentFactory, get_factory

# Model builder
from .model_builder import ModelBuilder, ModelConfig, ModularModel

# Auto-registration
from .register_new_components import auto_register_all

__all__ = [
    # Base interfaces
    'BaseComponent', 'BaseBackbone', 'BaseEmbedding', 'BaseAttention',
    'BaseProcessor', 'BaseFeedForward', 'BaseLoss', 'BaseOutput', 'BaseAdapter',
    
    # Registry and factory
    'ComponentRegistry', 'ComponentFactory', 'get_factory',
    
    # Model builder
    'ModelBuilder', 'ModelConfig', 'ModularModel',
    
    # Auto-registration
    'auto_register_all'
]

# Automatically register all components when package is imported
auto_register_all()
