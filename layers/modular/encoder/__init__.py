from .abstract_encoder import BaseEncoder
from .base_encoder import ModularEncoder
from .standard_encoder import StandardEncoder
from .enhanced_encoder import EnhancedEncoder
from .stable_encoder import StableEncoder
from .hierarchical_encoder import HierarchicalEncoder
from .registry import EncoderRegistry, get_encoder_component

# Compatibility re-exports for processor tests now point to local wrappers
try:
    from layers.modular.processor.wrapped_encoders import (  # type: ignore
        register_layers_encoders,
        EncoderProcessorConfig,
    )
except Exception:  # pragma: no cover
    register_layers_encoders = None  # type: ignore
    EncoderProcessorConfig = None  # type: ignore

__all__ = [
    "BaseEncoder",
    "ModularEncoder",
    "StandardEncoder",
    "EnhancedEncoder",
    "StableEncoder",
    "HierarchicalEncoder",
    "EncoderRegistry",
    "get_encoder_component",
    # Re-exported for tests
    "register_layers_encoders",
    "EncoderProcessorConfig",
]