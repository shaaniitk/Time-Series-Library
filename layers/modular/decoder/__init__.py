
from .base import BaseDecoder
from .standard_decoder import StandardDecoder
from .enhanced_decoder import EnhancedDecoder
from .stable_decoder import StableDecoder
from .registry import DecoderRegistry, get_decoder_component

# Compatibility re-exports for processor tests point to local wrappers
try:
    from ..processor.wrapped_decoders import (
        register_layers_decoders,
        DecoderProcessorConfig,
    )
except Exception:  # pragma: no cover
    register_layers_decoders = None  # type: ignore
    DecoderProcessorConfig = None  # type: ignore
from .decoder_output import DecoderOutput, standardize_decoder_output
from .unified_interface import UnifiedDecoderInterface, DecoderFactory
from .validation import ComponentValidator, DecoderTestRunner
from .core_decoders import CoreAutoformerDecoder, CoreEnhancedDecoder

__all__ = [
    "BaseDecoder",
    "StandardDecoder",
    "EnhancedDecoder",
    "StableDecoder",
    "DecoderRegistry",
    "get_decoder_component",
    # Re-exported for tests
    "register_layers_decoders",
    "DecoderProcessorConfig",
    "DecoderOutput",
    "standardize_decoder_output",
    "UnifiedDecoderInterface",
    "DecoderFactory",
    "ComponentValidator",
    "DecoderTestRunner",
    "CoreAutoformerDecoder",
    "CoreEnhancedDecoder",
]
