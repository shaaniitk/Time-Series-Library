from .registry import FusionRegistry, get_fusion_component

# Compatibility re-exports for processor tests now point to local wrappers
try:
    from layers.modular.processor.wrapped_fusions import (  # type: ignore
        register_layers_fusions,
        FusionProcessorConfig,
    )
except Exception:  # pragma: no cover
    register_layers_fusions = None  # type: ignore
    FusionProcessorConfig = None  # type: ignore

__all__ = [
    "FusionRegistry",
    "get_fusion_component",
    # Re-exported for tests
    "register_layers_fusions",
    "FusionProcessorConfig",
]
