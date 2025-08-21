# Backward-compatibility shim module for EnhancedAutoformer
# The original implementation has been unified into models.Autoformer
# This shim re-exports symbols so existing imports continue to work.

from .Autoformer import (
    Model as EnhancedAutoformer,
    LearnableSeriesDecomp,
    EnhancedEncoder,
    EnhancedDecoder,
    Model as Model,
)

# Import layer classes from their actual modules
from layers.modular.layers.enhanced_layers import (
    EnhancedEncoderLayer,
    EnhancedDecoderLayer,
)

# Expose StableSeriesDecomp if available, with fallback to LearnableSeriesDecomp
try:  # pragma: no cover - optional import
    from layers.modular.decomposition.stable_decomposition import (
        StableSeriesDecomposition as StableSeriesDecomp,  # type: ignore
    )
except Exception:  # pragma: no cover - fallback path
    # Fall back to LearnableSeriesDecomp when stable variant isn't available
    StableSeriesDecomp = LearnableSeriesDecomp  # type: ignore

__all__ = [
    "EnhancedAutoformer",
    "LearnableSeriesDecomp",
    "EnhancedEncoderLayer",
    "EnhancedDecoderLayer",
    "EnhancedEncoder",
    "EnhancedDecoder",
    "Model",
    "StableSeriesDecomp",
]