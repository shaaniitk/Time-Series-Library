# Backward-compatibility shim module for EnhancedAutoformer
# The original implementation has been unified into models.Autoformer
# This shim re-exports symbols so existing imports continue to work.

from .Autoformer import (
    EnhancedAutoformer,
    LearnableSeriesDecomp,
    EnhancedEncoderLayer as EnhancedEncoderLayer,
    EnhancedDecoderLayer as EnhancedDecoderLayer,
    EnhancedEncoder as EnhancedEncoder,
    EnhancedDecoder as EnhancedDecoder,
    Model as Model,
)

# Expose StableSeriesDecomp if available, with fallback to LearnableSeriesDecomp
try:  # pragma: no cover - optional import
    from .Autoformer import StableSeriesDecomp as StableSeriesDecomp
except Exception:  # pragma: no cover - fallback path
    from .Autoformer import LearnableSeriesDecomp as StableSeriesDecomp

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