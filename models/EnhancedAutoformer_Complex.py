# Backward-compatibility delegate for EnhancedAutoformer_Complex
# The functionality has been unified into models.Autoformer
# This delegate simply forwards to the unified EnhancedAutoformer implementation.

from .Autoformer import EnhancedAutoformer as _UnifiedEnhancedAutoformer


class EnhancedAutoformer(_UnifiedEnhancedAutoformer):
    pass

# Maintain Model alias for compatibility
Model = EnhancedAutoformer
