# Backward-compatibility delegate for EnhancedAutoformer_Fixed
# The functionality has been unified into models.Autoformer
# This delegate enables stable decomposition and gradient scaling features.

from .Autoformer import EnhancedAutoformer, LearnableSeriesDecomp
import copy

# Try to import StableSeriesDecomp from the legacy module if still present
try:
    from .Autoformer import StableSeriesDecomp  # exported conditionally in Autoformer.py
except Exception:
    StableSeriesDecomp = LearnableSeriesDecomp


class EnhancedAutoformer(EnhancedAutoformer):
    """Fixed EnhancedAutoformer variant delegate - stability features enabled."""
    def __init__(self, configs):
        configs_copy = copy.deepcopy(configs)
        configs_copy.use_stable_decomp = True
        configs_copy.use_gradient_scaling = True
        configs_copy.use_input_validation = True
        super().__init__(configs_copy)

# Maintain Model alias
Model = EnhancedAutoformer