# Backward-compatibility delegate for Autoformer_Fixed
# The original implementation has been unified into models.Autoformer
# This delegate creates an Autoformer instance with Fixed variant settings.

from .Autoformer import EnhancedAutoformer
import copy


class Model(EnhancedAutoformer):
    """Fixed Autoformer variant delegate - uses unified Autoformer with stability flags enabled."""
    
    def __init__(self, configs):
        # Make a copy to avoid mutating the original config
        configs_copy = copy.deepcopy(configs)
        
        # Enable Fixed variant features
        configs_copy.use_stable_decomp = True
        configs_copy.use_gradient_scaling = True
        configs_copy.use_input_validation = True
        
        # Initialize the unified model with Fixed settings
        super().__init__(configs_copy)