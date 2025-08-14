# Backward-compatibility delegate for Autoformer_Improved
# The original improved implementation has been unified into models.Autoformer
# This delegate creates an Autoformer instance with Improved variant settings.

from .Autoformer import EnhancedAutoformer
import copy


class ImprovedAutoformer(EnhancedAutoformer):
    """Improved Autoformer variant delegate - enables input validation and keeps defaults for other flags."""
    def __init__(self, configs):
        configs_copy = copy.deepcopy(configs)
        configs_copy.use_input_validation = True
        super().__init__(configs_copy)

# Preserve old alias if some code expects Model
Model = ImprovedAutoformer