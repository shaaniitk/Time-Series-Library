"""
UNIFIED PROCESSOR COMPONENTS
All processor mechanisms in one place - clean modular structure
"""
import torch.nn as nn
from typing import Optional

# Import all required models
from ....models import (
    Autoformer,
    DLinear,
    Transformer,
    Crossformer,
    MambaSimple,
    SCINet,
    SegRNN,
    TimesNet,
    ETSformer,
    Pyraformer
)

# Import the new base interface and config schema
from ..base_interfaces import BaseProcessor as NewBaseProcessor
from ..config_schemas import ComponentConfig, safe_config_from_dict
from ....utils.logger import logger


class BaseProcessor(NewBaseProcessor):
    """
    Base class for processor components, aligned with the new modular framework.
    A processor in this context defines the complete forward pass logic of a model.
    """
    def __init__(self, configs):
        # The configs object here is expected to be a Namespace or a class with attributes.
        # We'll adapt it to the ComponentConfig structure if needed.
        if not isinstance(configs, ComponentConfig):
            # Attempt to convert if it's a dict-like object (e.g., Namespace)
            try:
                config_dict = vars(configs)
                comp_config = safe_config_from_dict(ComponentConfig, config_dict)
            except TypeError:
                # Fallback for objects that are not dict-like
                comp_config = ComponentConfig()
        else:
            comp_config = configs
            
        super().__init__(comp_config)
        self.model = None

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.model is None:
            raise NotImplementedError("The model is not initialized in the base processor.")
        return self.model.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask=mask)

    def process_sequence(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None, **kwargs):
        return self.forward(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
    
    def get_output_dim(self) -> int:
        # This should be defined by the specific model's output dimension (c_out)
        if hasattr(self.config, 'c_out'):
            return self.config.c_out
        return 1 # Fallback

# --- Processor Implementations ---

class AutoformerProcessor(BaseProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.model = Autoformer(configs)

class DLinearProcessor(BaseProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.model = DLinear(configs)

class TransformerProcessor(BaseProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.model = Transformer(configs)

class CrossformerProcessor(BaseProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.model = Crossformer(configs)

class MambaSimpleProcessor(BaseProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.model = MambaSimple(configs)

class SCINetProcessor(BaseProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.model = SCINet(configs)

class SegRNNProcessor(BaseProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.model = SegRNN(configs)

class TimesNetProcessor(BaseProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.model = TimesNet(configs)

class ETSformerProcessor(BaseProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.model = ETSformer(configs)

class PyraformerProcessor(BaseProcessor):
    def __init__(self, configs):
        super().__init__(configs)
        self.model = Pyraformer(configs)

# --- Registry and Factory ---

PROCESSOR_REGISTRY = {
    "autoformer": AutoformerProcessor,
    "dlinear": DLinearProcessor,
    "transformer": TransformerProcessor,
    "crossformer": CrossformerProcessor,
    "mambasimple": MambaSimpleProcessor,
    "scinet": SCINetProcessor,
    "segrnn": SegRNNProcessor,
    "timesnet": TimesNetProcessor,
    "etsformer": ETSformerProcessor,
    "pyraformer": PyraformerProcessor,
}

def get_processor_component(name: str, configs, **kwargs):
    """Factory function to create processor components."""
    if name not in PROCESSOR_REGISTRY:
        raise ValueError(f"Processor '{name}' not found in registry. Available: {list(PROCESSOR_REGISTRY.keys())}")
    
    component_class = PROCESSOR_REGISTRY[name]
    # Pass configs directly as these processors expect the Namespace object
    return component_class(configs=configs, **kwargs)

def register_processor_components(registry):
    """Register all processor components with the main registry."""
    for name, component_class in PROCESSOR_REGISTRY.items():
        registry.register('processor', name, component_class)
    logger.info(f"Registered {len(PROCESSOR_REGISTRY)} processor components.")

def list_processor_components():
    """List all available processor components."""
    return list(PROCESSOR_REGISTRY.keys())

logger.info("✅ Unified processor components loaded successfully.")